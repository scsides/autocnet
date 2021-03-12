from collections import defaultdict, MutableMapping, Counter
from functools import wraps, singledispatch
import json
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist
from shapely.geometry import Point
import sqlalchemy

from autocnet.graph.node import Node
from autocnet.utils import utils
from autocnet.matcher import cpu_outlier_detector as od
from autocnet.matcher import suppression_funcs as spf
from autocnet.matcher import subpixel as sp
from autocnet.matcher import cpu_ring_matcher
from autocnet.transformation import fundamental_matrix as fm
from autocnet.transformation import homography as hm
from autocnet.transformation.spatial import reproject, og2oc
from autocnet.vis.graph_view import plot_edge, plot_node, plot_edge_decomposition, plot_matches
from autocnet.cg import cg
from autocnet.io.db.model import Images, Keypoints, Matches,\
                                 Cameras, Base, Overlay, Edges,\
                                 Costs, Measures, Points, Measures
from autocnet.io.db.wrappers import DbDataFrame

from plio.io.io_gdal import GeoDataset
from csmapi import csmapi

class Edge(dict, MutableMapping):
    """
    Attributes
    ----------
    source : hashable
             The source node

    destination : hashable
                  The destination node
    masks : set
            A list of the available masking arrays

    weights : dict
             Dictionary with two keys overlap_area, and overlap_percn
             overlap_area returns the area overlaped by both images
             overlap_percn retuns the total percentage of overlap
    """

    def __init__(self, source=None, destination=None):
        self.source = source
        self.destination = destination
        self['homography'] = None
        self['fundamental_matrix'] = None
        self.subpixel_matches = pd.DataFrame()
        self._matches = pd.DataFrame()

        self['source_mbr'] = None
        self['destin_mbr'] = None
        self['overlap_latlon_coords'] = None

    def __repr__(self):
        return """
        Source Image Index: {}
        Destination Image Index: {}
        Available Masks: {}
        """.format(self.source, self.destination, self.masks)

    def __eq__(self, other):
        return utils.compare_dicts(self.__dict__, other.__dict__) *\
               utils.compare_dicts(self, other)

    @property
    def weights(self):
        if not hasattr(self, '_weights'):
            self._weights = {}
        return self._weights

    @weights.setter
    def weights(self, kv):
        key, value = kv
        self.weights[key] = value

    @property
    def masks(self):
        if not hasattr(self, '_masks'):
            self._masks = pd.DataFrame()
        return self._masks

    @masks.setter
    def masks(self, value):
        if isinstance(value, pd.DataFrame):
            self._masks = value
        else:
            raise(TypeError)

    @property
    def matches(self):
        if not hasattr(self, '_matches'):
            self._matches = pd.DataFrame()
        return self._matches

    @matches.setter
    def matches(self, value):
        if isinstance(value, pd.DataFrame):
            self._matches = value
            # Ensure that the costs df remains in sync with the matches df
            if not self.costs.index.equals(value.index):
                self.costs = pd.DataFrame(index=value.index)
        else:
            raise(TypeError)

    @property
    def costs(self):
        if not hasattr(self, '_costs'):
            self._costs = pd.DataFrame(index=self.matches.index)
        return self._costs

    @costs.setter
    def costs(self, value):
        if isinstance(value, pd.DataFrame):
            self._costs = value
        else:
            raise(TypeError)

    @property
    def ring(self):
        if not hasattr(self, '_ring'):
            self._ring = None
        return self._ring

    @ring.setter
    def ring(self, val):
        self._ring = val

    def match(self, k=2, **kwargs):

        """
        Given two sets of descriptors, utilize a FLANN (Approximate Nearest
        Neighbor KDTree) matcher to find the k nearest matches.  Nearness is
        the euclidean distance between descriptors.

        The matches are then added as an attribute to the edge object.

        Parameters
        ----------
        k : int
            The number of neighbors to find
        """
        # Reset the edge masks because matching is happening (again)
        self.masks = pd.DataFrame()
        kwargs['aidx'] = self.get_keypoints('source', overlap=True).index
        kwargs['bidx'] = self.get_keypoints('destination', overlap=True).index
        Edge._match(self, k=k, **kwargs)

    @staticmethod
    def _match(edge, k=2, **kwargs):
        """
        Patches the static cpu_matcher.match(edge) or cuda_match.match(edge)
        into the member method Edge.match()

        Parameters
        ----------
        edge : Edge
               The edge object to compute matches for; Edge.match() calls this
               with self
        k : int
            The number of neighbors to find
        """
        pass

    def ring_match(self, *args, **kwargs):
        ref_kps =  self.source.keypoints
        ref_desc = self.source.descriptors
        tar_kps = self.destination.keypoints
        tar_desc = self.destination.descriptors

        if not 'xm' in ref_kps.columns:
            warnings.warn('To ring match body centered coordinates (xm, ym, zm) must be in the keypoints')
            return
        ref_feats = ref_kps[['x', 'y', 'xm', 'ym', 'zm']].values
        tar_feats = tar_kps[['x', 'y', 'xm', 'ym', 'zm']].values

        _, _, pidx, ring = cpu_ring_matcher.ring_match(ref_feats, tar_feats,
                                                           ref_desc, tar_desc,
                                                           *args, **kwargs)

        if pidx is None:
            return
        self.ring = ring
        pidx = cpu_ring_matcher.check_pidx_duplicates(pidx)

        #Set the columns of the matches df
        matches = np.empty((pidx.shape[0], 4))
        matches[:,0] = self.source['node_id']
        matches[:,1] = ref_kps.index[pidx[:,0]].values
        matches[:,2] = self.destination['node_id']
        matches[:,3] = tar_kps.index[pidx[:,1]].values

        matches = pd.DataFrame(matches, columns=['source',
                                                 'source_idx',
                                                 'destination',
                                                 'destination_idx']).astype(np.float32)

        matches = matches.drop_duplicates()

        self.matches = matches

    def add_coordinates_to_matches(self):
        """
        Add source and destination x/y columns to the matches dataframe. This
        will add to the overall memory needed to store matches, but makes
        access to x,y easier as a join on the keypoints is not requires.
        """
        skps = self.get_keypoints(self.source, index=self.matches.source_idx)
        skps.reindex(self.matches['source_idx'])
        self.matches['source_x'] = skps.values[:,0]
        self.matches['source_y'] = skps.values[:,1]
        dkps = self.get_keypoints(self.destination, index=self.matches.destination_idx)
        dkps.reindex(self.matches['destination_idx'])
        self.matches['destination_x'] = dkps.values[:,0]
        self.matches['destination_y'] = dkps.values[:,1]

    def project_matches(self, semimajor, semiminor, on='source', srid=None):
        """
        Project matches.
        """
        try:
            coords = self.matches[['{}_y'.format(on),'{}_x'.format(on)]].values
        except:
            self.add_coordinates_to_matches()
            coords = self.matches[['{}_y'.format(on),'{}_x'.format(on)]].values

        node = getattr(self, on)
        camera = getattr(node, 'camera')
        if camera is None:
            warnings.warn('Unable to project matches without a sensor model.')
            return

        matches = self.matches

        gnd = np.empty((len(coords), 3))
        # Project the points to the surface and reproject into latlon space
        for i in range(gnd.shape[0]):
            ic = csmapi.ImageCoord(coords[i][0], coords[i][1])
            ground = camera.imageToGround(ic, 0)
            gnd[i] = [ground.x, ground.y, ground.z]
        lon_og, lat_og, alt = reproject(gnd.T, semimajor, semiminor,
                                    'geocent', 'latlon')
        lon, lat = og2oc(lon_og, lat_og, semimajor, semiminor)

        if srid:
            geoms = []
            for coord in zip(lon, lat, alt):
                geoms.append('SRID={};POINTZ({} {} {})'.format(srid, coord[0],
                                                                     coord[1],
                                                                     coord[2]))
            matches['geom'] = geoms

        matches['lat'] = lat
        matches['lon'] = lon
        self.matches = matches

    def decompose(self):
        """
        Apply coupled decomposition to the images and
        match identified sub-images
        """
        pass

    def decompose_and_match(*args, **kwargs):
        pass

    def overlap_check(self):
        """Creates a mask for matches on the overlap"""
        if not (self["source_mbr"] and self["destin_mbr"]):
            warnings.warn(
                "Cannot use overlap constraint, minimum bounding rectangles"
                " have not been computed for one or more Nodes")
            return
        # Get overlapping keypts
        s_idx = self.get_keypoints(self.source, overlap=True).index
        d_idx = self.get_keypoints(self.destination, overlap=True).index
        # Create a mask from matches whose rows have both source idx &
        # dest idx in the overlapping keypts
        mask = pd.Series(False, index=self.matches.index)
        mask.loc[(self.matches["source_idx"].isin(s_idx)) &
                 (self.matches["destination_idx"].isin(d_idx))] = True
        self.masks['overlap'] = mask

    def symmetry_check(self, clean_keys=[], maskname='symmetry', **kwargs):
        self.masks[maskname] = od.mirroring_test(self.matches)

    def ratio_check(self, clean_keys=[], maskname='ratio', **kwargs):
        matches, mask = self.clean(clean_keys)
        self.masks[maskname] = self._ratio_check(self, matches, **kwargs)

    @staticmethod
    def _ratio_check(edge, matches, **kwargs):
        pass
        #return.masks[maskname] = od.distance_ratio(matches, **kwargs)

    @utils.methodispatch
    def get_keypoints(self, node, index=None, homogeneous=False, overlap=False):
        if not hasattr(index, '__iter__') and index is not None:
            raise TypeError
        keypts = node.get_keypoint_coordinates(index=index, homogeneous=homogeneous)
        # If the index is passed, the results are returned sorted. The index is not
        # necessarily sorted, so 'unsort' so that the return order matches the passed
        # order
        if index is not None:
            keypts = keypts.reindex(index)
        # If we only want keypoints in the overlap
        if overlap:
            if self.source == node:
                mbr = self['source_mbr']
            else:
                mbr = self['destin_mbr']
            # Can't use overlap if we haven't computed MBRs
            if mbr is None:
                return keypts
            return keypts.query('x >= {} and x <= {} and y >= {} and y <= {}'.format(*mbr))
        return keypts

    @get_keypoints.register(str)
    def _(self, node, index=None, homogeneous=False, overlap=False):
        if not hasattr(index, '__iter__') and index is not None:
            raise TypeError
        node = node.lower()
        node = getattr(self, node)
        return self.get_keypoints(node, index=index, homogeneous=homogeneous, overlap=overlap)

    def compute_fundamental_matrix(self, clean_keys=[], maskname='fundamental', **kwargs):
        """
        Estimate the fundamental matrix (F) using the correspondences tagged to this
        edge.


        Parameters
        ----------
        clean_keys : list
                     Of strings used to apply masks to omit correspondences

        method : {linear, nonlinear}
                 Method to use to compute F.  Linear is significantly faster at
                 the cost of reduced accuracy.

        See Also
        --------
        autocnet.transformation.transformations.FundamentalMatrix

        """
        _, mask = self.clean(clean_keys)
        s_keypoints, d_keypoints = self.get_match_coordinates(clean_keys=clean_keys)
        self.fundamental_matrix, fmask = fm.compute_fundamental_matrix(s_keypoints, d_keypoints, **kwargs)
        fmask = fmask.flatten()

        if isinstance(self.fundamental_matrix, np.ndarray):
            # Convert the truncated RANSAC mask back into a full length mask
            mask[mask] = fmask
            # Set the initial state of the fundamental mask in the masks
            self.masks[maskname] = mask

    def compute_fundamental_error(self, method='equality', clean_keys=[]):
        """
        Given a fundamental matrix, compute the reprojective error between
        a two sets of keypoints.

        Parameters
        ----------
        clean_keys : list
                     of string keys to masking arrays
                     (created by calling outlier detection)

        Returns
        -------
        error : pd.Series
                of reprojective error indexed to the matches data frame
        """
        if self.fundamental_matrix is None:
            warnings.warn('No fundamental matrix has been compute for this edge.')

        matches, mask = self.clean(clean_keys)
        s_keypoints, d_keypoints = self.get_match_coordinates(clean_keys=clean_keys)
        if method == 'equality':
            error = fm.compute_fundamental_error(self.fundamental_matrix, s_keypoints, d_keypoints)
        elif method == 'projection':
            error = fm.compute_reprojection_error(self.fundamental_matrix, s_keypoints, d_keypoints)

        self.costs.loc[mask, 'fundamental_{}'.format(method)] = error

    def compute_homography(self, method='ransac', clean_keys=[], pid=None, maskname='homography', **kwargs):
        """
        For each edge in the (sub) graph, compute the homography

        Parameters
        ----------
        outlier_algorithm : object
                            An openCV outlier detections algorithm, e.g. cv2.RANSAC

        clean_keys : list
                     of string keys to masking arrays
                     (created by calling outlier detection)
        Returns
        -------
        transformation_matrix : ndarray
                                The 3x3 transformation matrix

        mask : ndarray
               Boolean array of the outliers
        """
        matches, mask = self.clean(clean_keys)

        s_keypoints = self.source.get_keypoint_coordinates(index=matches['source_idx'])
        d_keypoints = self.destination.get_keypoint_coordinates(index=matches['destination_idx'])

        self['homography'], hmask = hm.compute_homography(s_keypoints.values, d_keypoints.values, method=method)

        # Convert the truncated RANSAC mask back into a full length mask
        mask[mask] = hmask
        self.masks['homography'] = mask

    def subpixel_register(self, method='phase', clean_keys=[],
                          template_size=251, search_size=251, **kwargs):
        """
        For the entire graph, compute the subpixel offsets using pattern-matching and add the result
        as an attribute to each edge of the graph.

        Parameters
        ----------
        clean_keys : list
                     of string keys to masking arrays
                     (created by calling outlier detection)

        threshold : float
                    On the range [-1, 1].  Values less than or equal to
                    this threshold are masked and can be considered
                    outliers

        upsampling : int
                     The multiplier to the template and search shapes to upsample
                     for subpixel accuracy

        template_size : int
                        The size of the template in pixels, must be odd. If using phase,
                        only the template size is used.

        search_size : int
                      The size of the search area. When method='template', this size should
                      be >= the template size

        """
        # Build up a composite mask from all of the user specified masks
        matches, mask = self.clean(clean_keys)

        # Get the img handles
        s_img = self.source.geodata
        d_img = self.destination.geodata

        # Determine which algorithm is going ot be used.
        if method == 'phase':
            func = sp.iterative_phase
            nstrengths = 2
        elif method == 'template':
            func = sp.subpixel_template
            nstrengths = 1
        shifts_x, shifts_y, strengths, new_x, new_y = sp._prep_subpixel(len(matches), nstrengths)

        # for each edge, calculate this for each keypoint pair
        for i, (idx, row) in enumerate(matches.iterrows()):
            s_idx = int(row['source_idx'])
            d_idx = int(row['destination_idx'])

            if 'source_x' in row.index:
                sx = row.source_x
                sy = row.source_y
            else:
                s_keypoint = self.source.get_keypoint_coordinates([s_idx])
                sx = s_keypoint.x
                sy = s_keypoint.y

            if 'destination_x' in row.index:
                dx = row.destination_x
                dy = row.destination_y
            else:
                d_keypoint = self.destination.get_keypoint_coordinates([d_idx])
                dx = d_keypoint.x
                dy = d_keypoint.y

            if method == 'phase':
                res = sp.iterative_phase(sx, sy, dx, dy, s_img, d_img, size=template_size, **kwargs)
                if res[0]:
                    new_x[i] = res[0]
                    new_y[i] = res[1]
                    strengths[i] = res[2]
            elif method == 'template':
                new_x[i], new_y[i], strengths[i], _ = sp.subpixel_template(sx, sy, dx, dy, s_img, d_img,
                                                                     search_size=search_size,
                                                                     template_size=template_size, **kwargs)

            # Capture the shifts
            shifts_x[i] = new_x[i] - dx
            shifts_y[i] = new_y[i] - dy

        self.matches.loc[mask, 'shift_x'] = shifts_x
        self.matches.loc[mask, 'shift_y'] = shifts_y
        self.matches.loc[mask, 'destination_x'] = new_x
        self.matches.loc[mask, 'destination_y'] = new_y

        if method == 'phase':
            self.costs.loc[mask, 'phase_diff'] = strengths[:,0]
            self.costs.loc[mask, 'rmse'] = strengths[:,1]
        elif method == 'template':
            self.costs.loc[mask, 'correlation'] = strengths[:,0]


    def suppress(self, suppression_func=spf.correlation, clean_keys=[], maskname='suppression', **kwargs):
        """
        Apply a disc based suppression algorithm to get a good spatial
        distribution of high quality points, where the user defines some
        function to be used as the quality metric.

        Parameters
        ----------
        suppression_func : object
                           A function that returns a scalar value to be used
                           as the strength of a given row in the matches data
                           frame.

        suppression_args : tuple
                           Arguments to be passed on to the suppression function

        clean_keys : list
                     of mask keys to be used to reduce the total size
                     of the matches dataframe.
        """
        if not isinstance(self.matches, pd.DataFrame):
            raise AttributeError('This edge does not yet have any matches computed.')

        matches, mask = self.clean(clean_keys)
        rs = self.source.geodata.raster_size
        domain = [0, 0, rs[0], rs[1]]
        # Massage the dataframe into the correct structure
        coords = self.source.get_keypoint_coordinates()
        merged = matches.merge(coords, left_on=['source_idx'], right_index=True)
        merged['strength'] = merged.apply(suppression_func, axis=1, args=([self]))

        smask, k = od.spatial_suppression(merged, domain, **kwargs)

        mask[mask] = smask
        self.masks[maskname] = mask

    def plot_source(self, ax=None, clean_keys=[], **kwargs):  # pragma: no cover
        matches, mask = self.clean(clean_keys=clean_keys)
        indices = pd.Index(matches['source_idx'].values)
        return plot_node(self.source, index_mask=indices, **kwargs)

    def plot_matches(self, clean_keys=[], **kwargs):  # pragme: no cover
        matches, mask = self.clean(clean_keys=clean_keys)
        sourcegd = self.source.geodata
        destingd = self.destination.geodata
        return plot_matches(matches, sourcegd, destingd, **kwargs)

    def plot_destination(self, ax=None, clean_keys=[], **kwargs):  # pragma: no cover
        matches, mask = self.clean(clean_keys=clean_keys)
        indices = pd.Index(matches['destination_idx'].values)
        return plot_node(self.destination, index_mask=indices, **kwargs)

    def plot(self, ax=None, clean_keys=[], node=None, **kwargs):  # pragma: no cover
        dest_keys = [0, '0', 'destination', 'd', 'dest']
        source_keys = [1, '1', 'source', 's']

        # If node is not none, plot a single node
        if node in source_keys:
            return self.plot_source(self, clean_keys=clean_keys, **kwargs)

        elif node in dest_keys:
            return self.plot_destination(self, clean_keys=clean_keys, **kwargs)

        # Else, plot the whole edge
        return plot_edge(self, ax=ax, clean_keys=clean_keys, **kwargs)

    def plot_decomposition(self, *args, **kwargs): #pragma: no cover
        return plot_edge_decomposition(self, *args, **kwargs)

    def clean(self, clean_keys):
        """
        Given a list of clean keys compute the mask of valid
        matches

        Parameters
        ----------
        clean_keys : list
                     of columns names (clean keys)

        Returns
        -------
        matches : dataframe
                  A masked view of the matches dataframe

        mask : series
               A boolean series to inflate back to the full match set
        """
        if clean_keys:
            mask = self.masks[clean_keys].all(axis=1)
        else:
            mask = pd.Series(True, self.matches.index)

        m = mask[mask==True]
        return self.matches.loc[m.index], mask

    def overlap(self):
        """
        Acts on an edge and returns the overlap area and percentage of overlap
        between the two images on the edge. Data is returned to the
        weights dictionary
        """
        poly1 = self.source.geodata.footprint
        poly2 = self.destination.geodata.footprint

        overlapinfo = cg.two_poly_overlap(poly1, poly2)

        self.weights = ('overlap_area', overlapinfo[1])
        self.weights = ('overlap_percn', overlapinfo[0])

    def coverage(self, clean_keys = []):
        """
        Acts on the edge given either the source node
        or the destination node and returns the percentage
        of overlap covered by the keypoints. Data for the
        overlap is gathered from the source node of the edge
        resulting in a maximum area difference of 2% when compared
        to the destination.

        Returns
        -------
        total_overlap_percentage : float
                                   returns the overlap area
                                   covered by the keypoints
        """
        matches, mask = self.clean(clean_keys)
        source_array = self.source.get_keypoint_coordinates(index=matches['source_idx']).values

        source_coords = self.source.geodata.latlon_corners
        destination_coords = self.destination.geodata.latlon_corners

        convex_hull = cg.convex_hull(source_array)

        convex_points = [self.source.geodata.pixel_to_latlon(row[0], row[1]) for row in convex_hull.points[convex_hull.vertices]]
        convex_coords = [(x, y) for x, y in convex_points]

        source_poly = utils.array_to_poly(source_coords)
        destination_poly = utils.array_to_poly(destination_coords)
        convex_poly = utils.array_to_poly(convex_coords)

        intersection_area = cg.get_area(source_poly, destination_poly)

        total_overlap_coverage = (convex_poly.GetArea()/intersection_area)

        return total_overlap_coverage

    def compute_weights(self, clean_keys, **kwargs):
        """
        Computes a voronoi diagram for the overlap between two images
        then gets the area of each polygon resulting in a voronoi weight.
        These weights are then appended to the matches dataframe.

        Parameters
        ----------
        clean_keys : list
                     Of strings used to apply masks to omit correspondences
        """
        if not isinstance(self.matches, pd.DataFrame):
            raise AttributeError('Matches have not been computed for this edge')
        voronoi = cg.compute_voronoi(self, clean_keys, **kwargs)
        self.matches = pd.concat([self.matches, voronoi[1]['vor_weights']], axis=1)

    def compute_overlap(self, buffer_dist=0, **kwargs):
        """
        Estimate a source and destination minimum bounding rectangle, in
        pixel space.
        """
        if not isinstance(self.source.geodata, GeoDataset):
            smbr = None
            dmbr = None
        else:
            try:
                self['overlap_latlon_coords'], smbr, dmbr = self.source.geodata.compute_overlap(self.destination.geodata, **kwargs)
                smbr = list(smbr)
                dmbr = list(dmbr)
                for i in range(4):
                    if i % 2:
                        buf = buffer_dist
                    else:
                        buf = -buffer_dist
                    smbr[i] += buf
                    dmbr[i] += buf

            except:
                smbr = self.source.geodata.xy_extent
                dmbr = self.source.geodata.xy_extent
                warnings.warn("Overlap between {} and {} could not be "
                                "computed.  Using the full image extents".format(self.source['image_name'],
                                                      self.destination['image_name']))
                smbr = [smbr[0][0], smbr[1][0], smbr[0][1], smbr[1][1]]
                dmbr = [dmbr[0][0], dmbr[1][0], dmbr[0][1], dmbr[1][1]]
        self['source_mbr'] = smbr
        self['destin_mbr'] = dmbr

    def get_match_coordinates(self, clean_keys=[]):
        matches = self.get_matches(clean_keys=clean_keys)

        # no matches, return empty dataframe
        if matches.empty:
            return pd.DataFrame(), pd.DataFrame()

        skps = matches[['source_x', 'source_y']].astype(np.float)
        dkps = matches[['destination_x', 'destination_y']].astype(np.float)

        return skps, dkps

    def get_matches(self, clean_keys=[]): # pragma: no cover
        if self.matches.empty:
            return pd.DataFrame()
        #self.add_coordinates_to_matches()
        matches, _ = self.clean(clean_keys=clean_keys)
        skps = matches[['source_x', 'source_y']]
        dkps = matches[['destination_x', 'destination_y']]
        return matches

class NetworkEdge(Edge):

    default_msg = {'sidx':None,
                    'didx':None,
                    'task':None,
                    'param_step':0,
                    'success':False}

    def __init__(self, *args, **kwargs):
        super(NetworkEdge, self).__init__(*args, **kwargs)
        self.job_status = defaultdict(dict)

    def _from_db(self, table_obj):
        """
        Generic database query to pull the row associated with this node
        from an arbitrary table. We assume that the row id matches the node_id.

        Parameters
        ----------
        table_obj : object
                    The declared table class (from db.model)
        """
        with self.parent.session_scope() as session:
            res = session.query(table_obj).\
                   filter(table_obj.source == self.source['node_id']).\
                   filter(table_obj.destination == self.destination['node_id'])
            session.expunge_all()
        return res

    def compute_homography(self, method='ransac', maskname='homography', **kwargs):
        """
        Estimate the homography and reprojective error on this edge of the graph.

        Parameters
        ----------
        method : {'ransac', 'lmeds', 'normal'}
                 The method that will be used when computing the homography.

        maskname : str
                   The column that the mask will be saved under in the masks dataframe.
                   If the column already exists, then the mask in that column will be overwritten.

        kwargs : dict
                 Extra arguments that will be passed to the homography computation.

        See Also
        --------
        autocnet.transformation.transformations.homography.compute_homography
        """
        matches = self.matches

        s_keypoints = matches[["source_x", "source_y"]].values.astype(np.float64)
        d_keypoints = matches[["destination_x", "destination_y"]].values.astype(np.float64)

        self['homography'], hmask = hm.compute_homography(s_keypoints, d_keypoints, method=method, **kwargs)

        self.masks[maskname] = hmask

    def compute_fundamental_matrix(self, method='ransac', maskname='fundamental', **kwargs):
        """
        Estimate the fundamental matrix (F) using the correspondences tagged to this
        edge.


        Parameters
        ----------
        method : {'ransac', 'lmeds', 'normal', '8point'}
                 The method that will be used when computing the homography.

        maskname : str
                   The column that the mask will be saved under in the masks dataframe.
                   If the column already exists, then the mask in that column will be overwritten.

        kwargs : dict
                 Extra arguments that will be passed to the fundamental matrix computation.

        See Also
        --------
        autocnet.transformation.transformations.fundamental_matrix.compute_fundamental_matrix

        """

        matches = self.matches

        s_keypoints = matches[["source_x", "source_y"]].values.astype(np.float64)
        d_keypoints = matches[["destination_x", "destination_y"]].values.astype(np.float64)
        self.fundamental_matrix, fmask = fm.compute_fundamental_matrix(s_keypoints, d_keypoints, method=method, **kwargs)

        self.masks[maskname] = fmask

    @property
    def weights(self):
        with self.parent.session_scope() as session:
            res = session.query(Edges.weights).\
                                            filter(Edges.source == self.source['node_id']).\
                                            filter(Edges.destination == self.destination['node_id']).\
                                            one()[0]
        self._weights = json.loads(res)
        return self._weights

    @weights.setter
    def weights(self, kv):
        key, value = kv
        with self.parent.session_scope() as session:
            res = session.query(Edges).\
                                            filter(Edges.source == self.source['node_id']).\
                                            filter(Edges.destination == self.destination['node_id']).\
                                            one()
            weights = json.loads(res.weights)
            weights[key] = value

            res.weights = json.dumps(weights)

    @property
    def parent(self):
        return getattr(self, '_parent', None)

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def masks(self):
        with self.parent.session_scope() as session:
            res = session.query(Edges.masks).\
                                            filter(Edges.source == self.source['node_id']).\
                                            filter(Edges.destination == self.destination['node_id']).\
                                            first()
        try:
            df = pd.DataFrame.from_records(res[0])
            df.index = df.index.map(int)
        except:
            ids = list(map(int, self.matches.index.values))
            df = pd.DataFrame(index=ids)
        df.index.name = 'match_id'
        return DbDataFrame(df, parent=self, name='masks')

    @masks.setter
    def masks(self, v):

        def dict_check(input):
            for k, v in input.items():
                if isinstance(v, dict):
                    dict_check(v)
                elif v is None:
                    continue
                elif np.isnan(v):
                    input[k] = None


        df = pd.DataFrame(v)
        with self.parent.session_scope() as session:
            res = session.query(Edges).\
                                    filter(Edges.source == self.source['node_id']).\
                                    filter(Edges.destination == self.destination['node_id']).first()
            if res:
                as_dict = df.to_dict()
                dict_check(as_dict)
                # Update the masks
                res.masks = as_dict
                session.add(res)

    @property
    def costs(self):
        # these are np.float coming out, sqlalchemy needs ints
        ids = list(map(int, self.matches.index.values))
        with self.parent.session_scope() as session:
            res = session.query(Costs).filter(Costs.match_id.in_(ids)).all()
            #qf = q.filter(Costs.match_id.in_(ids))
            if res:
                # Parse the JSON dicts in the cost field into a full dimension dataframe
                costs = {r.match_id:r._cost for r in res}
                df = pd.DataFrame.from_records(costs).T  # From records is important because from_dict drops rows with empty dicts
            else:
                df = pd.DataFrame(index=ids)
        df.index.name = 'match_id'
        return DbDataFrame(df, parent=self, name='costs')


    @costs.setter
    def costs(self, v):
        to_db_add = []
        # Get the query obj
        with self.parent.session_scope() as session:
            q = session.query(Costs)
            # Need the new instance here to avoid __setattr__ issues
            df = pd.DataFrame(v)
            for idx, row in df.iterrows():
                # Now invert the expanded dict back into a single JSONB column for storage
                res = q.filter(Costs.match_id == idx).first()
                if res:
                    #update the JSON blob
                    costs_new_or_updated = row.to_dict()
                    for k, v in costs_new_or_updated.items():
                        if v is None:
                            continue
                        elif np.isnan(v):
                            v = None
                        res._cost[k] = v
                    sqlalchemy.orm.attributes.flag_modified(res, '_cost')
                    session.add(res)
                else:
                    row = row.to_dict()
                    costs = row.pop('_costs', {})
                    for k, v in row.items():
                        if np.isnan(v):
                            v = None
                        costs[k] = v
                    cost = Costs(match_id=idx, _cost=costs)
                    to_db_add.append(cost)
            if to_db_add:
                session.bulk_save_objects(to_db_add)

    @property
    def matches(self):
        with self.parent.session_scope() as session:
            q = session.query(Matches)
            qf = q.filter(Matches.source == self.source['node_id'],
                          Matches.destination == self.destination['node_id'])
            odf = pd.read_sql(qf.statement, q.session.bind).set_index('id')
            df = pd.DataFrame(odf.values, index=odf.index.values, columns=odf.columns.values)
            df.index.name = 'id'
        return DbDataFrame(df,  parent=self, name='matches')

    @matches.setter
    def matches(self, v):
        to_db_add = []
        to_db_update = []
        df = pd.DataFrame(v)
        df.index.name = v.index.name
        # Get the query obj
        with self.parent.session_scope() as session:
            q = session.query(Matches)
            for idx, row in df.iterrows():
                # Determine if this is an update or the addition of a new row
                if hasattr(row, 'id'):
                    res = q.filter(Matches.id == row.id).first()
                    match_id = row.id
                elif v.index.name == 'id':
                    res = q.filter(Matches.id == row.name).first()
                    match_id = row.name
                else:
                    res = None
                if res:
                    # update
                    mapping = {}
                    mapping['id'] = match_id
                    for index in row.index:
                        row_val = row[index]
                        if isinstance(row_val, (np.int,)):
                            row_val = int(row_val)
                        elif isinstance(row_val, (np.float,)):
                            row_val = float(row_val)
                        # This should be uncommented if the matches
                        # df is refactored to be a geodataframe
                        #elif isinstance(row_val, WKBElement):
                        #    continue
                        mapping[index] = row_val
                    to_db_update.append(mapping)
                else:
                    match = Matches()
                    # Dynamically iterate over the columns and if the match has an
                    # attribute with the column name, set it.
                    for c in df.columns:
                        if hasattr(match, c):
                            setattr(match, c, row[c])
                    to_db_add.append(match)
            if to_db_add:
                session.bulk_save_objects(to_db_add)
            if to_db_update:
                session.bulk_update_mappings(Matches, to_db_update)

    @matches.deleter
    def matches(self):
        with self.parent.session_scope() as session:
            session.query(Matches).filter(Matches.source == self.source['node_id'], Matches.destination == self.destination['node_id']).delete()

    @property
    def ring(self):
        res = self._from_db(Edges).first()
        if res:
            return res.ring
        return

    @ring.setter
    def ring(self, ring):
        # Setters need a single session and so should not make use of the
        # syntax sugar _from_db
        with self.parent.session_scope() as session:
            res = session.query(Edges).\
                   filter(Edges.source == self.source['node_id']).\
                   filter(Edges.destination == self.destination['node_id']).first()
            if res:
                res.ring = ring
            else:
                edge = Edges(source=self.source['node_id'],
                             destination=self.destination['node_id'],
                             ring=ring)
                session.add(edge)

    @property
    def intersection(self):
        if not hasattr(self, '_intersection'):
            s_fp = self.source.footprint
            d_fp = self.destination.footprint
            self._intersection = s_fp.intersection(d_fp)
        return self._intersection

    @property
    def fundamental_matrix(self):
        res = self._from_db(Edges).first()
        if res:
            return np.asarray(res.fundamental)

    @fundamental_matrix.setter
    def fundamental_matrix(self, v):
        with self.parent.session_scope() as session:
            res = session.query(Edges).\
                   filter(Edges.source == self.source['node_id']).\
                   filter(Edges.destination == self.destination['node_id']).first()
            if res:
                res.fundamental = v
            else:
                edge = Edges(source=self.source['node_id'],
                             destination=self.destination['node_id'],
                             fundamental = v)
                session.add(edge)

    def get_overlapping_indices(self, kps):
        lons_og, lats_og, alts = reproject([kps.xm.values, kps.ym.values, kps.zm.values], semi_major, semi_minor, 'geocent', 'latlon')
        lons, lats = og2oc(lons_og, lats_og, semi_major, semi_minor)
        points = [Point(lons[i], lats[i]) for i in range(len(lons))]
        mask = [i for i in range(len(points)) if self.intersection.contains(points[i])]
        return mask

    def measures(self, filters={}):
        with self.parent.session_scope() as session:
            query = session.query(Measures).filter(sqlalchemy.or_(Measures.imageid == self.source['node_id'], Measures.imageid == self.destination['node_id']))
            for attr, value in filters.items():
                query = query.filter(getattr(Measures,attr)==value)
            res = query.all()
            session.expunge_all()
        return res

    def network_to_matches(self, ignore_point=False, ignore_measure=False, rejected_jigsaw=False):
        """
        For the edge, take any points/measures that are in the database and
        convert them into matches on the associated edge.

        Parameters
        ----------
        ignore_point : bool
                       If False (default) only select the points that are
                       not ignored (currently active).

        ignore_measure : bool
                         If False (default) only add the measures that are
                         not ignored (currently active).

        rejected_jigsaw : bool
                          If False (default) add any points that are not
                          set to jigsaw rejected.

        """
        source = self.source['node_id']
        destin = self.destination['node_id']

        if source > destin:
            source, destin = destin, source

        with self.parent.session_scope() as session:
            q = session.query(Points.id,
                      Points.pointtype,
                      Measures.id.label('mid'),
                      Measures.sample,
                      Measures.line,
                      Measures.measuretype,
                      Measures.imageid,
                      Measures.apriorisample,
                      Measures.aprioriline).\
                filter(Points.ignore==ignore_point,
                       Measures.ignore==ignore_measure,
                       Measures.jigreject==rejected_jigsaw,
                       sqlalchemy.or_(Measures.imageid==source,
                                      Measures.imageid==destin)).join(Measures)

            df = pd.read_sql(q.statement, self.parent.engine)
            matches = []
            columns = ['point_id', 'source_measure_id', 'destin_measure_id', 'source',
                       'source_idx', 'destination', 'destination_idx',
                       'lat', 'lon', 'geom',
                       'source_x', 'source_y', 'source_apriori_x', 'source_apriori_y',
                       'destination_x','destination_y', 'destination_apriori_x', 'destination_apriori_y',
                       'shift_x', 'shift_y', 'original_destination_x',
                       'original_destination_y']

        def net2matches(grp, matches, source, destin):
            # Grab the image ids and then get the cartesian product of the ids to know which
            # edges to put the matches onto
            if len(grp) != 2:
                return

            imagea = grp[grp['imageid'] == source].iloc[0]
            imageb = grp[grp['imageid'] == destin].iloc[0]
            match = [int(imagea.id), int(imagea.mid), int(imageb.mid),
                     source, 0, destin, 0,
                     None, None, None,
                     imagea['sample'], imagea['line'], imagea['apriorisample'], imagea['aprioriline'],
                     imageb['sample'], imageb['line'], imageb['apriorisample'], imageb['aprioriline'],
                     None, None, None, None]
            matches.append(match)

        df.groupby('id').apply(net2matches, matches, source, destin)
        self.matches = pd.DataFrame(matches, columns=columns)

    def mask_to_counter(self, mask):
        """
        Take a mask on an edge and convert the mask into a counter where
        the key is the match id and value is 1 (the match is flagged false).

        TODO: Allow the mask to be an iterable (list). The caller of this should
        then worry about normalization as n-mask strings can come in and we
        cannot anticipate how the user might want to normalize the return.

        Parameters
        ----------
        mask : str
               The name of the mask

        Returns
        -------
          : collections.Counter
            With keys equal to the indices of the False matches
            and values equal to one

        """
        mask = self.masks[mask]
        matches_to_disable = mask[mask == False].index


        bad = {}
        with self.parent.session_scope() as session:
            for o in session.query(Matches).filter(Matches.id.in_(matches_to_disable)).all():
                # This can't just set both to False, we loose a ton of good points - a bad point in 1 image is not
                # necessarily bad in all of the other images. Doing it this way assumes that it is...
                bad[o.source_measure_id] = 1
                bad[o.destin_measure_id] = 1
        return Counter(bad)

    def find_IQR_outliers(self,
                          scaling=1.5,
                          filters={'template_metric': 1, 'template_shift': 0},
                          n_tolerance=10):
        """
        Based on the interquartile range, find the measure outliers from line and sample shifts.

        Parameters
        ----------
        scaling: float
                 scaling factor to use on IQR when determining outlier range

        filters: dict
                 filters used on a match's source measure, only source measures which
                 satisfy filter will be used in outlier calculation

        n_tolerance: int
                     minimum number of measures needed to calculate outliers

        Returns
        -------
        measure_ids: list
                     list of measure ids corresponding to outliers

        resultlog:  dict
                    status of finding outlier
        """
        resultlog = []
        currentlog = {'edge': f'({self.source}, {self.destination})',
                      'status': ''}

        ref_measures = self.measures(filters=filters)
        ref_measure_ids = [m.id for m in ref_measures]

        # use matches where the source measure is the reference measure
        matches = self.matches
        new_match_idx = []
        for i, row in matches.iterrows():
            if (row['source_measure_id'] in ref_measure_ids):
                new_match_idx.append(i)
        matches = matches.loc[new_match_idx]

        # check these is enough data to stastically calculate outliers
        if len(new_match_idx) == 0:
            currentlog['status'] = 'no filtered measures in edge.'
            resultlog.append(currentlog)
            return None, resultlog
        elif len(new_match_idx) < n_tolerance:
            currentlog['status'] = f'{len(new_match_idx)} < {n_tolerance} filtered measures are not statistically signigicant.'
            resultlog.append(currentlog)
            return None, resultlog

        # calculate outliers
        sampleShift = matches['destination_x'] - matches['destination_apriori_x']
        lineShift =  matches['destination_y'] - matches['destination_apriori_y']

        sample_q1, sample_q3 = np.percentile(sampleShift,[25,75])
        line_q1, line_q3 = np.percentile(lineShift,[25,75])
        sample_iqr = sample_q3 - sample_q1
        line_iqr = line_q3 - line_q1

        sample_lowerBound = sample_q1 -(scaling * sample_iqr)
        sample_upperBound = sample_q3 +(scaling * sample_iqr)
        line_lowerBound = line_q1 -(scaling * line_iqr)
        line_upperBound = line_q3 +(scaling * line_iqr)

        sample_outlier_matches = sampleShift[(sampleShift <= sample_lowerBound) | (sampleShift >= sample_upperBound)]
        line_outlier_matches = lineShift[(lineShift <= line_lowerBound) | (lineShift >= line_upperBound)]
        outlier_match_idx = np.concatenate((sample_outlier_matches.index.values, line_outlier_matches.index.values))

        measure_ids = matches.loc[outlier_match_idx]['destin_measure_id'].values
        currentlog['status'] = f'{len(measure_ids)} outliers found.'
        resultlog.append(currentlog)

        return measure_ids, resultlog


    def ignore_outliers(self, outlier_method='IQR', **kwargs):
        """
        Find and ignore outlier measures as determined by outlier method

        Parameters
        ----------
        outlier_method: str
                        method used to determine outliers.
                        Current methods:
                        - interquartile range ('IQR') of line/sample shift

        Returns
        -------
        resultlog: dict
                   status of finding and ignoring outliers
        """
        outlier_dict = {'IQR': self.find_IQR_outliers}
        outlier_func = outlier_dict[outlier_method]

        outlier_destination_mids, resultlog = outlier_func(**kwargs)

        if outlier_destination_mids is None:
            return resultlog

        # ignore outlier measures
        with self.parent.session_scope() as session:
            for m in outlier_destination_mids:
                currentlog = {'measure id': m,
                              'status': 'Ignored'}
                session.query(Measures).filter(Measures.id==m).update({'ignore': True})
                resultlog.append(currentlog)

        return resultlog


