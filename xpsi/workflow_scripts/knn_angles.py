

# Based on https://github.com/scikit-learn/scikit-learn/blob/7e85a6d1f/sklearn/neighbors/_regression.py#L23
# Used under license: BSD 3 clause (C) INRIA, University of Amsterdam, University of Copenhagen

import warnings

import numpy as np
from sklearn.neighbors._base import NeighborsBase, KNeighborsMixin, SupervisedFloatMixin
from sklearn.base import RegressorMixin
from sklearn.utils import check_array
from math import atan, radians, degrees
from scipy.stats import circmean

def _fix_quadrents(elts):
    lon_elts = elts[:, :, 0]
    in_q1 = np.less(lon_elts, -90)
    in_q4 = np.greater(lon_elts, 90)
    using_both = np.logical_and(np.any(in_q4, axis=1), np.any(in_q1, axis=1))
    need_modify = np.logical_and(np.reshape(using_both, (-1, 1)), in_q4)
    elts[need_modify, 0] -= 360
    return elts

def _vector_mean(elts):
    phi = np.deg2rad(elts[:, :, 0])
    theta = np.deg2rad(elts[:, :, 1])
    sin_theta = np.sin(theta)

    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)

    x_mean = np.mean(x, axis=1)
    y_mean = np.mean(y, axis=1)
    z_mean = np.mean(z, axis=1)

    phi_mean = np.arctan2(y_mean, x_mean)
    hypot = np.hypot(x_mean, y_mean)
    theta_mean = np.arctan2(hypot, z_mean)

    return np.stack([np.rad2deg(phi_mean), np.rad2deg(theta_mean)],
                    axis=-1)

def _vector_mean_3_rad(elts):

    phi = np.deg2rad(elts[:, :, 0])
    theta = np.deg2rad(elts[:, :, 1])
    psi = np.deg2rad(elts[:, :, 2])

    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)

    x_mean = np.mean(x, axis=1)
    y_mean = np.mean(y, axis=1)
    z_mean = np.mean(z, axis=1)

    #print('x,y,z', x_mean.shape, y_mean.shape, z_mean.shape)
    phi_mean = np.arctan2(y_mean, x_mean)
    hypot = np.hypot(x_mean, y_mean)
    theta_mean = np.arctan2(hypot, z_mean)
    
    #psi_mean = circmean(psi, axis=1)
    psi_mean = np.mean(psi, axis=1)
    
    
    return np.stack([np.rad2deg(phi_mean), np.rad2deg(theta_mean), np.rad2deg(psi_mean)],
                    axis=-1)

def _vector_mean_3(elts):

    phi = np.deg2rad(elts[:, :, 0])
    theta = np.deg2rad(elts[:, :, 1])
    #psi = np.deg2rad(elts[:, :, 2])
    #print('size of input angles', phi.shape, theta.shape, psi.shape)

    #theta_mean = -1*np.arcsin(np.mean(-1*np.sin(theta), axis=1))
    #phi_mean = np.arctan2(np.mean(np.sin(phi)*np.cos(theta),axis=1)/np.mean(np.cos(theta),axis=1),np.mean(np.cos(phi)*np.cos(theta),axis=1)/np.mean(np.cos(theta),axis=1))
   
    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)

    x_mean = np.mean(x, axis=1)
    y_mean = np.mean(y, axis=1)
    z_mean = np.mean(z, axis=1)

    #print('x,y,z', x_mean.shape, y_mean.shape, z_mean.shape)
    phi_mean = np.arctan2(y_mean, x_mean)
    hypot = np.hypot(x_mean, y_mean)
    theta_mean = np.arctan2(hypot, z_mean)
    
    #psi_mean = circmean(psi, axis=1)
    #psi_mean = np.mean(psi, axis=1)
    
    
    #Average psi angle between 0, 360
    #1. Rotation this system anticlockwise through 15 degrees
    psi = np.asarray(elts[:,:,2]) - 15
    
    indices_negative = np.where(psi< 0)
    psi[indices_negative] = psi[indices_negative] + 360
   
    sin_psi= np.sin(np.deg2rad(psi))
    cos_psi= np.cos(np.deg2rad(psi))
    
    sum_sin = np.mean(sin_psi, axis=1)
    sum_cos = np.mean(cos_psi, axis=1)

    psi_mean = []
    for i in range(len(psi)):
        if (degrees(sum_sin[i]) > 0 and degrees(sum_cos[i]) >0): psi_m = degrees(atan(sum_sin[i]/sum_cos[i]))+ 15
        elif ( degrees(sum_cos[i]) < 0 ): psi_m = degrees(atan(sum_sin[i]/sum_cos[i])) + 180 + 15
        elif (degrees(sum_sin[i]) < 0 and degrees(sum_cos[i]) > 0):  psi_m = degrees(atan(sum_sin[i]/sum_cos[i])) + 360 + 15
        psi_mean.append(psi_m)
    psi_mean = np.asarray(psi_mean)
   
    #print('average', np.rad2deg(phi_mean).shape, np.rad2deg(theta_mean).shape, psi_mean.shape) 
    ''' 
    psi_mean = np.arctan2(np.mean(np.sin(psi)*np.cos(theta),axis=1)/np.mean(np.cos(theta),axis=1),np.mean(np.cos(psi)*np.cos(theta),axis=1)/np.mean(np.cos(theta),axis=1))
    
    temp_psi = np.rad2deg(psi_mean)
    for i in range(len(temp_psi)):
        if temp_psi[i] < 0:
            temp_psi[i] = 360 + temp_psi[i]
    '''
    return np.stack([np.rad2deg(phi_mean), np.rad2deg(theta_mean), psi_mean],
                    axis=-1)

def _vector_mean_q(elts):
    phi = np.deg2rad(elts[:, :, 0])
    theta = np.deg2rad(elts[:, :, 1])
    psi = np.deg2rad(elts[:, :, 2])

    #Abbreviations for the various angular functions
    cy = np.cos(psi * 0.5)
    sy = np.sin(psi * 0.5)
    cp = np.cos(theta * 0.5)
    sp = np.sin(theta * 0.5)
    cr = np.cos(phi * 0.5)
    sr = np.sin(phi * 0.5)

    #Quaternions
    w = np.mean(cr * cp * cy + sr * sp * sy, axis=1)
    x = np.mean(sr * cp * cy - cr * sp * sy, axis=1)
    y = np.mean(cr * sp * cy + sr * cp * sy, axis=1)
    z = np.mean(cr * cp * sy - sr * sp * cy, axis=1)

    #Back to euler angles
    #Phi
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    phi_mean = np.arctan2(sinr_cosp, cosr_cosp)

    #Theta
    sinp = 2 * (w * y - z * x)
    #if (np.absolute(sinp) >= 1):
    #    theta_mean= math.copysign(math.pi/2, sinp)

    #else:
    theta_mean=np.arcsin(sinp)

    #Psi
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    psi_mean = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([np.rad2deg(phi_mean), np.rad2deg(theta_mean), np.rad2deg(psi_mean)],
                    axis=-1)


class KNeighborsAngleRegressor(NeighborsBase, KNeighborsMixin,  SupervisedFloatMixin, RegressorMixin):
    """Regression based on k-nearest neighbors for angles.
    The target is predicted by local interpolation of the targets
    associated of the nearest neighbors in the training set.
    Read more in the :ref:`User Guide <regression>`.
    .. versionadded:: 0.9
    Parameters
    ----------
    average : string or callable, optional (default = 'mean')
        How to compute the average:
        - 'mean' will compute the mean without regard for the fact the
          first value of each pair is modulus 360
        - 'fix_quadrents' will reduce the angles greater than 270 by 360 in
          columns were there are both values greater than 270 and values less
          than 90, then take the mean.
        - 'vector_mean' will convert the angle pairs into unit vectors,
          average the vectors, then compute the angle of the vector
        - callables will be called with the array of neighbors
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    metric : string or callable, default 'minkowski'
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of the DistanceMetric class for a
        list of available metrics.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`Glossary <sparse graph>`,
        in which case only "nonzero" elements may be considered neighbors.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.
    Attributes
    ----------
    effective_metric_ : string or callable
        The distance metric to use. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.
    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.
    See also
    --------
    KNeighborsRegressor
    """

    def __init__(self, n_neighbors=5, average='mean',
                 algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None,
                 **kwargs):
        super().__init__(
              n_neighbors=n_neighbors,
              algorithm=algorithm,
              leaf_size=leaf_size, metric=metric, p=p,
              metric_params=metric_params, n_jobs=n_jobs, **kwargs)
        self._avg = average

    @property
    def _pairwise(self):
        # For cross-validation routines to split data correctly
        return self.metric == 'precomputed'

    def predict(self, X):
        """Predict the target for the provided data
        Parameters
        ----------
        X : array-like, shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : array of int, shape = [n_queries] or [n_queries, n_outputs]
            Target values
        """
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        _avg = self._avg
        if _avg == 'mean':
            y_pred = np.mean(_y[neigh_ind], axis=1)
            
        elif _avg == 'fix_quadrents':
            elts = _fix_quadrents(_y[neigh_ind])
            y_pred = np.mean(elts, axis=1)

        elif _avg == 'vector_mean':
            y_pred = _vector_mean(_y[neigh_ind])
        
        elif _avg == 'vector_mean_3':
            y_pred = _vector_mean_3(_y[neigh_ind])

        elif _avg == 'vector_mean_3_rad':
            y_pred = _vector_mean_3_rad(_y[neigh_ind])

        elif _avg == 'vector_mean_q':
            y_pred = _vector_mean_q(_y[neigh_ind])
        
        else:
            y_pred = np.fromiter((_avg(yi) for yi in _y[neigh_ind]), _y.dtype)

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred
