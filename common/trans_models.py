import numpy as np
import copy
import scipy.interpolate


class AbstractModel(object):
    def __init__(self):
        pass

    def score(self, x, y, epsilon=100.0, min_inlier_ratio=0.01, min_num_inlier=7):
        """
        Computes how good is the transformation.
        This is done by applying the transformation to the collection of points in x,
        and then computing the corresponding distance to the matched point in y.
        If the distance is less than epsilon, the match is considered good.
        """
        x2 = self.apply(x)
        dists_sqr = np.sum((y - x2) ** 2, axis=1)
        good_dists_mask = dists_sqr < epsilon ** 2
        good_dists_num = np.sum(good_dists_mask)
        accepted_ratio = float(good_dists_num) / x2.shape[0]

        # The transformation does not adhere to the wanted values, give it a very low score
        if good_dists_num < min_num_inlier or accepted_ratio < min_inlier_ratio:
            return -1, None, -1

        return accepted_ratio, good_dists_mask, 0

    def apply(self, p):
        raise RuntimeError("Not implemented, but probably should be")

    def fit(self, x, y):
        raise RuntimeError("Not implemented, but probably should be")

    def set_from_modelspec(self, s):
        raise RuntimeError("Not implemented, but probably should be")

    def to_modelspec(self):
        raise RuntimeError("Not implemented, but probably should be")

    def is_affine(self):
        return False


class AbstractAffineModel(AbstractModel):
    def __init__(self):
        super().__init__()

    def get_matrix(self):
        raise RuntimeError("Not implemented, but probably should be")

    def apply(self, p):
        """
        Returns a new 2D point(s) after applying the transformation on the given point(s) p
        """
        pts = np.atleast_2d(p)
        m = self.get_matrix()
        return np.dot(m[:2, :2], pts.T).T + np.asarray(m.T[2][:2]).reshape((1, 2))

    def apply_inv(self, p):
        """
        Returns a new 2D point(s) after applying the inverse transformation on the given point(s) p
        """
        pts = np.atleast_2d(p)
        m = self.get_matrix()
        m_inv = np.linalg.inv(m)
        return np.dot(m_inv[:2, :2], pts.T).T + np.asarray(m_inv.T[2][:2]).reshape((1, 2))

    def is_affine(self):
        return True


class TranslationModel(AbstractAffineModel):
    MIN_MATCHES_NUM = 2
    class_name = "mpicbg.trakem2.transform.TranslationModel2D"

    def __init__(self, delta=np.array([0, 0])):
        super().__init__()
        self.delta = None
        self.set(delta)

    def set(self, delta):
        self.delta = np.asarray(delta)

    def apply(self, p):
        if p.ndim == 1:
            return p + self.delta
        return np.atleast_2d(p) + np.asarray(self.delta).reshape((-1, 2))

    def apply_inv(self, p):
        if p.ndim == 1:
            return p - self.delta
        return np.atleast_2d(p) - np.asarray(self.delta).reshape((-1, 2))

    def to_str(self):
        return "T={}".format(self.delta)

    def to_modelspec(self):
        return {
            "className": self.class_name,
            "dataString": "{}".format(' '.join([str(float(x)) for x in self.delta]))
        }

    def set_from_modelspec(self, s):
        self.delta = np.array([float(d) for d in s.split()])

    def get_matrix(self):
        return np.array([
            [1.0, 0.0, self.delta[0]],
            [0.0, 1.0, self.delta[1]],
            [0.0, 0.0, 1.0]
        ])

    def fit(self, x, y):
        """
        A non-weighted fitting of a collection of 2D points in X to a collection of 2D points in y.
        X and y are assumed to be arrays of 2D points of the same shape.
        """
        assert (x.shape[0] >= 2)  # the minimal number of matches for a 2d rigid transformation

        pc = np.mean(x, axis=0)
        qc = np.mean(y, axis=0)

        self.delta = qc - pc
        return True


class RigidModel(AbstractAffineModel):
    MIN_MATCHES_NUM = 2
    class_name = "mpicbg.trakem2.transform.RigidModel2D"

    def __init__(self, r=0.0, delta=np.array([0, 0])):
        super().__init__()
        self.delta = None
        self.sin_val = None
        self.cos_val = None
        self.set(r, delta)

    def set(self, r, delta):
        self.cos_val = np.cos(r)
        self.sin_val = np.sin(r)
        self.delta = np.asarray(delta)

    def apply(self, p):
        """
        Returns a new 2D point(s) after applying the transformation on the given point(s) p
        """
        if p.ndim == 1:
            return np.dot([[self.cos_val, -self.sin_val],
                           [self.sin_val, self.cos_val]],
                          p).T + np.asarray(self.delta).reshape((1, 2))
        pts = np.atleast_2d(p)
        return np.dot([[self.cos_val, -self.sin_val],
                       [self.sin_val, self.cos_val]],
                      pts.T).T + np.asarray(self.delta).reshape((1, 2))

    def apply_inv(self, p):
        """
        Returns a new 2D point(s) after applying the inverse transformation on the given point(s) p
        """
        # The inverse matrix of the [2,2] rigid matrix is similar to the forward matrix (the angle is negative),
        # the delta needs to be computed by R-1*delta
        inv_delta = np.dot([[self.cos_val, self.sin_val],
                            [-self.sin_val, self.cos_val]], self.delta).T
        if p.ndim == 1:
            return np.dot([[self.cos_val, self.sin_val],
                           [-self.sin_val, self.cos_val]],
                          p).T + inv_delta
        pts = np.atleast_2d(p)
        return np.dot([[self.cos_val, self.sin_val],
                       [-self.sin_val, self.cos_val]],
                      pts.T).T + inv_delta

    def to_str(self):
        return "R={}, T={}".format(np.arctan2(-self.sin_val, self.cos_val), self.delta)

    def to_modelspec(self):
        return {
            "className": self.class_name,
            "dataString": "{} {}".format(np.arctan2(-self.sin_val, self.cos_val),
                                         ' '.join([str(float(x)) for x in self.delta]))
        }

    def set_from_modelspec(self, s):
        split = s.split()
        r = float(split[0])
        self.cos_val = np.cos(r)
        self.sin_val = np.sin(r)
        self.delta = np.array([float(d) for d in split[1:]])

    def get_matrix(self):
        return np.array([
            [self.cos_val, -self.sin_val, self.delta[0]],
            [self.sin_val, self.cos_val, self.delta[1]],
            [0, 0, 1]
        ])

    def fit(self, x, y):
        """
        A non-weighted fitting of a collection of 2D points in x to a collection of 2D points in y.
        x and y are assumed to be arrays of 2D points of the same shape.
        """
        assert (x.shape[0] >= 2)  # the minimal number of matches for a 2d rigid transformation

        pc = np.mean(x, axis=0)
        qc = np.mean(y, axis=0)

        delta_c = pc - qc
        # dx = pc[0] - qc[0]
        # dy = pc[1] - qc[1]

        delta1 = x - pc
        # delta2 = y - qc + np.array([dx, dy])
        delta2 = y - qc + delta_c

        # for xy1, xy2 in zip(delta1, delta2):
        #     sind += xy1[0] * xy2[1] - xy1[1] * xy2[0]
        #     cosd += xy1[0] * xy2[0] + xy1[1] * xy2[1]
        sind = np.sum(delta1[:, 0] * delta2[:, 1] - delta1[:, 1] * delta2[:, 0])
        cosd = np.sum(delta1[:, 0] * delta2[:, 0] + delta1[:, 1] * delta2[:, 1])
        norm = np.sqrt(cosd * cosd + sind * sind)
        if norm < 0.0001:
            # print "normalization may be invalid, skipping fitting"
            return False
        cosd /= norm
        sind /= norm

        self.cos_val = cosd
        self.sin_val = sind
        self.delta[0] = qc[0] - cosd * pc[0] + sind * pc[1]
        self.delta[1] = qc[1] - sind * pc[0] - cosd * pc[1]
        return True


class SimilarityModel(AbstractAffineModel):
    MIN_MATCHES_NUM = 2
    class_name = "mpicbg.trakem2.transform.SimilarityModel2D"

    def __init__(self, s=0.0, delta=np.array([0, 0])):
        super().__init__()
        self.delta = None
        self.ssin_val = None
        self.scos_val = None
        self.set(s, delta)

    def set(self, s, delta):
        self.scos_val = np.cos(s)
        self.ssin_val = np.sin(s)
        self.delta = np.array(delta)

    def apply(self, p):
        """
        Returns a new 2D point(s) after applying the transformation on the given point(s) p
        """
        if p.ndim == 1:
            return np.dot([[self.scos_val, -self.ssin_val],
                           [self.ssin_val, self.scos_val]],
                          p).T + np.asarray(self.delta).reshape((1, 2))
        pts = np.atleast_2d(p)
        return np.dot([[self.scos_val, -self.ssin_val],
                       [self.ssin_val, self.scos_val]],
                      pts.T).T + np.asarray(self.delta).reshape((1, 2))

    def apply_inv(self, p):
        """
        Returns a new 2D point(s) after applying the inverse transformation on the given point(s) p
        """
        # The inverse matrix of the [2,2] rigid matrix is similar to the forward matrix (the angle is negative),
        # the delta needs to be computed by R-1*delta
        inv_delta = np.dot([[self.scos_val, self.ssin_val],
                            [-self.ssin_val, self.scos_val]], self.delta).T
        if p.ndim == 1:
            return np.dot([[self.scos_val, self.ssin_val],
                           [-self.ssin_val, self.scos_val]],
                          p).T + inv_delta
        pts = np.atleast_2d(p)
        return np.dot([[self.scos_val, self.ssin_val],
                       [-self.ssin_val, self.scos_val]],
                      pts.T).T + inv_delta

    def to_str(self):
        return "S={}, T={}".format(np.arccos(self.scos_val), self.delta)

    def to_modelspec(self):
        return {
            "className": self.class_name,
            "dataString": "{} {} {}".format(self.scos_val, self.ssin_val, ' '.join([str(float(x)) for x in self.delta]))
        }

    def set_from_modelspec(self, s):
        split = s.split()
        r = float(split[0])
        self.scos_val = np.cos(r)
        self.ssin_val = np.sin(r)
        self.delta = np.array([float(d) for d in split[1:]])

    def get_matrix(self):
        return np.array([
            np.array([self.scos_val, -self.ssin_val, self.delta[0]]),
            np.array([self.ssin_val, self.scos_val, self.delta[1]]),
            np.array([0, 0, 1])
        ])

    def fit(self, x, y):
        """
        A non-weighted fitting of a collection of 2D points in X to a collection of 2D points in y.
        X and y are assumed to be arrays of 2D points of the same shape.
        """
        assert (x.shape[0] >= 2)  # the minimal number of matches for a 2d rigid transformation

        pc = np.mean(x, axis=0)
        qc = np.mean(y, axis=0)

        delta_c = pc - qc
        # dx = pc[0] - qc[0]
        # dy = pc[1] - qc[1]

        scosd = 0.0
        ssind = 0.0
        delta1 = x - pc
        # delta2 = y - qc + np.array([dx, dy])
        delta2 = y - qc + delta_c

        norm = 0.0
        for xy1, xy2 in zip(delta1, delta2):
            ssind += xy1[0] * xy2[1] - xy1[1] * xy2[0]
            scosd += xy1[0] * xy2[0] + xy1[1] * xy2[1]
            norm += xy1[0] ** 2 + xy1[1] ** 2
        if norm < 0.0001:
            # print "normalization may be invalid, skipping fitting"
            return False
        scosd /= norm
        ssind /= norm

        self.scos_val = scosd
        self.ssin_val = ssind
        self.delta[0] = qc[0] - scosd * pc[0] + ssind * pc[1]
        self.delta[1] = qc[1] - ssind * pc[0] - scosd * pc[1]
        return True


class AffineModel(AbstractAffineModel):
    MIN_MATCHES_NUM = 3
    class_name = "mpicbg.trakem2.transform.AffineModel2D"

    def __init__(self, m=np.eye(3)):
        """m is a 3x3 matrix"""
        super().__init__()
        self.m_inv = None
        self.m = None
        self.set(m)

    def set(self, m):
        """m is a 3x3 matrix"""
        # make sure that this a 3x3 matrix
        m = np.array(m)
        if m.shape != (3, 3):
            raise RuntimeError("Error when parsing the given affine matrix, should be of size 3x3")
        self.m = m

    def apply(self, p):
        """
        Returns a new 2D point(s) after applying the transformation on the given point(s) p
        """
        if p.ndim == 1:
            return np.dot(self.m[:2, :2], p) + np.asarray(self.m.T[2][:2]).reshape((1, 2))
        pts = np.atleast_2d(p)
        return np.dot(self.m[:2, :2], pts.T).T + np.asarray(self.m.T[2][:2]).reshape((1, 2))

    def apply_inv(self, p):
        """
        Returns a new 2D point(s) after applying the inverse transformation on the given point(s) p
        """
        # The inverse matrix of the [2,2] rigid matrix is similar to the forward matrix (the angle is negative),
        # the delta needs to be computed by R-1*delta
        self.m_inv = np.linalg.inv(self.m)
        if p.ndim == 1:
            return np.dot(self.m_inv[:2, :2], p) + np.asarray(self.m_inv.T[2][:2]).reshape((1, 2))
        pts = np.atleast_2d(p)
        return np.dot(self.m_inv[:2, :2],
                      pts.T).T + np.asarray(self.m_inv.T[2][:2]).reshape((1, 2))

    def to_str(self):
        return "M={}".format(self.m)

    def to_modelspec(self):
        return {
            "className": self.class_name,
            # keeping it in the Fiji model format
            "dataString": "{}".format(' '.join([str(float(x)) for x in self.m[:2].T.flatten()]))
        }

    def set_from_modelspec(self, s):
        split = s.split()
        # The input is 6 numbers that correspond to m00 m10 m01 m11 m02 m12
        self.m = np.vstack(
            (np.array([float(d) for d in split[0::2]]),
             np.array([float(d) for d in split[1::2]]),
             np.array([0.0, 0.0, 1.0]))
        )

    def get_matrix(self):
        return self.m

    def fit(self, x, y):
        """
        A non-weighted fitting of a collection of 2D points in X to a collection of 2D points in y.
        X and y are assumed to be arrays of 2D points of the same shape.
        """
        assert (x.shape[0] >= 2)  # the minimal number of matches for a 2d rigid transformation

        pc = np.mean(x, axis=0)
        qc = np.mean(y, axis=0)

        delta1 = x - pc
        delta2 = y - qc

        a00 = np.sum(delta1[:, 0] * delta1[:, 0])
        a01 = np.sum(delta1[:, 0] * delta1[:, 1])
        a11 = np.sum(delta1[:, 1] * delta1[:, 1])
        b00 = np.sum(delta1[:, 0] * delta2[:, 0])
        b01 = np.sum(delta1[:, 0] * delta2[:, 1])
        b10 = np.sum(delta1[:, 1] * delta2[:, 0])
        b11 = np.sum(delta1[:, 1] * delta2[:, 1])

        det = a00 * a11 - a01 * a01

        if det == 0:
            # print "determinant is 0, skipping fitting"
            return False

        m00 = (a11 * b00 - a01 * b10) / det
        m01 = (a00 * b10 - a01 * b00) / det
        m10 = (a11 * b01 - a01 * b11) / det
        m11 = (a00 * b11 - a01 * b01) / det

        self.m = np.array([
            [m00, m01, qc[0] - m00 * pc[0] - m01 * pc[1]],
            [m10, m11, qc[1] - m10 * pc[0] - m11 * pc[1]],
            [0.0, 0.0, 1.0]
        ])
        return True


class PointsTransformModel(AbstractModel):
    # Not really part of trakem2, but can be interpolated using ThinPlate or MovingLeastSquares transformations
    class_name = "mpicbg.trakem2.transform.PointsTransformModel"

    def __init__(self, point_map=None):
        super().__init__()
        self.point_map = point_map
        self.interpolator = None

    def apply(self, p):
        self.update_interpolator()

        pts = np.atleast_2d(p)
        if p.ndim == 1:
            return self.interpolator(pts)[0]
        return self.interpolator(pts)

    def set_from_modelspec(self, s):
        points_data = s.split()  
        # format is: p1_src_x p1_src_y p1_dest_x p1_dest_y 1.0 ... (1.0 is the weight of the match)
        src = np.array(
            [np.array(points_data[0::5], dtype=np.float32),
             np.array(points_data[1::5], dtype=np.float32)]
        ).T
        dest = np.array(
            [np.array(points_data[2::5], dtype=np.float32),
             np.array(points_data[3::5], dtype=np.float32)]
        ).T
        self.point_map = (src, dest)
        self.interpolator = None

    def to_modelspec(self):
        return {
            "className": self.class_name,
            "dataString": "{}".format(' '.join(
                ['{} {} {} {} 1.0'.format(float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])) for p1, p2 in
                 zip(self.point_map[0], self.point_map[1])]))
        }

    def get_point_map(self):
        return self.point_map

    def update_interpolator(self):
        if self.interpolator is None:
            # Create a cubic spline interpolator from the source point to the dest points
            self.interpolator = scipy.interpolate.CloughTocher2DInterpolator(self.point_map[0], self.point_map[1])


class Transforms(object):
    transformations = [TranslationModel(), RigidModel(), SimilarityModel(), AffineModel()]
    transforms_classnames = {
        TranslationModel.class_name: TranslationModel(),
        RigidModel.class_name: RigidModel(),
        SimilarityModel.class_name: SimilarityModel(),
        AffineModel.class_name: AffineModel(),
        PointsTransformModel.class_name: PointsTransformModel(),
    }

    @classmethod
    def create(cls, model_type_idx):
        return copy.deepcopy(cls.transformations[model_type_idx])

    @classmethod
    def from_tilespec(cls, ts_transform):
        transform = copy.deepcopy(cls.transforms_classnames[ts_transform["className"]])
        transform.set_from_modelspec(ts_transform["dataString"])
        return transform
