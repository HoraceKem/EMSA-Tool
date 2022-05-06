#cython: boundscheck=False, wraparound=False, cdivision=True
from __future__ import division
import numpy as np
cimport numpy
cimport openmp
from cython.parallel import parallel, threadid, prange
from libc.math cimport sqrt, fabs

ctypedef numpy.float64_t FLOAT_TYPE
ctypedef numpy.int32_t int32
ctypedef numpy.uint32_t uint32

npFLOAT_TYPE = np.float64

cdef extern from "math.h":
    float INFINITY

cdef:
    FLOAT_TYPE small_value = 0.0001

from libc.stdio cimport printf


##################################################
# HUBER LOSS FUNCTION
##################################################
cdef  inline FLOAT_TYPE c_huber(FLOAT_TYPE value,
                                FLOAT_TYPE target,
                                FLOAT_TYPE sigma,
                                FLOAT_TYPE d_value_dx,
                                FLOAT_TYPE d_value_dy,
                                FLOAT_TYPE *d_huber_dx,
                                FLOAT_TYPE *d_huber_dy) nogil:
    cdef:
        FLOAT_TYPE diff, a, b, l

    diff = value - target
    if fabs(diff) <= sigma:
        a = (diff ** 2) / 2
        d_huber_dx[0] = diff * d_value_dx
        d_huber_dy[0] = diff * d_value_dy
        return a
    else:
        b = sigma * (fabs(diff) - sigma / 2)
        d_huber_dx[0] = sigma * d_value_dx
        d_huber_dy[0] = sigma * d_value_dy
        return b

cpdef huber(value, target, sigma, dx, dy):
    cdef:
        FLOAT_TYPE dhx, dhy
    val = c_huber(<FLOAT_TYPE> value, <FLOAT_TYPE> target, <FLOAT_TYPE> sigma, <FLOAT_TYPE> dx, <FLOAT_TYPE> dy, &(dhx), &(dhy))
    return val, dhx, dhy

##################################################
# REGULARIZED LENGTH FUNCTION
##################################################

cdef  inline FLOAT_TYPE c_reglen(FLOAT_TYPE vx,
                                 FLOAT_TYPE vy,
                                 FLOAT_TYPE d_vx_dx,
                                 FLOAT_TYPE d_vy_dy,
                                 FLOAT_TYPE *d_reglen_dx,
                                 FLOAT_TYPE *d_reglen_dy) nogil:
    cdef:
        FLOAT_TYPE sq_len, sqrt_len

    sq_len = vx * vx + vy * vy + small_value
    sqrt_len = sqrt(sq_len)
    d_reglen_dx[0] = vx / sqrt_len
    d_reglen_dy[0] = vy / sqrt_len
    return sqrt_len

cpdef reglen(vx, vy):
    cdef:
        FLOAT_TYPE drx, dry
    val = c_reglen(<FLOAT_TYPE> vx, <FLOAT_TYPE> vy, <FLOAT_TYPE> 1.0, 1.0, &(drx), &(dry))
    return val, drx, dry

##################################################
# MESH CROSS-LINK DERIVS
##################################################
cpdef FLOAT_TYPE crosslink_mesh_derivs(FLOAT_TYPE[:, ::1] mesh1,
                                       FLOAT_TYPE[:, ::1] mesh2,
                                       FLOAT_TYPE[:, ::1] d_cost_d_mesh1,
                                       FLOAT_TYPE[:, ::1] d_cost_d_mesh2,
                                       uint32[:, ::1] indices1,
                                       uint32[:, ::1] indices2,
                                       FLOAT_TYPE[:, ::1] barys1,
                                       FLOAT_TYPE[:, ::1] barys2,
                                       FLOAT_TYPE all_weight,
                                       FLOAT_TYPE sigma) nogil:
    cdef:
        int i
        FLOAT_TYPE px, py, qx, qy
        int pidx0, pidx1, pidx2
        int qidx0, qidx1, qidx2
        FLOAT_TYPE pb0, pb1, pb2
        FLOAT_TYPE qb0, qb1, qb2
        FLOAT_TYPE r, h
        FLOAT_TYPE dr_dx, dr_dy, dh_dx, dh_dy
        FLOAT_TYPE cost

    cost = 0
    for i in range(indices1.shape[0]):
        pidx0 = indices1[i, 0]
        pidx1 = indices1[i, 1]
        pidx2 = indices1[i, 2]
        pb0 = barys1[i, 0]
        pb1 = barys1[i, 1]
        pb2 = barys1[i, 2]

        qidx0 = indices2[i, 0]
        qidx1 = indices2[i, 1]
        qidx2 = indices2[i, 2]
        qb0 = barys2[i, 0]
        qb1 = barys2[i, 1]
        qb2 = barys2[i, 2]

        px = (mesh1[pidx0, 0] * pb0 +
              mesh1[pidx1, 0] * pb1 +
              mesh1[pidx2, 0] * pb2)
        py = (mesh1[pidx0, 1] * pb0 +
              mesh1[pidx1, 1] * pb1 +
              mesh1[pidx2, 1] * pb2)

        qx = (mesh2[qidx0, 0] * qb0 +
              mesh2[qidx1, 0] * qb1 +
              mesh2[qidx2, 0] * qb2)
        qy = (mesh2[qidx0, 1] * qb0 +
              mesh2[qidx1, 1] * qb1 +
              mesh2[qidx2, 1] * qb2)

        r = c_reglen(px - qx, py - qy,
                     1, 1,
                     &(dr_dx), &(dr_dy))
        h = c_huber(r, 0, sigma,
                    dr_dx, dr_dy,
                    &(dh_dx), &(dh_dy))
#        with gil:
#            print("COST: ({}, {}) to ({}, {}) = {}, L = {}, H = {}, all_weight = {}".format(px, py, qx, qy, cost, r, h, all_weight))
        cost += h * all_weight
        dh_dx *= all_weight
        dh_dy *= all_weight

        # update derivs
        d_cost_d_mesh1[pidx0, 0] += pb0 * dh_dx
        d_cost_d_mesh1[pidx1, 0] += pb1 * dh_dx
        d_cost_d_mesh1[pidx2, 0] += pb2 * dh_dx
        d_cost_d_mesh1[pidx0, 1] += pb0 * dh_dy
        d_cost_d_mesh1[pidx1, 1] += pb1 * dh_dy
        d_cost_d_mesh1[pidx2, 1] += pb2 * dh_dy
        # opposite direction for other end of spring, and distributed according to weight
        d_cost_d_mesh2[qidx0, 0] -= qb0 * dh_dx
        d_cost_d_mesh2[qidx1, 0] -= qb1 * dh_dx
        d_cost_d_mesh2[qidx2, 0] -= qb2 * dh_dx
        d_cost_d_mesh2[qidx0, 1] -= qb0 * dh_dy
        d_cost_d_mesh2[qidx1, 1] -= qb1 * dh_dy
        d_cost_d_mesh2[qidx2, 1] -= qb2 * dh_dy
    return cost


##################################################
# MESH INTERNAL-LINK DERIVS
##################################################
cpdef FLOAT_TYPE internal_mesh_derivs(FLOAT_TYPE[:, ::1] mesh,
                                      FLOAT_TYPE[:, ::1] d_cost_d_mesh,
                                      uint32[:, ::1] edge_indices,
                                      FLOAT_TYPE[:] rest_lengths,
                                      FLOAT_TYPE all_weight,
                                      FLOAT_TYPE sigma) nogil:
    cdef:
        int i
        int idx1, idx2
        FLOAT_TYPE px, py, qx, qy
        FLOAT_TYPE r, h
        FLOAT_TYPE dr_dx, dr_dy, dh_dx, dh_dy
        FLOAT_TYPE cost

    cost = 0
    for i in range(edge_indices.shape[0]):
        idx1 = edge_indices[i, 0]
        idx2 = edge_indices[i, 1]

        px = mesh[idx1, 0]
        py = mesh[idx1, 1]
        qx = mesh[idx2, 0]
        qy = mesh[idx2, 1]

        r = c_reglen(px - qx, py - qy,
                     1, 1,
                     &(dr_dx), &(dr_dy))
        h = c_huber(r, rest_lengths[i], sigma,
                    dr_dx, dr_dy,
                    &(dh_dx), &(dh_dy))
        cost += h * all_weight
        dh_dx *= all_weight
        dh_dy *= all_weight

        # update derivs
        d_cost_d_mesh[idx1, 0] += dh_dx
        d_cost_d_mesh[idx1, 1] += dh_dy
        d_cost_d_mesh[idx2, 0] -= dh_dx
        d_cost_d_mesh[idx2, 1] -= dh_dy

    return cost

##################################################
# MESH AREA DERIVS
##################################################
cpdef FLOAT_TYPE area_mesh_derivs(FLOAT_TYPE[:, ::1] mesh,
                                  FLOAT_TYPE[:, ::1] d_cost_d_mesh,
                                  uint32[:, ::1] triangle_indices,
                                  FLOAT_TYPE[:] rest_areas,
                                  FLOAT_TYPE all_weight) nogil:
    cdef:
        int i
        int idx0, idx1, idx2
        FLOAT_TYPE v01x, v01y, v02x, v02y, area, r_area
        FLOAT_TYPE cost, c, dc_da

    cost = 0
    for i in range(triangle_indices.shape[0]):
        idx0 = triangle_indices[i, 0]
        idx1 = triangle_indices[i, 1]
        idx2 = triangle_indices[i, 2]

        v01x = mesh[idx1, 0] - mesh[idx0, 0]
        v01y = mesh[idx1, 1] - mesh[idx0, 1]
        v02x = mesh[idx2, 0] - mesh[idx0, 0]
        v02y = mesh[idx2, 1] - mesh[idx0, 1]

        area = 0.5 * (v02x * v01y - v01x * v02y)
        r_area = rest_areas[i]
        if (area * r_area <= 0):
            c = INFINITY
            dc_da = 0
        else:
            # cost is ((A - A_rest) / A) ^ 2 * A_rest  (last term is for area normalization)
            #
            #      / A  -  A     \ 2
            #      |        rest |     |       |
            #      | ----------- |   * | A     |
            #      \      A      /     |  rest |
            c = all_weight * (((area - r_area) / area) ** 2)
            dc_da = 2 * all_weight * r_area * (area - r_area) / (area ** 3)

        cost += c

        # update derivs
        d_cost_d_mesh[idx1, 0] += dc_da * 0.5 * (-v02y)
        d_cost_d_mesh[idx1, 1] += dc_da * 0.5 * (v02x)
        d_cost_d_mesh[idx2, 0] += dc_da * 0.5 * (v01y)
        d_cost_d_mesh[idx2, 1] += dc_da * 0.5 * (-v01x)

        # sum of negative of above
        d_cost_d_mesh[idx0, 0] += dc_da * 0.5 * (v02y - v01y)
        d_cost_d_mesh[idx0, 1] += dc_da * 0.5 * (v01x - v02x)

    return cost

##################################################
# MESH INTERNAL DERIVS
##################################################
cpdef FLOAT_TYPE internal_grad(FLOAT_TYPE[:, ::1] mesh,
                               FLOAT_TYPE[:, ::1] d_cost_d_mesh,
                               uint32[:, ::1] edge_indices,
                               FLOAT_TYPE[::1] rest_lengths,
                               uint32[:, ::1] triangle_indices,
                               FLOAT_TYPE[::1] triangle_rest_areas,
                               FLOAT_TYPE within_mesh_weight,
                               FLOAT_TYPE within_winsor) except -1:
    cdef:
        FLOAT_TYPE cost = 0

    with nogil:
        cost += internal_mesh_derivs(mesh, d_cost_d_mesh,
                                     edge_indices, rest_lengths,
                                     within_mesh_weight, within_winsor)
        cost += area_mesh_derivs(mesh, d_cost_d_mesh,
                                 triangle_indices, triangle_rest_areas,
                                 within_mesh_weight)
    return cost

##################################################
# MESH EXTERNAL DERIVS
##################################################
cpdef FLOAT_TYPE external_grad(FLOAT_TYPE[:, ::1] mesh1,
                               FLOAT_TYPE[:, ::1] mesh2,
                               FLOAT_TYPE[:, ::1] d_cost_d_mesh1,
                               FLOAT_TYPE[:, ::1] d_cost_d_mesh2,
                               uint32[:, ::1] indices1,
                               FLOAT_TYPE[:, ::1] barys1,
                               uint32[:, ::1] indices2,
                               FLOAT_TYPE[:, ::1] barys2,
                               FLOAT_TYPE between_weight,
                               FLOAT_TYPE between_winsor) except -1:

    return crosslink_mesh_derivs(mesh1, mesh2,
                                 d_cost_d_mesh1, d_cost_d_mesh2,
                                 indices1, indices2,
                                 barys1, barys2,
                                 between_weight, between_winsor)


def compare(x, y, eps, restlen, sigma):
    l, dl_dx, dl_dy = reglen(x, y)
    h0, dh_dx, dh_dy = huber(l, restlen, sigma, dl_dx, dl_dy)
    hx = huber(reglen(x + eps, y)[0], restlen, sigma, dl_dx, dl_dy)[0]
    hy = huber(reglen(x, y + eps)[0], restlen, sigma, dl_dx, dl_dy)[0]
    print x, y, restlen, sigma, "->", (dh_dx, dh_dy), "vs", ((hx - h0) / eps, (hy - h0) / eps)

if __name__ == '__main__':
    eps = 0.00001
    compare(2.0, 10.0, eps, 5.0, 20.0)
    compare(2.0, 10.0, eps, 5.0, 2.0)
    compare(2.0, 10.0, eps, 3.0, 20.0)
    compare(2.0, 10.0, eps, 3.0, 2.0)
    compare(-2.0, 10.0, eps, 3.0, 2.0)

    compare(10.0, 2.0, eps, 5.0, 20.0)
    compare(10.0, 2.0, eps, 5.0, 2.0)
    compare(10.0, 2.0, eps, 3.0, 20.0)
    compare(10.0, 2.0, eps, 3.0, 2.0)
    compare(-10.0, -2.0, eps, 3.0, 2.0)
