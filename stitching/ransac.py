import numpy as np
import copy
from common.trans_models import Transforms
from scipy.special import comb


def array_to_string(arr):
    return arr.tostring()


def tri_area(p1, p2, p3):
    scalar = p1.ndim == 1
    p1 = np.atleast_2d(p1)
    p2 = np.atleast_2d(p2)
    p3 = np.atleast_2d(p3)

    area = (p1[:, 0]*(p2[:, 1] - p3[:, 1]) +
            p2[:, 0]*(p3[:, 1] - p1[:, 1]) +
            p3[:, 0]*(p1[:, 1] - p2[:, 1])) / 2.0
    # area might be negative
    return area[0] if scalar else area


def choose_forward(n, k, n_draws):
    """Choose k without replacement from among N
    :param n: number of samples to choose from
    :param k: number of samples to choose
    :param n_draws: number of tuples to return

    returns an n_draws by k array of k-tuples
    """
    if n == 0:
        return np.zeros((0, k), int)
    max_combinations = comb(n, k)
    if max_combinations / 3 < n_draws:
        return choose_forward_dense(n, k, n_draws)
    else:
        return choose_forward_sparse(n, k, n_draws)


def enumerate_choices(n, k):
    """Enumerate all the ways to choose k from n
    returns choices sorted lexigraphically, e.g.
    0, 1
    0, 2
    1, 2
    """
    if k == 1:
        return np.arange(n).reshape(n, 1)
    #
    # Enumerate ways to choose k-1 from n-1 (there are no ways
    # to choose the last, so n-1)
    last = enumerate_choices(n-1, k-1)
    #
    # number of possible choices for each of the previous
    # is from among the remaining.
    #
    n_choices = n - 1 - last[:, -1]
    index = np.hstack([[0], np.cumsum(n_choices)])
    #
    # allocate memory for the result
    #
    result = np.zeros((index[-1], k), int)
    #
    # Create a back pointer into "last" for each element of the new array
    #
    back_ptr = np.zeros(result.shape[0], int)
    back_ptr[index[:-1]] = 1
    back_ptr = np.cumsum(back_ptr) - 1
    #
    # Broadcast the elements of the old array into the new one
    # using the back pointer
    result[:, :-1] = last[back_ptr, :]
    #
    # pull a cumsum trick: fill the last column with all "1" except
    # for the first element which is - its place in the array
    #
    result[1:, -1] = 1
    #
    # Then we subtract the number of entries to get back to zero
    result[index[1:-1], -1] = -n_choices[:-1]+1
    #
    # The last result has to start at the next to last + 1
    # 0, 1 <-
    # 0, 2
    # 1, 2 <-
    result[:, -1] = np.cumsum(result[:, -1]) + result[:, -2] + 1
    return result


def choose_forward_dense(n, k, n_draws):
    """Choose k without replacement from among N where n_draws ~ # of combos
    :param n: number of samples to choose from
    :param k: number of samples to choose
    :param n_draws: number of tuples to return
    returns an n_draws by k array of k-tuples
    """
    all_possible = enumerate_choices(n, k)
    choices = np.random.choice(np.arange(all_possible.shape[0]), n_draws,
                               replace=False)
    return all_possible[choices]


def choose_forward_sparse(n, k, n_draws):
    """Choose k without replacement from among N where n_draws << combos
    :param n: number of samples to choose from
    :param k: number of samples to choose
    :param n_draws: number of tuples to return
    returns an n_draws by k array of k-tuples
    """
    # We assume that there is very little chance of collisions, and we choose a few more than asked
    extra = int(np.sqrt(n_draws)) + 1
    n1_draws = n_draws + extra
    choices = np.random.randint(0, n, (n1_draws, k))
    while True:
        #
        # We sort in the k direction to get indices from low to high per draw
        #
        choices.sort(axis=1)
        #
        # We then argsort and duplicates should be adjacent in argsortland
        #
        order = np.lexsort([choices[:, k_] for k_ in range(k-1, -1, -1)])
        to_remove = np.where(
            np.all(choices[order[:-1]] == choices[order[1:]], axis=1))
        result = np.delete(choices, order[to_remove], axis=0)
        if len(result) >= n_draws:
            return result
        # Add some more choices if we didn't get enough
        choices = np.vstack((result, np.random.randint(0, n, (extra, k))))


def check_model_stretch(model_matrix, max_stretch=0.25):
    # Use the eigen values to validate the stretch
    assert(0.0 <= max_stretch <= 1.0)
    eig_vals, _ = np.linalg.eig(model_matrix)
    # Note that this also takes flipping as an incorrect transformation
    valid_eig_vals = [eig_val for eig_val in eig_vals if 1.0 - max_stretch <= eig_val <= 1.0 + max_stretch]
    return len(valid_eig_vals) == 2


def filter_triangles(m0, m1, choices,
                     max_stretch=0.25,
                     max_area=.2):
    """Filter a set of match choices
    :param m0: set of points in one domain
    :param m1: set of matching points to m0 in another domain
    :param choices: an N x 3 array of triangles
    :param max_stretch: filter out a choice if it shrinks by 1-max_stretch
        or stretches by 1+max_stretch
    :param max_area: filter out a choice if the area of m1's triangle is
        less than 1-max_area or more than 1 + max_area

    If a triangle in m0 has a different absolute area than in m1, exclude it
    If the eigenvalues of the affine transform array indicate a shrink of
    a factor of 1-max_stretch or a stretch of a factor of 1+max_stretch,
    exclude
    """
    pt1a, pt2a, pt3a = [m0[choices][:, _, :] for _ in range(3)]
    pt1b, pt2b, pt3b = [m1[choices][:, _, :] for _ in range(3)]
    areas_a = tri_area(pt1a, pt2a, pt3a)
    areas_b = tri_area(pt1b, pt2b, pt3b)
    area_ratio = areas_a / (areas_b + np.finfo(areas_b.dtype).eps)
    mask = (area_ratio <= 1+max_area) & (area_ratio >= 1-max_area)
    choices = choices[mask]
    x = m0[choices]
    y = m1[choices]

    pc = np.mean(x, axis=1)
    qc = np.mean(y, axis=1)

    delta1 = x - pc[:, np.newaxis, :]
    delta2 = y - qc[:, np.newaxis, :]

    a00 = np.sum(delta1[:, 0] * delta1[:, 0], axis=1)
    a01 = np.sum(delta1[:, 0] * delta1[:, 1], axis=1)
    a11 = np.sum(delta1[:, 1] * delta1[:, 1], axis=1)
    b00 = np.sum(delta1[:, 0] * delta2[:, 0], axis=1)
    b01 = np.sum(delta1[:, 0] * delta2[:, 1], axis=1)
    b10 = np.sum(delta1[:, 1] * delta2[:, 0], axis=1)
    b11 = np.sum(delta1[:, 1] * delta2[:, 1], axis=1)

    det = a00 * a11 - a01 * a01 + np.finfo(a00.dtype).eps

    m00 = (a11 * b00 - a01 * b10) / det
    m01 = (a00 * b10 - a01 * b00) / det
    m10 = (a11 * b01 - a01 * b11) / det
    m11 = (a00 * b11 - a01 * b01) / det
    det = m00 * m11 - m01*m10
    #
    # The eigenvalues, L, are the roots of
    # L**2 - (m00 + m11) * L + det = 0
    #
    # L = ((m00 + m11) +/- sqrt((m00+m11) **2 - 4 * det)) / 2
    #
    b = m00 + m11
    b2_minus_4ac = b*b + np.finfo(b.dtype).eps - 4 * det
    mask = b2_minus_4ac >= 0
    choices, b, b2_minus_4ac = choices[mask], b[mask], b2_minus_4ac[mask]
    sb2_minus_4ac = np.sqrt(b2_minus_4ac)
    l1 = (b + sb2_minus_4ac) / 2
    l2 = (b - sb2_minus_4ac) / 2

    mask = (l1 <= 1+max_stretch) & (l2 >= 1-max_stretch)
    return choices[mask]


def ransac(matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier,
           det_delta=0.35, max_stretch=0.25):
    assert(len(matches[0]) == len(matches[1]))

    best_model = None
    best_model_score = 0
    # The higher, the better
    best_inlier_mask = None
    best_model_mean_dists = 0
    proposed_model = Transforms.create(target_model_type)

    if proposed_model.MIN_MATCHES_NUM > matches.shape[1]:
        print("RANSAC cannot find a good model because "
              "the number of initial matches ({}) is too small.".format(matches.shape[1]))
        return None, None, None

    # Avoiding repeated indices permutations using a dictionary
    # Limit the number of possible matches that we can search for using n choose k
    max_combinations = int(comb(len(matches[0]), proposed_model.MIN_MATCHES_NUM))
    max_iterations = min(iterations, max_combinations)
    choices = choose_forward(len(matches[0]),
                             proposed_model.MIN_MATCHES_NUM,
                             max_iterations)
    if proposed_model.MIN_MATCHES_NUM == 3:
        choices = filter_triangles(matches[0], matches[1], choices,
                                   max_stretch=max_stretch)
    for min_matches_indexes in choices:
        # Try to fit them to the model
        if not proposed_model.fit(matches[0][min_matches_indexes], matches[1][min_matches_indexes]):
            continue
        model_matrix = proposed_model.get_matrix()[:2, :2]
        if proposed_model.MIN_MATCHES_NUM == 3:
            # check the stretch of the new transformation
            if not check_model_stretch(model_matrix, max_stretch):
                continue
            # if the proposed model distorts the image too much, skip the model
            det = np.linalg.det(model_matrix)
            if det < 1.0 - det_delta or det > 1.0 + det_delta:
                continue
        # print "proposed_model", proposed_model.to_str()
        # Verify the new model 
        proposed_model_score, inlier_mask, proposed_model_mean = proposed_model.score(matches[0], matches[1], epsilon,
                                                                                      min_inlier_ratio, min_num_inlier)
        # print "proposed_model_score", proposed_model_score
        if proposed_model_score > best_model_score:
            best_model = copy.deepcopy(proposed_model)
            best_model_score = proposed_model_score
            best_inlier_mask = inlier_mask
            best_model_mean_dists = proposed_model_mean
    return best_inlier_mask, best_model, best_model_mean_dists


def filter_after_ransac(candidates, model, max_trust, min_num_inliers):
    """
    Estimate the AbstractModel and filter potential outliers by robust iterative regression.
    This method performs well on data sets with low amount of outliers (or after RANSAC).
    """
    # copy the model
    new_model = copy.deepcopy(model)
    dists = []

    # iteratively find a new model, by fitting the candidates, and removing those that are farther
    # than max_trust*median-distance until the set of remaining candidates does not change its size

    # for the initial iteration, we set a value that is higher the given candidates size
    prev_iteration_num_inliers = candidates.shape[1] + 1

    # keep a copy of the candidates that will be changed due to fitting and error 
    inliers = copy.copy(candidates[0])

    # keep track of the candidates using a mask
    candidates_mask = np.ones((candidates.shape[1]), dtype=np.bool)

    while prev_iteration_num_inliers > np.sum(candidates_mask):
        prev_iteration_num_inliers = np.sum(candidates_mask)
        # Get the inliers and their corresponding matches
        inliers = candidates[0][candidates_mask]
        to_image_candidates = candidates[1][candidates_mask]

        # try to fit the model
        if not new_model.fit(inliers, to_image_candidates):
            break

        # get the median error (after transforming the points)
        pts_after_transform = new_model.apply(inliers)
        dists = np.sqrt(np.sum((pts_after_transform - to_image_candidates) ** 2, axis=1))
        median = np.median(dists)
        inliers_mask = dists <= (median * max_trust)
        candidates_mask[candidates_mask == True] = inliers_mask

    if np.sum(candidates_mask) < min_num_inliers:
        return None, None, -1

    return new_model, candidates_mask, np.mean(dists)


def filter_matches(matches, target_model_type, iterations, epsilon, min_inlier_ratio, min_num_inlier,
                   max_trust, det_delta=0.35, max_stretch=0.25):
    """Perform a RANSAC filtering given all the matches"""
    new_model = None
    filtered_matches = None

    # Apply RANSAC
    # print "Filtering {} matches".format(matches.shape[1])
    print("pre-ransac matches count: {}".format(matches.shape[1]))
    inliers_mask, model, _ = ransac(matches, target_model_type, iterations, epsilon, min_inlier_ratio,
                                    min_num_inlier, det_delta, max_stretch)
    if inliers_mask is None:
        print("post-ransac matches count: 0")
    else:
        print("post-ransac matches count: {}".format(inliers_mask.shape[0]))

    # Apply further filtering
    if inliers_mask is not None:
        inliers = np.array([matches[0][inliers_mask], matches[1][inliers_mask]])
        # print "Found {} good matches out of {} matches after RANSAC".format(inliers.shape[1], matches.shape[1])
        new_model, filtered_inliers_mask, mean_dists = filter_after_ransac(inliers, model, max_trust, min_num_inlier)
        filtered_matches = np.array([inliers[0][filtered_inliers_mask], inliers[1][filtered_inliers_mask]])
    if filtered_matches is None:
        print("post-ransac-filter matches count: 0")
    else:
        print("post-ransac-filter matches count: {}".format(filtered_matches.shape[1]))
    return new_model, filtered_matches
