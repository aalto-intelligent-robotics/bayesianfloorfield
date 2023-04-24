import collections
import json
import math
import sys

import numpy as np
from jsonschema import Draft7Validator, exceptions, validators

MIN_DISTANCE = 0.000001

GROUP_DISTANCE_TOLERANCE = 0.1


def wrap_to_pi(a):
    """
    Args:
        a:
    """
    if (a < -math.pi) or (a > math.pi):
        a = (a + math.pi) % (2 * math.pi) - math.pi
    else:
        a = a
    return a


def wrap_to_pi_vec(a):
    """
    Args:
        a:
    """
    res_1 = ((a + math.pi) % (2 * math.pi) - math.pi) * (
        (a < -math.pi) | (a > math.pi)
    )
    res_2 = a * ~((a < -math.pi) | (a > math.pi))
    res = res_1 + res_2
    return res


def wrap_to_2pi(a):
    """
    Args:
        a:
    """
    if (a < 0) or (a > 2 * math.pi):
        a = abs(a % (2 * math.pi))
    else:
        a = a
    return a


def wrap_to_2pi_vec(a):
    """
    Args:
        a:
    """
    res_1 = (abs(a % (2 * math.pi))) * ((a < 0) | (a > 2 * math.pi))
    res_2 = a * ~((a < 0) | (a > 2 * math.pi))
    res = res_1 + res_2
    return res


def distance_cos_2d(p1, p2):
    """
    Args:
        p1 ():
        p2 ():
    """
    dist = 1 - math.cos(p1.Th - p2.Th) + np.linalg.norm([p1.Rho - p2.Rho])
    return dist


def distance_cos_2d_vec(p1, p2):
    """
    Args:
        p1 ():
        p2 ():
    """
    sub = np.subtract(p1, p2)
    dist = 1 - np.cos(sub[:, 0]) + abs(sub[:, 1])
    return dist


def distance_wrap_2d(p1, p2):
    """
    Args:
        p1 ():
        p2 ():
    """
    ad = abs(wrap_to_pi(p1[0] - p2[0]))
    ld = abs(p1[1] - p2[1])
    dist = math.sqrt(ad * ad + ld * ld)
    return dist


def distance_wrap_2d_vec(p1, p2):
    """
    Args:
        p1 ():
        p2 ():
    """
    diff = np.subtract(p1, p2)
    r = np.hsplit(diff, 2)
    ar = r[0].flatten()
    lr = r[1].flatten()
    ad = abs(wrap_to_pi_vec(ar))
    ld = abs(lr)
    ad_ad = np.multiply(ad, ad)
    ld_ld = np.multiply(ld, ld)
    dist = np.sqrt(ad_ad + ld_ld)
    return dist


def distance_wrap_2d_vec_pair(p1, p2):
    """
    Args:
        p1:
        p2:
    """
    shape1 = np.shape(p1)[0]
    shape2 = np.shape(p2)[0]
    p2 = p2.flatten()

    p1_rep = np.tile(p1, (1, shape2))
    p2_rep = np.tile(p2, (shape1, 1))

    diff = np.subtract(p1_rep, p2_rep)

    columns = int(np.shape(diff)[1] / 2)
    diff_col = np.hsplit(diff, columns)
    dist = np.empty((0, columns), float)
    for c in range(0, columns):
        r = np.hsplit(diff_col[c], 2)
        ar = r[0].flatten()
        lr = r[1].flatten()
        ad = abs(wrap_to_pi_vec(ar))
        ld = abs(lr)
        ad_ad = np.multiply(ad, ad)
        ld_ld = np.multiply(ld, ld)
        d = np.array(np.sqrt(ad_ad + ld_ld))
        d = d.reshape(1, columns)
        dist = np.append(dist, d, axis=0)
    return dist


def distance_disjoint_2d(p1, p2):
    """
    Args:
        p1:
        p2:
    """
    ad = abs(wrap_to_pi(p1.Th - p2.Th))
    ld = abs(p1.Rho - p2.Rho)
    distance = collections.namedtuple("distance", ["ad", "ld"])
    dist = distance(ad, ld)
    return dist


def weighted_mean_2d_vec(p, w):
    """
    Args:
        p:
        w:
    """
    a = p[:, 0]
    le = p[:, 1]

    c = np.sum(np.multiply(np.cos(a), w)) / np.sum(w)
    s = np.sum(np.multiply(np.sin(a), w)) / np.sum(w)

    if c >= 0:
        cr_m = np.arctan(s / c)
    else:
        cr_m = np.arctan(s / c) + math.pi
    l_m = np.sum(np.multiply(le, w)) / np.sum(w)
    mean = [wrap_to_2pi(cr_m), l_m]
    return mean


def mean_2d_vec(p):
    """
    Args:
        p:
    """
    a = p[:, 0]
    le = p[:, 1]

    c = np.sum(np.cos(a)) / len(a)
    s = np.sum(np.sin(a)) / len(a)

    if c >= 0:
        cr_m = np.arctan(s / c)
    else:
        cr_m = np.arctan(s / c) + math.pi
    l_m = np.sum(le) / len(le)
    mean = [wrap_to_2pi(cr_m), l_m]
    return mean


class PointGrouper(object):
    def __init__(self, distance=distance_wrap_2d_vec):
        """
        Args:
            distance:
        """
        self.distance = distance

    def group_points(self, points):
        """
        Args:
            points:
        """
        group_assignment = []
        groups = []
        group_index = 0
        for point in points:
            nearest_group_index = self._determine_nearest_group(point, groups)
            if nearest_group_index is None:
                # create new group
                groups.append([point])
                group_assignment.append(group_index)
                group_index += 1
            else:
                group_assignment.append(nearest_group_index)
                groups[nearest_group_index].append(point)
        return np.array(group_assignment)

    def _determine_nearest_group(self, point, groups):
        """
        Args:
            point:
            groups:
        """
        nearest_group_index = None
        index = 0
        for group in groups:
            distance_to_group = self._distance_to_group(point, group)
            if distance_to_group < GROUP_DISTANCE_TOLERANCE:
                nearest_group_index = index
            index += 1
        return nearest_group_index

    def _distance_to_group(self, point, group):
        """
        Args:
            point:
            group:
        """
        min_distance = sys.float_info.max
        for pt in group:
            dist = self.distance(point, pt)
            if dist < min_distance:
                min_distance = dist
        return min_distance


def gaussian_kernel(distance, bandwidth):
    # euclidean_distance = np.sqrt(((distance)**2).sum(axis=1))
    """
    Args:
        distance:
        bandwidth:
    """
    val = (1 / (bandwidth * math.sqrt(2 * math.pi))) * np.exp(
        -0.5 * (distance / bandwidth) ** 2
    )
    return val


def gaussian_kernel_mv(distances, bandwidths):
    # Number of dimensions of the multivariate gaussian
    """
    Args:
        distances:
        bandwidths:
    """
    dim = len(bandwidths)

    # Covariance matrix
    cov = np.multiply(np.power(bandwidths, 2), np.eye(dim))

    # Compute Multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(
        np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1
    )
    val = (
        1
        / np.power((2 * math.pi), (dim / 2))
        * np.power(np.linalg.det(cov), 0.5)
    ) * np.exp(exponent)

    return val


def cutoff(distances, treshold):
    """
    Args:
        distances:
        treshold:
    """
    closer = (distances < treshold) & (distances > 0)
    closer = closer.astype(int)
    count = np.sum(closer, axis=1)
    return count


def extended_validator(json_path, schema_path):
    schema_file = open(schema_path, "r")
    my_schema = json.load(schema_file)

    json_file = open(json_path, "r")
    my_json = json.load(json_file)

    validate_properties = Draft7Validator.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, sub_schema in properties.items():
            if "default" in sub_schema:
                instance.setdefault(prop, sub_schema["default"])

        for error in validate_properties(
            validator,
            properties,
            instance,
            schema,
        ):
            yield error

    ext_validator = validators.extend(
        Draft7Validator,
        {"properties": set_defaults},
    )

    try:
        ext_validator(my_schema).validate(my_json)
    except exceptions.ValidationError as e:
        return False, e
    except exceptions.SchemaError as e:
        return False, e
    return True, my_json


def get_local_settings(
    json_path="config/local_settings.json",
    schema_path="config/local_settings_schema.json",
):
    return extended_validator(json_path, schema_path)


class MeanShift(object):
    def __init__(
        self,
        kernel=gaussian_kernel,
        distance=distance_wrap_2d_vec,
        weight=weighted_mean_2d_vec,
    ):
        """
        Args:
            kernel:
            distance:
            weight:
        """
        self.kernel = kernel
        self.distance = distance
        self.weight = weight

    def cluster(self, points, kernel_bandwidth, iteration_callback=None):
        """
        Args:
            points:
            kernel_bandwidth:
            iteration_callback:
        """
        if iteration_callback:
            iteration_callback(points, 0)
        shift_points = np.array(points)
        max_min_dist = 1
        iteration_number = 0

        history = points
        history = history.tolist()
        for i in range(0, len(history)):
            history[i] = [history[i]]

        still_shifting = [True] * points.shape[0]
        while max_min_dist > MIN_DISTANCE:
            # print max_min_dist
            max_min_dist = 0
            iteration_number += 1
            for i in range(0, len(shift_points)):
                if not still_shifting[i]:
                    continue
                p_new = shift_points[i]
                p_new_start = p_new
                p_new = self._shift_point(p_new, points, kernel_bandwidth)

                dist = self.distance(p_new, p_new_start)

                history[i].append(p_new)
                # print(history[i])

                if dist > max_min_dist:
                    max_min_dist = dist
                if dist < MIN_DISTANCE:
                    still_shifting[i] = False
                shift_points[i] = p_new
            if iteration_callback:
                iteration_callback(shift_points, iteration_number)
        point_grouper = PointGrouper()
        group_assignments = point_grouper.group_points(shift_points.tolist())

        return MeanShiftResult(
            points, shift_points, group_assignments, history
        )

    def _shift_point(self, point, points, kernel_bandwidth):
        # from http://en.wikipedia.org/wiki/Mean-shift
        """
        Args:
            point:
            points:
            kernel_bandwidth:
        """
        points = np.array(points)
        point_rep = np.tile(point, [len(points), 1])
        dist = self.distance(point_rep, points)
        point_weights = self.kernel(dist, kernel_bandwidth)

        shifted_point = self.weight(points, point_weights)
        return shifted_point


class MeanShiftResult:
    def __init__(self, original_points, shifted_points, cluster_ids, history):
        """
        Args:
            original_points:
            shifted_points:
            cluster_ids:
            history:
        """
        self.original_points = original_points
        self.shifted_points = shifted_points
        self.cluster_ids = cluster_ids
        self.history = history
        self.mixing_factors = []
        self.covariances = []
        self.mean_values = []
        # compute GMM parameters
        unique_cluster_ids, counts = np.unique(
            self.cluster_ids, return_counts=True
        )
        for uid, c in zip(unique_cluster_ids, counts):
            self.mixing_factors.append(c / self.cluster_ids.size)
            self.mean_values.append(
                mean_2d_vec(self.original_points[self.cluster_ids == uid, :])
            )
            self.covariances.append(
                np.cov(self.original_points[self.cluster_ids == uid, :])
            )
