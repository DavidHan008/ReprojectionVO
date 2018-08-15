import numpy as np
import ceres_reprojection
from pyquaternion import Quaternion


def pixel_coords_2d_to_camera_coords_3d(cam, pixel_coords, depth):
    coords = np.array(pixel_coords, dtype=int)
    ix = coords[:, 0]
    iy = coords[:, 1]
    depth = depth[iy, ix]

    zs = depth / cam.scale
    xs = (ix - cam.cx) * zs / cam.fx
    ys = (iy - cam.cy) * zs / cam.fy
    ones = np.ones((len(zs), 1))
    return np.column_stack([xs, ys, zs, ones])


class ReprojectionOptimizer:
    def __init__(self,
                 params,
                 ground_truth_init_transformation,
                 init_transformation=np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float64).reshape(1, 8)):

        self.init_transformation = init_transformation
        self.accumulated_transformation = ground_truth_init_transformation
        self.current_transformation = init_transformation
        self.params = params
        self.K = params.intrinsic
        self.Rt = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

        self.result_ceres = np.zeros((0, 8), dtype=np.float64)
        self.residuals_ceres = None

        self.descriptors_1 = None
        self.descriptors_2 = None
        self.key_points_1 = None
        self.key_points_2 = None
        self.inlier_key_point_matches = None
        self.depth_1 = None
        self.depth_2 = None
        self.timestamp_1 = None
        self.timestamp_2 = None

        self.measured_k_1_pixel_2d_pos = None
        self.measured_k_camera_3d_pos = None

    def update_measurements(self, measurements):
        self.descriptors_1 = measurements.descriptors_1
        self.descriptors_2 = measurements.descriptors_2
        self.key_points_1 = measurements.key_points_1
        self.key_points_2 = measurements.key_points_2
        self.inlier_key_point_matches = measurements.inlier_key_point_matches
        self.depth_1 = measurements.depth_1
        self.depth_2 = measurements.depth_2
        self.timestamp_1 = measurements.timestamp_1
        self.timestamp_2 = measurements.timestamp_2

    def filter_matches(self):

        # measure pixels in 2. frame
        self.measured_k_1_pixel_2d_pos = np.zeros((0, 2), dtype=np.float64)

        # preparing unfiltered measured_k_1_pixel_2d_pos
        tmp_measured_k_1_pixel_2d_pos = np.zeros((len(self.inlier_key_point_matches), 2), dtype=np.float32)
        for i, match in enumerate(self.inlier_key_point_matches):
            tmp_measured_k_1_pixel_2d_pos[i, :] = self.key_points_2[match.trainIdx].pt

        # converting 2d pixel coords + depth to 3d point clouds in 1. frame
        pixel_2d_pos = np.zeros((len(self.inlier_key_point_matches), 2), dtype=np.float32)
        for i, match in enumerate(self.inlier_key_point_matches):
            pixel_2d_pos[i, :] = self.key_points_1[match.queryIdx].pt
        self.measured_k_camera_3d_pos = pixel_coords_2d_to_camera_coords_3d(self.params, pixel_2d_pos, self.depth_1)

        # filter key points having zero depth information
        tmp_inlier_key_point_matches = []
        tmp_measured_k_camera_3d_pos = np.zeros((0, 3), dtype=np.float64)
        for i, point in enumerate(self.measured_k_camera_3d_pos):
            if not (self.params.depth_near <= point[2] <= self.params.depth_far):
                continue

            # remove zero depth key points from inlier_matches also
            tmp_inlier_key_point_matches.append(self.inlier_key_point_matches[i])
            # prep measured pixels in 2. frame
            self.measured_k_1_pixel_2d_pos = np.vstack((self.measured_k_1_pixel_2d_pos,
                                                        tmp_measured_k_1_pixel_2d_pos[i, :]))
            # prep measure point clouds in 1. frame
            tmp_measured_k_camera_3d_pos = np.vstack((tmp_measured_k_camera_3d_pos,
                                                      self.measured_k_camera_3d_pos[i, 0:3]))
        self.inlier_key_point_matches = tmp_inlier_key_point_matches
        self.measured_k_camera_3d_pos = tmp_measured_k_camera_3d_pos

    def min_projection_error(self):

        T = ceres_reprojection.optimize(self.init_transformation[0, 1:8],
                                        self.measured_k_1_pixel_2d_pos, # observed pixels
                                        self.measured_k_camera_3d_pos)  # point clouds
        t = np.zeros(shape=(1, 8))
        t[0, 0] = self.timestamp_2

        # converting camera coords to world coords
        # swap x and y axis and multiply w -1
        q = Quaternion(T[6], -T[4], -T[3], T[5])    # qw, qx, qy, qz
        qw = q.normalised[0]
        qx = q.normalised[1]
        qy = q.normalised[2]
        qz = q.normalised[3]

        tx = -T[1]
        ty = -T[0]
        tz = T[2]

        t[0, 1:4] = [tx, ty, tz]  # translation
        t[0, 4:8] = [qx, qy, qz, qw]  # orientation

        self.current_transformation = t

    def get_residuals(self):
        res = ceres_reprojection.calculate_residuals(self.current_transformation[0, 1:8],
                                               self.measured_k_1_pixel_2d_pos,
                                               self.measured_k_camera_3d_pos)
        num_rows = int(len(res)/2)
        if self.residuals_ceres is None:
            self.residuals_ceres = np.zeros((num_rows, 2), dtype=np.float64)
            self.residuals_ceres[:, 0] = np.array(res[0:num_rows], dtype=np.float64)
            self.residuals_ceres[:, 1] = np.array(res[num_rows:len(res)], dtype=np.float64)
        else:
            tmp_res = np.zeros((num_rows, 2), dtype=np.float64)
            tmp_res[:, 0] = np.array(res[0:num_rows], dtype=np.float64)
            tmp_res[:, 1] = np.array(res[num_rows:len(res)], dtype=np.float64)

            current_row_size = self.residuals_ceres.shape[0]
            current_column_size = self.residuals_ceres.shape[1]
            if num_rows > current_row_size:
                tmp_nans = np.empty((num_rows-current_row_size, current_column_size), dtype=np.float64)
                tmp_nans[:] = np.nan
                self.residuals_ceres = np.vstack((self.residuals_ceres, tmp_nans))
            elif num_rows < current_row_size:
                tmp_nans = np.empty((current_row_size-num_rows, 2), dtype=np.float64)
                tmp_nans[:] = np.nan
                tmp_res = np.vstack((tmp_res, tmp_nans))

            self.residuals_ceres = np.hstack((self.residuals_ceres, tmp_res))

    def update_position(self):
        # add timestamp
        self.accumulated_transformation[0, 0] = self.current_transformation[0, 0]

        # translate
        self.accumulated_transformation[0, 1:4] = self.accumulated_transformation[0, 1:4] + self.current_transformation[0, 1:4]

        # rotate
        prev_rotation = Quaternion(self.accumulated_transformation[0, 7],
                                   self.accumulated_transformation[0, 4],
                                   self.accumulated_transformation[0, 5],
                                   self.accumulated_transformation[0, 6])
        current_rotation = Quaternion(self.current_transformation[0, 7],
                                      self.current_transformation[0, 4],
                                      self.current_transformation[0, 5],
                                      self.current_transformation[0, 6])
        product_rotation = current_rotation * prev_rotation
        self.accumulated_transformation[0, 7] = product_rotation.normalised[0]    # qw
        self.accumulated_transformation[0, 4] = product_rotation.normalised[1]    # qx
        self.accumulated_transformation[0, 5] = product_rotation.normalised[2]    # qy
        self.accumulated_transformation[0, 6] = product_rotation.normalised[3]    # qz

        # finish up update process
        self.result_ceres = np.vstack((self.result_ceres, self.accumulated_transformation))
