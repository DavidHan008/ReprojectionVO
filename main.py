# TODO: Here your tested features go! Encapsulate and Do your versioning


import numpy as np
import csv

from datasets.tum_rgbd import TumRgbd
from vo.feature.brief import Brief
from vo.feature.orb import Orb
from vo.optimization.ceres_reprojection_error import ReprojectionOptimizer

"""Initialization Stuff"""

# reading tum dataset
tum_dataset = TumRgbd()
params = tum_dataset.config_params

# getting initial frame to align pose result
tum_rgd_img_1 = tum_dataset.tum_rgd_img_1
tum_rgd_img_1_timestamp = tum_dataset.tum_rgd_img_1_timestamp
tum_depth_img_1 = tum_dataset.tum_depth_img_1
tum_depth_img_1_timestamp = tum_dataset.tum_depth_img_1_timestamp
ground_truth_init_transformation = tum_dataset.ground_truth_init_transformation

# ceres reprojection error minimizer and initialize transformation with the ground truth
optimizer = ReprojectionOptimizer(tum_dataset.config_params,
                                  ground_truth_init_transformation=ground_truth_init_transformation)

# variables to save pose results
result_ceres = np.zeros((0, 8))


"""Main Loop of Visual Odometry"""
i = 0
while True:

    # getting 2. frame
    tum_rgd_img_2, tum_rgd_img_2_timestamp,  tum_depth_img_2, tum_depth_img_2_timestamp = \
        tum_dataset.pop_rgbd_pair_image()

    # finish if there is no img left
    if tum_rgd_img_2 == []:
        break

    # use BRIEF+BRUTE_FORCE to detect&match features
    measurements = Orb(tum_dataset.config_params,
                         tum_rgd_img_1, tum_depth_img_1, tum_rgd_img_1_timestamp,
                         tum_rgd_img_2, tum_depth_img_2, tum_rgd_img_2_timestamp)
    # measurements.draw_all_key_point_matches()
    # measurements.draw_inlier_key_point_matches()

    # update current feature matchings
    optimizer.update_measurements(measurements)
    # RANSAC
    optimizer.filter_matches()

    # optimization
    optimizer.min_projection_error()

    # # residual vectors for optimized transformation
    # optimizer.get_residuals()

    # accumulate current transformation to total result
    optimizer.update_position()

    # closing up current iteration by assign 2. frame to 1. frame
    i += 1
    tum_rgd_img_1, tum_rgd_img_1_timestamp,  tum_depth_img_1, tum_depth_img_1_timestamp = \
        tum_rgd_img_2, tum_rgd_img_2_timestamp, tum_depth_img_2, tum_depth_img_2_timestamp


# Writing out pose results
result_file = 'logs/pos_results_ceres.txt'
with open(result_file, 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(optimizer.result_ceres)

# Visualize results with evo
from visualization import evo_traj, evo_rpe, evo_ape
evo_traj.launch(tum_dataset.ground_truth_file, result_file)
evo_rpe.launch(tum_dataset.ground_truth_file, result_file)
evo_ape.launch(tum_dataset.ground_truth_file, result_file)

print('end')


