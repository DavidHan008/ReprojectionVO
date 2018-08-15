import cv2 as cv
import numpy as np


class Orb:

    # Check out SIFT_create(), if you want to tune the SIFT parameters
    def __init__(self,
                 params,
                 img_1,
                 depth_1,
                 timestamp_1,
                 img_2,
                 depth_2,
                 timestamp_2,):

        self.img_1 = img_1
        self.img_2 = img_2
        self.depth_1 = depth_1
        self.depth_2 = depth_2
        self.timestamp_1 = timestamp_1
        self.timestamp_2 = timestamp_2

        self.orb = cv.ORB_create(nfeatures=params.orb_nfeatures,
                                 scaleFactor=params.orb_scaleFactor,
                                 nlevels=params.orb_nlevels)

        self.key_points_1, self.descriptors_1 = self.orb.detectAndCompute(self.img_1, None)
        self.key_points_2, self.descriptors_2 = self.orb.detectAndCompute(self.img_2, None)

        self.normalizationType = cv.NORM_HAMMING
        # If it is true, Matcher returns only those matches with value (i,j)
        # such that i-th descriptor in set A has j-th descriptor in set B
        # as the best match and vice-versa.
        self.isCrossCheckEnabled = True

        self.bf_matcher = cv.BFMatcher(self.normalizationType, self.isCrossCheckEnabled)
        self.key_point_matches = self.bf_matcher.match(self.descriptors_1, self.descriptors_2)

        # Apply RANSAC with findHomography function
        # (Homography is not relevant but there is a RANSAC functionality builtin inside function)
        good_matches_pos_1 = np.zeros((len(self.key_point_matches), 2), dtype=np.float32)
        good_matches_pos_2 = np.zeros((len(self.key_point_matches), 2), dtype=np.float32)

        for i, match in enumerate(self.key_point_matches):
            good_matches_pos_1[i, :] = self.key_points_1[match.queryIdx].pt
            good_matches_pos_2[i, :] = self.key_points_2[match.trainIdx].pt

        H_martix, mask = cv.findHomography(good_matches_pos_1, good_matches_pos_2, cv.RANSAC, params.ransac_treshold)

        self.inlier_key_point_matches = []
        for i in range(len(mask)):
            if mask[i] == 1:
                self.inlier_key_point_matches.append(self.key_point_matches[i])


    def draw_key_points(self):
        out_img = cv.drawKeypoints(self.img_1,
                                   self.key_points_1,
                                   np.array([]),
                                   flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow('BRIEF Key Points of 1. Frame', out_img)
        # TODO: describe the behavior of showing/closing image window
        cv.waitKey(0)
        out_img = cv.drawKeypoints(self.img_2,
                                   self.key_points_2,
                                   np.array([]),
                                   flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow('BRIEF Key Points of 1. Frame', out_img)
        cv.waitKey(0)
        # cv.destroyAllWindows()

    def draw_all_key_point_matches(self):

        img_matched = cv.drawMatches(self.img_1,
                                     self.key_points_1,
                                     self.img_2,
                                     self.key_points_2,
                                     self.key_point_matches,
                                     None,
                                     flags=2)
        cv.imshow('All Key Point Matches with Brute Froce', img_matched)
        cv.waitKey(0)



    def draw_inlier_key_point_matches(self):
        img_matched = cv.drawMatches(self.img_1,
                                     self.key_points_1,
                                     self.img_2,
                                     self.key_points_2,
                                     self.inlier_key_point_matches,
                                     None,
                                     flags=2)
        cv.imshow('Only Inlier Key Point Matches with Brute Froce', img_matched)
        cv.waitKey(0)