import numpy as np
import cv2 as cv


class ConfigParams(object):
    def __init__(self, fx, fy, cx, cy, scale,
                 width, height,
                 depth_near, depth_far):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

        self.virtual_baseline = 0.1

        self.intrinsic = np.array([
            [self.fx, 0, self.cx, 0],
            [0, self.fy, self.cy, 0],
            [0, 0, 1, 0]])

        self.depth_near = depth_near
        self.depth_far = depth_far

        self.width = width
        self.height = height

        self.pnp_min_measurements = 30
        self.pnp_max_iterations = 10
        self.init_min_points = 30

        # BRIEF related parameters
        self.brief_maxCorners = 600
        self.brief_minDistance = 15.0
        self.brief_qualityLevel = 0.001
        self.brief_useHarrisDetector = False
        self.brief_bytes = 32
        self.brief_use_orientation = False
        self.brief_normalizationType = cv.NORM_HAMMING
        self.brief_isCrossCheckEnabled = True

        # ORB related parameters
        # TODO: figure out config parameters for ORB
        self.orb_nfeatures = 1000
        self.orb_scaleFactor = 1.2
        self.orb_nlevels = 8
        self.orb_edgeThreshold = None
        self.orb_firstLevel = None
        self.orb_WTA_K = None
        self.orb_scoreType = None
        self.orb_patchSize = None
        self.orb_fastThreshold = None

        # RANSAC related parameters
        # threshold selected 2*sigma of pixel error
        self.ransac_treshold = 10.0
