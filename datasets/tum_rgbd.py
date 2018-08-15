import cv2 as cv
import numpy as np
import csv
from .parameters import ConfigParams


# Default folder structure for tum rgbd dataset
# if you change any files/folders location, provide it through TumRgbd
class TumRgbd():
    def __init__(self,
                 tum_main_path='../tum_rgbd_dataset/rgbd_dataset_freiburg1_xyz/',
                 img_format='.png',
                 rgb_file='rgb.txt',
                 rgbd_path='rgb/',
                 depth_file='depth.txt',
                 depth_path='depth/',
                 ground_truth_file='groundtruth.txt',
                 accelerometer_file='accelerometer.txt'):

        self.img_format = img_format
        self.tum_main_path = tum_main_path
        self.depth_file = tum_main_path + depth_file
        self.rgb_file = tum_main_path + rgb_file
        self.rgb_path = tum_main_path
        self.depth_path = tum_main_path
        self.ground_truth_file = tum_main_path + ground_truth_file
        self.accelerometer_file = tum_main_path + accelerometer_file
        self.tum_rgd_img_1 = None
        self.tum_rgd_img_1_timestamp = None
        self.tum_depth_img_1 = None
        self.tum_depth_img_1_timestamp = None

        # intrinsic parameters for TUM RGB-D dataset
        self.config_params = ConfigParams(525.0, 525.0, 319.5, 239.5, 5000, 640, 480, 0.1, 10)


        self.rgb_list = self.read_file_list(self.rgb_file)
        self.depth_list = self.read_file_list(self.depth_file)
        self.matches = self.associate(self.rgb_list, self.depth_list, float(0.0), float(0.02))
        self.ground_truth_init_transformation = self.read_init_ground_truth(self.ground_truth_file)


    # Taken from TUM tools
    def read_file_list(self, filename):
        """
        Reads a trajectory from a text file.

        File format:
        The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
        and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

        Input:
        filename -- File name

        Output:
        dict -- dictionary of (stamp,data) tuples

        """
        file = open(filename)
        data = file.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                len(line) > 0 and line[0] != "#"]
        list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
        return dict(list)

    # Taken from TUM tools
    def associate(self, first_list, second_list, offset, max_difference):
        """
        Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
        to find the closest match for every input tuple.

        Input:
        first_list -- first dictionary of (stamp,data) tuples
        second_list -- second dictionary of (stamp,data) tuples
        offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
        max_difference -- search radius for candidate generation

        Output:
        matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

        """
        first_keys = list(first_list.keys())
        second_keys = list(second_list.keys())
        potential_matches = [(abs(a - (b + offset)), a, b)
                             for a in first_keys
                             for b in second_keys
                             if abs(a - (b + offset)) < max_difference]
        potential_matches.sort()
        matches = []
        for diff, a, b in potential_matches:
            if a in first_keys and b in second_keys:
                first_keys.remove(a)
                second_keys.remove(b)
                matches.append((a, b))

        matches.sort()
        return matches

    def get_all_rgbd_pair_images(self):

        depth = []
        rgb = []
        for i in range(len(self.matches)):
            rgb_image_name = self.rgb_path + str(self.matches[i][0])+self.img_format
            rgb.append(cv.imread(rgb_image_name))
            depth_image_name = self.depth_path + str(self.matches[i][1])+self.img_format
            depth.append(cv.imread(depth_image_name))

        return rgb, depth

    def pop_rgbd_pair_image(self):

        try:
            a_match = self.matches.pop(0)

            rgb_timestamp = str(a_match[0])
            rgb_image_name = self.rgb_path + self.rgb_list[a_match[0]][0]
            rgb_data = cv.imread(rgb_image_name, cv.IMREAD_GRAYSCALE)

            depth_timestamp = str(a_match[1])
            depth_image_name = self.depth_path + self.depth_list[a_match[1]][0]
            depth_data = cv.imread(depth_image_name, -cv.IMREAD_ANYDEPTH)

            return rgb_data, rgb_timestamp, depth_data, depth_timestamp
        except IndexError:
            return [], [], [], []

    def read_init_ground_truth(self, filename):
        # getting initial frame to align pose result
        self.tum_rgd_img_1, self.tum_rgd_img_1_timestamp, self.tum_depth_img_1, self.tum_depth_img_1_timestamp = \
            self.pop_rgbd_pair_image()
        # reading ground truth transformation
        ground_truth = np.zeros((0, 8), dtype=np.float64)
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                if row[0] != '#':
                    ground_truth = np.vstack((ground_truth, np.array(row, dtype=np.float64)))
        idx_1 = np.argmin(np.abs(ground_truth[:, 0] - float(self.tum_rgd_img_1_timestamp)))
        ground_truth_init_transformation = ground_truth[idx_1, :].copy()
        ground_truth_init_transformation = ground_truth_init_transformation.reshape(1, 8)

        return ground_truth_init_transformation
