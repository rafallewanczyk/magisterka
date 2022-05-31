import numpy as np
import pickle as pkl


class DatasetProcessor:
    def __init__(self):
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])

    @staticmethod
    def get_pose_from_file(filename):
        with open(filename, 'rb') as file:
            frames = pkl.load(file)

        all_frames = []
        for frame in frames:
            keypoints = frame['result'][0]['keypoints']
            keypoints -= (np.min(keypoints[:, 0]), np.min(keypoints[:, 1]))
            all_frames.append(keypoints)
        all_frames = np.array(all_frames)

        w = np.ceil(np.max(all_frames[:, :, 0])).astype(int)
        h = np.ceil(np.max(all_frames[:, :, 1])).astype(int)
        return all_frames[:, ] / (w, h), (w, h)

    @staticmethod
    def split_pose_to_frame_seq(pose, frames_num):
        if frames_num > pose.shape[0]:
            return np.array([pose])

        total_frames = []
        start, end = 0, frames_num
        for idx in range(pose.shape[0] - end + 1):
            total_frames.append(pose[start + idx: end + idx])
        return np.array(total_frames)
