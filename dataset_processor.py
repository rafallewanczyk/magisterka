import pickle as pkl
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class DatasetProcessor:
    def __init__(self) -> None:
        self.train_x: NDArray[np.float64] = np.array([])
        self.train_y: NDArray[np.float64] = np.array([])
        self.test_x: NDArray[np.float64] = np.array([])
        self.test_y: NDArray[np.float64] = np.array([])

    @staticmethod
    def get_pose_from_file(filename: str) -> Tuple[NDArray[np.float64], Tuple[int, int], str]:
        klass = filename.split('/')[-2]
        with open(filename, 'rb') as file:
            frames = pkl.load(file)

        frame_list = []
        for frame in frames:
            keypoints = frame['result'][0]['keypoints']
            keypoints -= (np.min(keypoints[:, 0]), np.min(keypoints[:, 1]))
            frame_list.append(keypoints)
        frame_array = np.array(frame_list)

        w = np.ceil(np.max(frame_array[:, :, 0])).astype(int)
        h = np.ceil(np.max(frame_array[:, :, 1])).astype(int)
        return frame_array[:, ] / (w, h), (w, h), klass

    @staticmethod
    def split_pose_to_frame_seq(pose: NDArray[np.float64], frames_num: int) -> NDArray[np.float64]:
        if frames_num > pose.shape[0]:
            return np.array([pose])

        total_frames = []
        start, end = 0, frames_num
        for idx in range(pose.shape[0] - end + 1):
            total_frames.append(pose[start + idx: end + idx])
        return np.array(total_frames)
