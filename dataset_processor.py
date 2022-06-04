import pickle as pkl
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

import numpy as np
from matplotlib import pyplot as plt, animation
from numpy.typing import NDArray


class VideoProcessor:
    @dataclass
    class Video:
        path: str
        frames: NDArray[np.float64]
        shape: Tuple[int, int]
        klass: str

    def __init__(self, sub_video_frames_num: int) -> None:
        self.videos: List[VideoProcessor.Video] = []
        self.sub_video_frames_num: int = sub_video_frames_num

    def add_video(self, video_path: str) -> None:
        frames, shape, klass = VideoProcessor._get_pose_from_file(video_path)
        read_video = VideoProcessor.Video(video_path, frames, shape, klass)

        if self.sub_video_frames_num:
            sub_videos = VideoProcessor._split_video_to_sub_videos(read_video, self.sub_video_frames_num)
            self.videos.extend(sub_videos)
        else:
            self.videos.append(read_video)

    def get_num_of_videos(self) -> int:
        return len(self.videos)

    def load_data(self, test_split_ratio: float) \
            -> Tuple[
                Tuple[NDArray[np.float64], NDArray[np.int64]],
                Tuple[NDArray[np.float64], NDArray[np.int64]],
                Dict[int, str]]:

        if test_split_ratio > 1.0 or test_split_ratio < 0:
            raise ValueError("Incorrect split ratio, use values <0;1>")
        split_point = int(self.get_num_of_videos() * test_split_ratio)
        klass_to_int, int_to_klass = self._get_klass_dict()

        np.random.seed(1)
        p = np.random.permutation(self.get_num_of_videos())

        x = np.array([self._flatten_last_dim(video.frames) for video in self.videos])
        y = np.array([klass_to_int[video.klass] for video in self.videos])

        return (x[p[split_point:]], y[p[split_point:]]), (x[p[:split_point]], y[p[:split_point]]), int_to_klass

    def _get_klass_dict(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        klass_to_int = {}
        int_to_klass = {}
        klass_num = 0

        for video in self.videos:
            if video.klass in klass_to_int:
                continue
            klass_to_int[video.klass] = klass_num
            int_to_klass[klass_num] = video.klass
            klass_num += 1
        return klass_to_int, int_to_klass

    def _flatten_last_dim(self, array: NDArray[np.float64]) -> NDArray[np.float64]:
        return array.reshape((self.sub_video_frames_num, 34))

    @staticmethod
    def _split_video_to_sub_videos(video: Video, frames_num: int) -> List['VideoProcessor.Video']:
        splitted_videos = []
        splitted_frames = VideoProcessor._split_pose_to_frame_seq(video.frames, frames_num)
        if splitted_frames is not None:
            for split in splitted_frames:
                splitted_videos.append(VideoProcessor.Video(video.path, split, video.shape, video.klass))
        return splitted_videos

    @staticmethod
    def _get_pose_from_file(filename: str) -> Tuple[NDArray[np.float64], Tuple[int, int], str]:
        klass = filename.split('/')[-2]
        with open(filename, 'rb') as file:
            frames = pkl.load(file)

        frame_list = []
        for frame in frames:
            if not frame or 'result' not in frame or len(frame['result']) < 1:
                continue

            keypoints = frame['result'][0]['keypoints']
            frame_list.append(keypoints)
        frame_array = np.array(frame_list)

        x_offset = np.min(frame_array[:, :, 0])
        y_offset = np.min(frame_array[:, :, 1])
        frame_array = frame_array[:, ] - (x_offset, y_offset)

        w = np.ceil(np.max(frame_array[:, :, 0])).astype(int)
        h = np.ceil(np.max(frame_array[:, :, 1])).astype(int)

        return frame_array[:, ] / (w, h), (w, h), klass

    @staticmethod
    def _split_pose_to_frame_seq(pose: NDArray[np.float64], frames_num: int) -> Optional[NDArray[np.float64]]:
        if frames_num > pose.shape[0]:
            return None

        total_frames = []
        start, end = 0, frames_num
        for idx in range(pose.shape[0] - end + 1):
            total_frames.append(pose[start + idx: end + idx])
        return np.array(total_frames)

    @staticmethod
    def preview_video(video: Video) -> animation.FuncAnimation:

        def draw_point(image: NDArray[np.int64], x: int, y: int) -> NDArray[np.int64]:
            image[y][x][0] = 255
            return image

        def draw_pose(pose: NDArray[np.float64], shape: Tuple[int, int]) -> NDArray[np.int64]:
            w, h = shape
            image = np.zeros((h, w, 3)).astype(np.int64)
            for point in pose:
                image = draw_point(image, int(point[0] * w), int(point[1] * h))
            return image

        def animate_func(i: int) -> List[plt.axes]:
            im.set_array(draw_pose(video.frames[i], video.shape))
            return [im]

        fig = plt.figure(figsize=(8, 8))
        draw = draw_pose(video.frames[0], video.shape)
        im = plt.imshow(draw, interpolation='none', aspect='auto', vmin=0, vmax=1)

        anim = animation.FuncAnimation(
            fig,
            animate_func,
            frames=video.frames.shape[0], interval=1000 / 30)  # in ms
        return anim
