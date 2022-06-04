import pickle as pkl
from dataclasses import dataclass
from typing import Tuple, List, Optional

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

    def __init__(self) -> None:
        self.videos: List[VideoProcessor.Video] = []
        self.x: NDArray[np.float64] = np.array([])
        self.y: NDArray[np.float64] = np.array([])

    def add_video(self, video_path: str, split_on_frames_num: Optional[int] = None) -> None:
        frames, shape, klass = VideoProcessor.get_pose_from_file(video_path)
        read_video = VideoProcessor.Video(video_path, frames, shape, klass)

        if split_on_frames_num:
            sub_videos = VideoProcessor.split_video_to_sub_videos(read_video, split_on_frames_num)
            self.videos.extend(sub_videos)
        else:
            self.videos.append(read_video)

    @staticmethod
    def split_video_to_sub_videos(video: Video, frames_num: int) -> List['VideoProcessor.Video']:
        splitted_videos = []
        splitted_frames = VideoProcessor.split_pose_to_frame_seq(video.frames, frames_num)
        for split in splitted_frames:
            splitted_videos.append(VideoProcessor.Video(video.path, split, video.shape, video.klass))
        return splitted_videos

    @staticmethod
    def get_pose_from_file(filename: str) -> Tuple[NDArray[np.float64], Tuple[int, int], str]:
        klass = filename.split('/')[-2]
        with open(filename, 'rb') as file:
            frames = pkl.load(file)

        frame_list = []
        for frame in frames:
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
    def split_pose_to_frame_seq(pose: NDArray[np.float64], frames_num: int) -> NDArray[np.float64]:
        if frames_num > pose.shape[0]:
            return np.array([pose])

        total_frames = []
        start, end = 0, frames_num
        for idx in range(pose.shape[0] - end + 1):
            total_frames.append(pose[start + idx: end + idx])
        return np.array(total_frames)

    @staticmethod
    def preview_video(video: Video) -> animation.FuncAnimation:

        def draw_point(image: NDArray[np.float64], x: int, y: int) -> NDArray[np.float64]:
            image[y][x][0] = 255
            return image

        def draw_pose(pose: NDArray[np.float64], shape: Tuple[int, int]) -> NDArray[np.float64]:
            w, h = shape
            image = np.zeros((h, w, 3))
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
