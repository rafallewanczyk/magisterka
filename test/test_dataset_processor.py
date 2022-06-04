from typing import cast

import numpy as np
from numpy.typing import NDArray

from dataset_processor import VideoProcessor


class TestDatasetProcessor:

    def test_split_pose_should_return_2_sequences(self) -> None:
        # given
        example_pose = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        frame_num = 3

        # when
        out_poses = cast(NDArray[np.float64],
                         VideoProcessor._split_pose_to_frame_seq(pose=example_pose, frames_num=frame_num))

        # then
        assert np.array_equal(out_poses, np.array([
            [
                [1, 1],
                [2, 2],
                [3, 3]
            ], [
                [2, 2],
                [3, 3],
                [4, 4]
            ],
        ]))

    def test_split_pose_should_return_same_sequence(self) -> None:
        # given
        example_pose = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        frame_num = 5

        # when
        out_poses = cast(NDArray[np.float64],
                         VideoProcessor._split_pose_to_frame_seq(pose=example_pose, frames_num=frame_num))

        # then
        assert np.array_equal(out_poses, np.array([example_pose]))

    def test_split_pose_should_not_return_any_sequences(self) -> None:
        # given
        example_pose = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        frame_num = 5

        # when
        out_poses = VideoProcessor._split_pose_to_frame_seq(pose=example_pose, frames_num=frame_num)

        # then
        assert out_poses is None

    def test_add_video(self) -> None:
        # given
        split_to_frames = 60
        frames_num = 487
        video_processor = VideoProcessor(split_to_frames)

        # when
        video_processor.add_video('test_data/487_frames.pkl')
        (x_train, y_train), (x_test, y_test), int_to_klass = video_processor.load_data(0)

        # then
        assert x_train.shape == (frames_num-split_to_frames+1, split_to_frames, 34)
        assert x_train.dtype == np.float64

    def test_add_multiple_videos(self) -> None:
        # given
        split_to_frames = 60
        video_processor = VideoProcessor(60)

        # when
        video_processor.add_video('test_data/487_frames.pkl')
        video_processor.add_video('test_data/366_frames.pkl')
        (x_train, y_train), (x_test, y_test), int_to_klass = video_processor.load_data(0.3)

        # then
        assert x_train.dtype == np.float64
        assert x_train.shape[1] == split_to_frames
        assert x_test.dtype == np.float64
        assert x_test.shape[1] == split_to_frames

    def test_should_return_flatten_points(self) -> None:
        # given
        input_array = np.array([
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
             [1, 1], [1, 1], [1, 1], [1, 1]],
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
             [1, 1], [1, 1], [1, 1], [1, 1]],
        ])

        expected_output_array = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        # when
        output_array = VideoProcessor(2)._flatten_last_dim(input_array)

        # then
        assert np.array_equal(output_array, expected_output_array)
