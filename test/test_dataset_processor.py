import numpy as np

from dataset_processor import VideoProcessor


class TestDatasetProcessor:

    def test_split_pose_should_return_2_sequences(self) -> None:
        # given
        example_pose = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        frame_num = 3

        # when
        out_poses = VideoProcessor.split_pose_to_frame_seq(pose=example_pose, frames_num=frame_num)

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
        example_pose = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        frame_num = 5

        # when
        out_poses = VideoProcessor.split_pose_to_frame_seq(pose=example_pose, frames_num=frame_num)

        # then
        assert np.array_equal(out_poses, np.array([example_pose]))
