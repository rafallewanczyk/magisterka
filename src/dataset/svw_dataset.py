from pathlib import Path

import pandas as pd

from src.dataset.video import Video


class SVWDataset:
    BBOX_FILE = 'BoundingBoxes.csv'

    def __init__(self, svw_path: str):
        self.svw_path = Path(svw_path)
        self._bbox_df = pd.read_csv(self.svw_path / self.BBOX_FILE)
        self._videos_paths = self._load_videos_paths()

    def _get_video_bboxes(self, video_name: str):
        return self._bbox_df.where(self._bbox_df['FOLDER & FILE'] == video_name).dropna(subset=['FOLDER & FILE'])

    def _load_videos_paths(self):
        videos_paths = []
        klass_dir = self.svw_path.joinpath('Videos')
        for klass_path in klass_dir.iterdir():
            for video_path in klass_path.iterdir():
                videos_paths.append(video_path)
        return videos_paths

    def get_as_video(self, video_path: Path):
        full_video_path = self.svw_path.joinpath('Videos', video_path)
        if full_video_path not in self._videos_paths:
            raise ValueError("no such video in database")

        return Video(full_video_path, self._get_video_bboxes(video_path.as_posix()))




svw_dataset = SVWDataset('/home/rafa/SVW')
video_name = 'archery/25888___c633c455c7be4afe84e6bb5e12ff7add.mp4'
video = svw_dataset.get_as_video(Path(video_name))
video.preview_svw_bboxes()