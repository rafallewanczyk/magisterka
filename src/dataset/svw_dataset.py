import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.dataset.video import Video
from gluoncv.data import LstDetection, RecordFileDetection


class SVWDataset:
    BBOX_FILE = 'BoundingBoxes.csv'
    VIDEOS_DIR = 'Videos'
    LABELED_FRAMES_DIR = 'labeled_frames'
    LST_FILE = LABELED_FRAMES_DIR + '/frames.lst'
    REC_FILE = LABELED_FRAMES_DIR + '/frames.rec'
    IDX_FILE = LABELED_FRAMES_DIR + '/frames.idx'

    def __init__(self, svw_path: str):
        self.svw_path = Path(svw_path)
        self._bbox_df = pd.read_csv(self.svw_path / self.BBOX_FILE)
        self._videos_paths = self._load_videos_paths()

    def _get_video_bboxes(self, video_name: str):
        return self._bbox_df.where(self._bbox_df['FOLDER & FILE'] == video_name).dropna(subset=['FOLDER & FILE'])

    def _load_videos_paths(self):
        videos_paths = []
        klass_dir = self.svw_path / SVWDataset.VIDEOS_DIR
        for klass_path in klass_dir.iterdir():
            for video_path in klass_path.iterdir():
                videos_paths.append(video_path)
        return videos_paths

    def get_as_video(self, video_path: Path):
        full_video_path = self.svw_path.joinpath(SVWDataset.VIDEOS_DIR, video_path)
        if full_video_path not in self._videos_paths:
            raise ValueError(f"{full_video_path} no such video in database")

        return Video(full_video_path, self._get_video_bboxes(video_path.as_posix()))

    def dump_labeled_frames(self):
        lines = []
        progress = tqdm(self._bbox_df.to_dict(orient='records'))
        for record in progress:
            filename_pattern = record['FOLDER & FILE'].split('.')[0] + '_{idx}.jpg'
            frame_path_pattern = self.svw_path.joinpath(SVWDataset.LABELED_FRAMES_DIR, filename_pattern).as_posix()
            progress.set_postfix({"img": frame_path_pattern})

            try:
                video = self.get_as_video(Path(record['FOLDER & FILE']))
            except ValueError as exc:
                print(exc)
                continue

            lines += video.dump_bboxes_frames(frame_path_pattern)
        enumerated_lines = [line.format(idx=idx) for idx, line in enumerate(lines)]
        with (self.svw_path / SVWDataset.LST_FILE).open('w') as file:
            for line in enumerated_lines:
                file.write(line)
        self.load_lst_dataset()

    def load_lst_dataset(self):
        lst_dataset = LstDetection((self.svw_path / self.LST_FILE).as_posix(),
                                   (self.svw_path / self.LABELED_FRAMES_DIR).as_posix())
        self._print_dataset_stats(lst_dataset)
        return lst_dataset

    def generate_record_file(self):
        subprocess.check_output(
            [sys.executable,
             '/home/rafa/DataspellProjects/magisterka/venv/lib/python3.10/site-packages/mxnet/tools/im2rec.py',
             f'{(self.svw_path / self.LST_FILE).as_posix()}',
             f'{(self.svw_path / self.LABELED_FRAMES_DIR).as_posix()}',
             '--no-shuffle', '--pass-through', '--pack-label'])
        self.load_rec_dataset()

    def load_rec_dataset(self):
        record_dataset = RecordFileDetection((self.svw_path / self.REC_FILE).as_posix(), coord_normalized=True)
        self._print_dataset_stats(record_dataset)
        return record_dataset

    @staticmethod
    def _print_dataset_stats(dataset: Any):
        print('dataset length:', len(dataset))
        first_img = dataset[0][0]
        print('image shape:', first_img.shape)
        print('Label example:', dataset[0][1])


svw_dataset = SVWDataset('/home/rafa/SVW')
lst_dataset = svw_dataset.load_lst_dataset()
rec_dataset = svw_dataset.load_rec_dataset()
# svw_dataset.generate_record_file()
# svw_dataset.dump_labeled_frames()
# video_name = 'baseball/311___5044583402e14d6fb442a78fb2a9fcf5.mp4'
# video = svw_dataset.get_as_video(Path(video_name))
# print(video.dump_bboxes_frames('./file_{idx}.jpg'))

# python im2rec.py /home/rafa/SVW/labeled_frames/frames.lst /home/rafa/SVW/labeled_frames --no-shuffle --pass-through --pack-label
