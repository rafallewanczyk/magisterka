from pathlib import Path
from time import sleep
from typing import Optional, List

import cv2
import numpy as np
import pandas as pd
import skvideo.io

from src.dataset.svw_bounding_box import SVWBoundingBox


class Video:

    def __init__(self, path: Path, svw_bounding_boxes: Optional[pd.DataFrame] = None):
        self.path = Path(path)
        self.video_data = skvideo.io.vread(path.as_posix())
        self.w, self.h = self.video_data.shape[2], self.video_data.shape[1]
        self.svw_bounding_boxes = self._generate_svw_bboxes(svw_bounding_boxes)

    @staticmethod
    def _generate_svw_bboxes(svw_bounding_boxes: pd.DataFrame) -> List[SVWBoundingBox]:
        as_list = svw_bounding_boxes.to_dict(orient='records')

        bboxes = []
        for record in as_list:
            for key_type in [SVWBoundingBox.BBOX_START_FRAME_KEYS, SVWBoundingBox.BBOX_MID_FRAME_KEYS,
                             SVWBoundingBox.BBOX_END_FRAME_KEYS]:
                if all([not np.isnan(record[key]) for key in key_type]):
                    bboxes.append(SVWBoundingBox.from_w_h(*[record[key] for key in key_type]))

        return bboxes

    def preview(self):
        for frame in self.video_data:
            as_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('frame', as_rgb)
            if cv2.waitKey(1) == ord('q'):
                break
            sleep(0.03)
        cv2.destroyAllWindows()

    def preview_svw_bboxes(self):
        for bbox, frame in zip(*self.get_svw_bboxes_frames()):
            cv2.rectangle(img=frame, color=(255, 0, 0), thickness=2, **bbox.get_as_cv_rec())
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            sleep(0.3)
        cv2.destroyAllWindows()

    def get_svw_bboxes_frames(self):
        bbox_labels, frames = [], []
        for bbox in self.svw_bounding_boxes:
            imarr_plot = np.copy(self.video_data[bbox.frame])
            as_rgb = cv2.cvtColor(imarr_plot, cv2.COLOR_BGR2RGB)
            bbox_labels.append(bbox.fit_to_frame(self.w, self.h))
            frames.append(as_rgb)
        return bbox_labels, frames

    def dump_bboxes_frames(self, filename_pattern: str):
        lines = []
        Path(filename_pattern).parent.mkdir(parents=True, exist_ok=True)
        for idx, (bbox, frame) in enumerate(zip(*self.get_svw_bboxes_frames())):
            image_path = Path(filename_pattern.format(idx=idx))
            cv2.imwrite(image_path.as_posix(), frame)
            lines.append(Video.generate_lst_line(image_path, bbox, frame, "{idx}"))
        return lines

    @staticmethod
    def generate_lst_line(frame_path: Path, bbox: SVWBoundingBox, frame: np.array, idx: str):
        # only one bbox per class

        h, w, c = frame.shape
        header_length = 4
        length_of_label = 5
        str_idx = [idx]
        str_header = [str(x) for x in [header_length, length_of_label, w, h]]

        # 0 for class person
        str_labels = ['0', str(bbox.x_min/w), str(bbox.y_min/h), str(bbox.x_max/w), str(bbox.y_max/h)]
        str_path = [frame_path.as_posix()]

        line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
        return line
