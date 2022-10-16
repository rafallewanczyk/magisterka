from pathlib import Path
from time import sleep
from typing import Optional, List

import cv2
import numpy as np
import pandas as pd
import skvideo.io

from src.dataset.svw_dataset import SVWDataset


class Video:

    def __init__(self, path: Path, svw_bounding_boxes: Optional[pd.DataFrame] = None):
        self.path = Path(path)
        self.video_data = skvideo.io.vread(path.as_posix())
        self.w, self.h = self.video_data.shape[2], self.video_data.shape[1]
        self.svw_bounding_boxes = self._generate_svw_bboxes(svw_bounding_boxes)

    @staticmethod
    def _generate_svw_bboxes(svw_bounding_boxes: pd.DataFrame) -> List[SVWDataset.SVWBoundingBox]:
        as_list = svw_bounding_boxes.to_dict(orient='records')

        bboxes = []
        for record in as_list:
            for key_type in [SVWDataset.BBOX_START_FRAME_KEYS, SVWDataset.BBOX_MID_FRAME_KEYS,
                             SVWDataset.BBOX_END_FRAME_KEYS]:
                if all([not np.isnan(record[key]) for key in key_type]):
                    bboxes.append(SVWDataset.SVWBoundingBox(*[record[key] for key in key_type]))

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
        for bbox in self.svw_bounding_boxes:
            imarr_plot = np.copy(self.video_data[bbox.frame])
            cv2.rectangle(img=imarr_plot, color=(255, 0, 0), thickness=2, **bbox.fit_to_frame(self.w, self.h))
            as_rgb = cv2.cvtColor(imarr_plot, cv2.COLOR_BGR2RGB)
            cv2.imshow('frame', as_rgb)
            if cv2.waitKey(1) == ord('q'):
                break
            sleep(0.3)
        cv2.destroyAllWindows()


svw_dataset = SVWDataset('/home/rafa/SVW')
video_name = 'swimming/321___87c6e054ec324b729072d636b8708bac.mp4'

vid = Video(svw_dataset.svw_path.joinpath('Videos', video_name),
            svw_bounding_boxes=svw_dataset.get_video_bboxes(video_name))

vid.preview()
vid.preview_svw_bboxes()
