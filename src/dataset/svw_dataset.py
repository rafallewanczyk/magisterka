from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from pydantic import validate_arguments


class SVWDataset:
    BBOX_FILE = 'BoundingBoxes.csv'
    BBOX_START_FRAME_KEYS = ['START FRAME', 'BOX-start: x', 'BOX-start: y', 'BOX-start: w', 'BOX-start: h']
    BBOX_MID_FRAME_KEYS = ['MID frame', 'BOX-mid: x', 'BOX-mid: y', 'BOX-mid: w', 'BOX-mid: h']
    BBOX_END_FRAME_KEYS = ['END FRAME', 'BOX-end: x', 'BOX-end: y', 'BOX-end: w', 'BOX-end: h']

    @validate_arguments
    @dataclass
    class SVWBoundingBox:
        frame: int
        x: float
        y: float
        w: float
        h: float

        def __post_init__(self):
            self.frame -= 1

        def fit_to_frame(self, frame_w: float, frame_h: float):
            return {"pt1": (int(self.x * frame_w), int(self.y * frame_h)),
                    "pt2": (int(self.x * frame_w) + int(self.w * frame_w), int(self.y * frame_h + self.h * frame_h))}

    def __init__(self, svw_path: str):
        self.svw_path = Path(svw_path)
        self._bbox_df = pd.read_csv(self.svw_path / self.BBOX_FILE)

    def get_video_bboxes(self, video_name: str):
        return self._bbox_df.where(self._bbox_df['FOLDER & FILE'] == video_name).dropna(subset=['FOLDER & FILE'])
