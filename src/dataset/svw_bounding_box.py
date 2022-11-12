from dataclasses import dataclass


@dataclass
class SVWBoundingBox:
    BBOX_START_FRAME_KEYS = ['START FRAME', 'BOX-start: x', 'BOX-start: y', 'BOX-start: w', 'BOX-start: h']
    BBOX_MID_FRAME_KEYS = ['MID frame', 'BOX-mid: x', 'BOX-mid: y', 'BOX-mid: w', 'BOX-mid: h']
    BBOX_END_FRAME_KEYS = ['END FRAME', 'BOX-end: x', 'BOX-end: y', 'BOX-end: w', 'BOX-end: h']

    frame: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @classmethod
    def from_w_h(cls, frame: int, x_min: float, y_min: float, w: float, h: float):
        return cls(frame, x_min, y_min, x_min + w, y_min + h)

    def __post_init__(self):
        self.frame = int(self.frame)
        self.x_min = float(self.x_min)
        self.y_min = float(self.y_min)
        self.x_max = float(self.x_max)
        self.y_max = float(self.y_max)

    def fit_to_frame(self, frame_w: float, frame_h: float) -> 'SVWBoundingBox':
        return SVWBoundingBox(self.frame, int(self.x_min * frame_w), int(self.y_min * frame_h),
                              int(self.x_max * frame_w), int(self.y_max * frame_h))

    def get_as_cv_rec(self):
        return {"pt1": (int(self.x_min), int(self.y_min)),
                "pt2": (int(self.x_max), int(self.y_max))}
