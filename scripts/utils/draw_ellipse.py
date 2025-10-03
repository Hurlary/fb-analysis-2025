import cv2
import numpy as np
from .bbox_utils import get_bbox_width, get_center_of_bbox

def draw_ellipse(frame, bbox, track_id, role=None):
    # inside draw_ellipse
    if track_id is None:                       # ball
        colour = (0, 0, 255)              # red ball
    else:                                 # player
        colour = (200, 255, 255)          # light-cyan player
    # single light-cyan for every player / ref
    colour = (200, 255, 255)
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    w = max(int(get_bbox_width(bbox)), 1)
    h = max(int(0.35 * w), 1)

    cv2.ellipse(frame, (x_center, y2), (w, h),
                0.0, -45, 235, colour, 1, cv2.LINE_4)

    # auto-fit label
    label = f"{track_id}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    rw, rh = tw + 12, th + 6
    x1 = x_center - rw // 2
    y1 = y2 + 5
    cv2.rectangle(frame, (x1, y1), (x1 + rw, y1 + rh), colour, -1)
    cv2.putText(frame, label, (x1 + 6, y1 + th + 2),
                font, scale, (0, 0, 0), thick)