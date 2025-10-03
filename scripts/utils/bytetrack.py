# utils/bytetrack.py
import numpy as np

class BYTETracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.5, buffer_size=30, frame_rate=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.buffer_size = buffer_size
        self.frame_rate = frame_rate

        self.tracks = []
        self.next_id = 1  # global counter for all objects

    class Track:
        def __init__(self, track_id, bbox, cls_id):
            self.id = track_id
            self.bbox = bbox  # [x1, y1, x2, y2]
            self.cls_id = cls_id
            self.lost_frames = 0

    def update(self, detections, frame):
        """
        detections: list of [x1, y1, x2, y2, conf, cls_id]
        cls_id: 0 = player, 1 = ball, 2 = referee
        """
        updated_tracks = []

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)

            if conf < self.track_thresh:
                continue

            # Try to match with existing tracks (naive IoU matching)
            matched = None
            for trk in self.tracks:
                iou = self._iou(trk.bbox, [x1, y1, x2, y2])
                if trk.cls_id == cls_id and iou > self.match_thresh:
                    matched = trk
                    break

            if matched:
                matched.bbox = [x1, y1, x2, y2]
                matched.lost_frames = 0
                updated_tracks.append(matched)
            else:
                # create new track with new ID
                new_trk = self.Track(self.next_id, [x1, y1, x2, y2], cls_id)
                self.next_id += 1
                updated_tracks.append(new_trk)

        # Only keep currently updated tracks
        self.tracks = updated_tracks

        # return format: [x1, y1, x2, y2, track_id, cls_id]
        return [[*trk.bbox, trk.id, trk.cls_id] for trk in self.tracks]

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea

        return interArea / unionArea if unionArea > 0 else 0
