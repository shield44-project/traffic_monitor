"""
Lightweight centroid tracker for vehicle counting.

A full DeepSORT/ByteTrack is overkill (and heavy) for a laptop demo, so this
implements the classic *centroid tracking* algorithm: match each new detection
to the nearest existing track within a distance threshold, register new tracks
for unmatched detections and retire tracks that disappear for N frames.

It also supports a horizontal **counting line**: every time a track's centroid
crosses the line, a directional counter is incremented. This gives a cumulative
"vehicles passed" figure on top of the instantaneous per-frame count.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 80.0):
        self.next_id = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()  # id -> centroid
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # Counting-line state.
        self.line_y: int | None = None
        self.count_up = 0      # crossed line moving up (toward smaller y)
        self.count_down = 0    # crossed line moving down (toward larger y)
        self._last_y: dict[int, float] = {}

    def set_counting_line(self, y: int) -> None:
        self.line_y = int(y)

    @property
    def total_crossings(self) -> int:
        return self.count_up + self.count_down

    def _register(self, centroid: np.ndarray) -> None:
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self._last_y[self.next_id] = float(centroid[1])
        self.next_id += 1

    def _deregister(self, oid: int) -> None:
        self.objects.pop(oid, None)
        self.disappeared.pop(oid, None)
        self._last_y.pop(oid, None)

    def _check_crossing(self, oid: int, centroid: np.ndarray) -> None:
        if self.line_y is None:
            return
        prev_y = self._last_y.get(oid)
        cur_y = float(centroid[1])
        if prev_y is not None:
            if prev_y < self.line_y <= cur_y:
                self.count_down += 1
            elif prev_y > self.line_y >= cur_y:
                self.count_up += 1
        self._last_y[oid] = cur_y

    def update(self, boxes: list[tuple[float, float, float, float]]
               ) -> dict[int, tuple[int, int]]:
        """Update tracks with current-frame boxes (x1,y1,x2,y2).

        Returns a mapping of track-id -> (cx, cy) centroid for drawing.
        """
        if len(boxes) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return {oid: tuple(c.astype(int)) for oid, c in self.objects.items()}

        centroids = np.array(
            [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in boxes])

        if len(self.objects) == 0:
            for c in centroids:
                self._register(c)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # Pairwise distances between existing tracks and new detections.
            dists = np.linalg.norm(
                object_centroids[:, None] - centroids[None, :], axis=2)

            # Greedy match: smallest distances first.
            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if dists[row, col] > self.max_distance:
                    continue
                oid = object_ids[row]
                self.objects[oid] = centroids[col]
                self.disappeared[oid] = 0
                self._check_crossing(oid, centroids[col])
                used_rows.add(row)
                used_cols.add(col)

            # Unmatched existing tracks -> disappeared.
            for row in set(range(dists.shape[0])) - used_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

            # Unmatched detections -> new tracks.
            for col in set(range(dists.shape[1])) - used_cols:
                self._register(centroids[col])

        return {oid: tuple(c.astype(int)) for oid, c in self.objects.items()}
