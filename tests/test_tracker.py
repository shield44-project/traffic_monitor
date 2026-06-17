from detection.tracker import CentroidTracker


def test_centroid_tracker_registers_and_counts_crossing():
    tracker = CentroidTracker(max_disappeared=2, max_distance=50)
    tracker.set_counting_line(50)

    tracker.update([(10, 10, 30, 30)])
    tracker.update([(10, 40, 30, 60)])
    tracker.update([(10, 70, 30, 90)])

    assert tracker.total_crossings >= 1
    assert len(tracker.objects) == 1


def test_centroid_tracker_returns_stable_assignments():
    tracker = CentroidTracker(max_disappeared=2, max_distance=50)

    _tracks, assignments1 = tracker.update_with_assignments([(10, 10, 30, 30)])
    _tracks, assignments2 = tracker.update_with_assignments([(12, 12, 32, 32)])

    assert assignments1 == assignments2
    assert assignments1[0] == 0
