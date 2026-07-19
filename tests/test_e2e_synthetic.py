"""E2E ядра конвейера: синтетика → mock-детектор → трекер → FSM → события.

Детерминированные сценарии агентов; проверяем точный счёт enter/exit и
отсутствие ложных событий от passby/peek (спека §5).
"""
import numpy as np

from people_counter.capture.synthetic import SyntheticSource
from people_counter.config import DetectorConfig, TrackerConfig
from people_counter.counting import TrajectoryCounter, ZonePair
from people_counter.detect.mock import MockDetector
from people_counter.track.simple import SimpleTracker


def run_sim(gate_cam, script: dict[int, list[str]], frames: int) -> dict[str, int]:
    """script: номер кадра → список сценариев для спауна."""
    gate_cam.synthetic.enters_per_min = 0     # только ручной спаун
    gate_cam.synthetic.exits_per_min = 0
    gate_cam.synthetic.passby_per_min = 0
    gate_cam.synthetic.peek_per_min = 0

    src = SyntheticSource(gate_cam, 560, 560, realtime=False)
    det = MockDetector(DetectorConfig(backend="mock", mock={
        "jitter_px": 0.8, "dropout": 0.04, "false_pos_per_frame": 0.02}))
    trk = SimpleTracker(TrackerConfig(backend="simple", track_buffer_s=2.0), fps=10)
    counter = TrajectoryCounter(gate_cam, ZonePair(gate_cam.zones), fps=10, reid=None)

    counts = {"enter": 0, "exit": 0}
    for i in range(frames):
        for kind in script.get(i, []):
            src.spawn(kind)
        frame = src.render()
        detections = det.infer([frame])[0]
        tracks, removed = trk.update(detections, frame)
        counter.update(tracks, frame)
        for ev in counter.finalize(removed):
            counts[ev.type] += 1
    return counts


def test_enters_and_exits_counted_exactly(gate_cam):
    script = {
        5: ["enter"],
        60: ["exit"],
        120: ["enter", "enter"],       # двое одновременно (разные y)
        200: ["exit"],
        260: ["enter"],
    }
    counts = run_sim(gate_cam, script, frames=520)
    assert counts == {"enter": 4, "exit": 2}


def test_passby_and_peek_produce_nothing(gate_cam):
    script = {5: ["passby"], 50: ["peek"], 100: ["passby"], 150: ["peek"]}
    counts = run_sim(gate_cam, script, frames=400)
    assert counts == {"enter": 0, "exit": 0}


def test_mixed_traffic(gate_cam):
    script = {
        5: ["enter", "passby"],
        70: ["exit", "peek"],
        140: ["enter"],
        210: ["passby", "exit"],
    }
    counts = run_sim(gate_cam, script, frames=520)
    assert counts == {"enter": 2, "exit": 2}


def test_gt_counters_track_spawns(gate_cam):
    gate_cam.synthetic.enters_per_min = 0
    gate_cam.synthetic.exits_per_min = 0
    gate_cam.synthetic.passby_per_min = 0
    gate_cam.synthetic.peek_per_min = 0
    src = SyntheticSource(gate_cam, 560, 560, realtime=False)
    src.spawn("enter")
    src.spawn("exit")
    src.spawn("passby")
    assert src.gt_enters == 1
    assert src.gt_exits == 1
