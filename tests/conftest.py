from __future__ import annotations

import pytest

from people_counter.config import (CameraConfig, FsmConfig, SyntheticConfig,
                                   ZonesConfig)

# геометрия согласована с config/mock.yaml и capture/synthetic.py:
# кадр 560×560, дверь x=280, A — левая (уличная), B — правая (внутренняя)
ZONE_A = [(60, 60), (275, 60), (275, 500), (60, 500)]
ZONE_B = [(285, 60), (500, 60), (500, 500), (285, 500)]


@pytest.fixture
def gate_cam() -> CameraConfig:
    return CameraConfig(
        id="gate-test",
        role="gate",
        source="synthetic",
        fps=10,
        zones=ZonesConfig(a=ZONE_A, b=ZONE_B),
        fsm=FsmConfig(k_frames=3, min_track_frames=8, min_path_px=40),
        synthetic=SyntheticConfig(seed=7, speed_px=4.0),
    )
