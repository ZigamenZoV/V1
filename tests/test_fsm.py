"""FSM: K-дебаунс стабильных состояний (спека §5.1)."""
from people_counter.config import FsmConfig
from people_counter.counting.fsm import TrackFSM


def feed(fsm: TrackFSM, states: list[str | None]) -> None:
    for i, z in enumerate(states):
        fsm.update(z, (float(i), 0.0), 0.9)


def stable_states(fsm: TrackFSM) -> list[str | None]:
    return [s.state for s in fsm.stable]


def test_debounce_requires_k_frames():
    fsm = TrackFSM(FsmConfig(k_frames=3))
    feed(fsm, ["A", "A"])
    assert stable_states(fsm) == []
    feed(fsm, ["A"])
    assert stable_states(fsm) == ["A"]


def test_flicker_does_not_register():
    fsm = TrackFSM(FsmConfig(k_frames=3))
    # топтание на пороге: серии короче K не фиксируются
    feed(fsm, ["A", "A", "A", "B", "A", "B", "B", "A", "A", "A"])
    assert stable_states(fsm) == ["A"]


def test_full_pass_sequence():
    fsm = TrackFSM(FsmConfig(k_frames=3))
    feed(fsm, [None] * 3 + ["A"] * 5 + ["B"] * 5 + [None] * 3)
    assert stable_states(fsm) == [None, "A", "B", None]


def test_short_out_gap_between_zones_ignored_by_debounce():
    fsm = TrackFSM(FsmConfig(k_frames=3))
    # разрыв OUT в 2 кадра (< K) между зонами вообще не фиксируется
    feed(fsm, ["A"] * 4 + [None] * 2 + ["B"] * 4)
    assert stable_states(fsm) == ["A", "B"]


def test_mean_score_and_path():
    fsm = TrackFSM(FsmConfig(k_frames=1))
    fsm.update("A", (0.0, 0.0), 0.8)
    fsm.update("A", (3.0, 4.0), 0.6)
    assert abs(fsm.mean_score - 0.7) < 1e-9
    assert abs(fsm.path_px - 5.0) < 1e-9
