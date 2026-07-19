"""Финализация траекторий (спека §5.2): классификация начал/концов и полной
последовательности; фильтры шума; фрагментация без двойного счёта."""
import numpy as np

from people_counter.counting import TrajectoryCounter, ZonePair
from people_counter.types import Track


def walk(counter: TrajectoryCounter, tid: int, xs: list[float], y: float = 280.0) -> None:
    """Прогоняет трек по горизонтальной траектории (кадр за кадром)."""
    for x in xs:
        box = np.array([x - 9, y - 9, x + 9, y + 9], np.float32)
        counter.update([Track(tid, box, 0.9)], None)


def path(x0: float, x1: float, step: float = 8.0) -> list[float]:
    n = max(2, int(abs(x1 - x0) / step))
    return list(np.linspace(x0, x1, n))


def make_counter(gate_cam) -> TrajectoryCounter:
    return TrajectoryCounter(gate_cam, ZonePair(gate_cam.zones), fps=10, reid=None)


def test_enter(gate_cam):
    c = make_counter(gate_cam)
    walk(c, 1, path(70, 540))                 # A → B → почти OUT справа
    evs = c.finalize([1])
    assert [e.type for e in evs] == ["enter"]
    assert evs[0].camera_id == "gate-test"
    assert 0 < evs[0].confidence <= 1


def test_exit(gate_cam):
    c = make_counter(gate_cam)
    walk(c, 2, path(540, 70))
    assert [e.type for e in c.finalize([2])] == ["exit"]


def test_passby_no_event(gate_cam):
    c = make_counter(gate_cam)
    # вертикальный проход только по зоне A
    for y in np.linspace(70, 490, 40):
        c.update([Track(3, np.array([150 - 9, y - 9, 150 + 9, y + 9], np.float32), 0.9)], None)
    assert c.finalize([3]) == []


def test_peek_no_event(gate_cam):
    c = make_counter(gate_cam)
    # заглянул: A → чуть-чуть в B → назад в A → ушёл; конец на стороне A
    xs = path(70, 300) + path(300, 70)
    walk(c, 4, xs)
    assert c.finalize([4]) == []


def test_deep_uturn_no_event(gate_cam):
    c = make_counter(gate_cam)
    # глубокий заход в B и возврат: первая зона A, последняя A → события нет
    xs = path(70, 450) + path(450, 70)
    walk(c, 5, xs)
    assert c.finalize([5]) == []


def test_short_track_filtered(gate_cam):
    c = make_counter(gate_cam)
    walk(c, 6, [270, 275, 280, 285, 290])     # 5 кадров < min_track_frames
    assert c.finalize([6]) == []


def test_static_track_filtered(gate_cam):
    c = make_counter(gate_cam)
    walk(c, 7, [150.0 + (i % 2) * 0.5 for i in range(30)])   # дрожит на месте
    assert c.finalize([7]) == []


def test_fragment_born_inside_no_double_count(gate_cam):
    """Обломок трека человека, уже прошедшего в B, не порождает событие."""
    c = make_counter(gate_cam)
    walk(c, 8, path(70, 400))                 # полный вход
    walk(c, 9, path(400, 540))                # фрагмент: родился в B, ушёл вглубь
    evs = c.finalize([8, 9])
    assert [e.type for e in evs] == ["enter"]


def test_two_people_in_a_row_both_counted(gate_cam):
    """Плотный поток: два человека друг за другом — оба события засчитаны."""
    c = make_counter(gate_cam)
    xs1 = path(70, 540)
    xs2 = path(70, 540)
    lag = 6
    for i in range(len(xs1) + lag):
        tracks = []
        if i < len(xs1):
            x = xs1[i]
            tracks.append(Track(10, np.array([x - 9, 271, x + 9, 289], np.float32), 0.9))
        if i >= lag and i - lag < len(xs2):
            x = xs2[i - lag]
            tracks.append(Track(11, np.array([x - 9, 331, x + 9, 349], np.float32), 0.9))
        c.update(tracks, None)
    evs = c.finalize([10, 11])
    assert [e.type for e in evs] == ["enter", "enter"]
