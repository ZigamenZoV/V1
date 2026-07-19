"""ReID-леджер: матчинг, временной приор, TTL, unmatched-метрика (спека §6)."""
import time

import numpy as np

from people_counter.config import ReidConfig
from people_counter.reid.ledger import Ledger


def unit(i: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, np.float32)
    v[i] = 1.0
    return v


def mix(a: np.ndarray, b: np.ndarray, wa: float) -> np.ndarray:
    v = wa * a + (1 - wa) * b
    return v / np.linalg.norm(v)


def make_ledger(tmp_path, **over) -> Ledger:
    cfg = ReidConfig(dump_path=str(tmp_path / "ledger.npz"), **over)
    return Ledger(cfg)


def test_enter_exit_match(tmp_path):
    led = make_ledger(tmp_path, threshold=0.6)
    eid = led.on_enter("gate-1", unit(0))
    assert led.size == 1
    m = led.on_exit("gate-1", mix(unit(0), unit(1), 0.9))   # почти тот же человек
    assert m.entry_id == eid
    assert led.size == 0
    assert led.unmatched_ratio == 0.0


def test_no_match_below_threshold(tmp_path):
    led = make_ledger(tmp_path, threshold=0.6)
    led.on_enter("gate-1", unit(0))
    m = led.on_exit("gate-1", unit(1))                      # ортогонален
    assert m.entry_id is None
    assert led.size == 1                                    # запись не погашена
    assert led.unmatched_ratio == 1.0


def test_time_prior_prefers_recent(tmp_path):
    led = make_ledger(tmp_path, threshold=0.5, time_bonus=0.2, time_half_life_s=600)
    e_old = led.on_enter("gate-1", unit(0))
    e_new = led.on_enter("gate-1", unit(0))                 # одинаковая внешность
    led._entries[e_old].ts -= 3600                          # старая запись — час назад
    m = led.on_exit("gate-1", unit(0))
    assert m.entry_id == e_new                              # недавний ранжируется выше


def test_ttl_purge(tmp_path):
    led = make_ledger(tmp_path, ttl_s=10)
    led.on_enter("gate-1", unit(0))
    for e in led._entries.values():
        e.ts = time.time() - 100
    led.on_exit("gate-1", unit(1))                          # exit триггерит purge
    assert led.size == 0


def test_dump_restore(tmp_path):
    led = make_ledger(tmp_path)
    led.on_enter("gate-1", unit(2))
    led.dump_now()
    led2 = make_ledger(tmp_path)
    assert led2.size == 1
    m = led2.on_exit("gate-2", unit(2))
    assert m.entry_id is not None


def test_exit_without_embedding_is_unmatched(tmp_path):
    led = make_ledger(tmp_path)
    m = led.on_exit("gate-1", None)
    assert m.entry_id is None
    assert led.unmatched_ratio == 1.0
