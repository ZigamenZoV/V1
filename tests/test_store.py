"""SQLite Event Store: запись/чтение, восстановление occupancy."""
from people_counter.store import EventStore
from people_counter.types import Event


def test_event_roundtrip(tmp_path):
    st = EventStore(tmp_path / "t.db")
    ev = Event.make("gate-1", 42, "enter", trajectory_len=47, confidence=0.93)
    ev.embedding_id = "abc"
    st.add_event(ev)
    rows = st.events(limit=10)
    assert len(rows) == 1
    assert rows[0]["type"] == "enter"
    assert rows[0]["track_id"] == 42
    assert rows[0]["embedding_id"] == "abc"
    assert st.counts_today() == {"enter": 1, "exit": 0}
    st.close()


def test_occupancy_restore(tmp_path):
    st = EventStore(tmp_path / "t.db")
    st.log_occupancy(5.0, "events", 1.0)
    st.log_occupancy(7.0, "snapshot")
    assert st.last_occupancy() == 7.0
    st.close()
    st2 = EventStore(tmp_path / "t.db")          # рестарт сервиса
    assert st2.last_occupancy() == 7.0
    st2.close()


def test_filters(tmp_path):
    st = EventStore(tmp_path / "t.db")
    for cam in ("gate-1", "gate-2"):
        st.add_event(Event.make(cam, 1, "enter", 10, 0.9))
    assert len(st.events(camera_id="gate-1")) == 1
    assert len(st.events(limit=1)) == 1
    st.close()


def test_audit_log(tmp_path):
    st = EventStore(tmp_path / "t.db")
    st.add_audit("hall-1", raw=23.0, applied=1.5, occ_after=21.5)
    hist = st.occupancy_history()
    assert hist == []                            # аудит пишется в свою таблицу
    st.close()
