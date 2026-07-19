"""Occupancy-интегратор и аудит-коррекция (спека §7)."""
from people_counter.config import AuditConfig
from people_counter.occupancy import OccupancyIntegrator


def make(occ: float = 0.0, **over) -> OccupancyIntegrator:
    params = {"enabled": True, "alpha": 0.25, "ema": 0.5, "outlier_abs": 15,
              "alert_delta": 10, **over}
    return OccupancyIntegrator(AuditConfig(**params), occ)


def test_events_integrate():
    o = make()
    o.apply_event("enter")
    o.apply_event("enter")
    o.apply_event("exit")
    assert o.snapshot()["occupancy"] == 1
    assert o.snapshot()["enters_total"] == 2


def test_never_negative():
    o = make()
    o.apply_event("exit")
    assert o.value == 0.0


def test_audit_pulls_towards_measurement():
    o = make(occ=10.0)
    applied, value, alert = o.apply_audit(14.0)     # в пределах outlier_abs
    # ema=14 → occupancy += 0.25*(14-10) = 1
    assert abs(applied - 1.0) < 1e-9
    assert abs(value - 11.0) < 1e-9
    assert not alert


def test_outlier_needs_confirmation():
    o = make(occ=0.0)
    applied, value, _ = o.apply_audit(30.0)         # выброс: |30-0| > 15
    assert applied == 0.0 and value == 0.0          # одиночный замер не применён
    applied, value, alert = o.apply_audit(29.0)     # подтверждение
    assert applied > 0.0
    assert alert                                    # рассинхрон выше alert_delta


def test_audit_ema_smooths():
    o = make(occ=0.0, outlier_abs=1000, alert_delta=1000)
    o.apply_audit(10.0)                             # ema=10, occ=2.5
    o.apply_audit(2.0)                              # ema=6 → occ += 0.25*(6-2.5)
    snap = o.snapshot()
    assert snap["audit_ema"] == 6.0
    assert abs(snap["occupancy_raw"] - 3.38) < 0.01
