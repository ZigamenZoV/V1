"""People Counter v2 — счёт вход/выход/occupancy по головам.

Архитектура (см. README): RTSP → NVDEC decode → RF-DETR Nano (GPU, батч)
→ BoT-SORT (CPU) → Zone FSM → финализация траекторий → события enter/exit
→ ReID-леджер → occupancy-интегратор с аудит-коррекцией → FastAPI.
"""

__version__ = "2.0.0"
