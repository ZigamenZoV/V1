# People Counter v2

Подсчёт посетителей по камерам: **вход / выход / «сколько людей сейчас
внутри»** (occupancy) в реальном времени. Детекция **голов** (в плотной толпе
они, в отличие от тел, почти не перекрываются), трекинг траекторий через
пару зон у двери и события только по полностью пройденному пути — поэтому
«заглянул и ушёл», проход мимо двери и топтание на пороге счёт не портят.

Рассчитан на слабое железо: одна NVIDIA GTX 1650 (4 GB) тянет 2–4 камеры.
Работает как Windows-служба 24/7.

## Возможности

- **Счёт вход/выход** по RTSP-камерам над входными группами (RF-DETR Nano,
  TensorRT FP16 / ONNX Runtime + BoT-SORT/ByteTrack).
- **Occupancy** — интеграл входов/выходов с автокоррекцией: по обзорным
  камерам периодически пересчитывается вся толпа (LWCC/P2PNet), дрейф
  счётчика втягивается к замеру, сильное расхождение даёт алерт.
- **ReID-леджер**: «кто внутри» по эмбеддингам тел (OSNet) — матчит выходы
  со входами; доля несматченных выходов сигнализирует о деградации системы.
- **Live-дашборд** — occupancy, снапшоты камер с зонами и детекциями,
  лента событий (SSE), без внешних CDN.
- **REST API + Prometheus-метрики** для интеграций и мониторинга.
- **Mock-режим**: весь конвейер на синтетической сцене — можно попробовать
  без камер, GPU и обученных весов.

## Быстрый старт

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m pytest                    # 37 тестов
.venv\Scripts\python -m people_counter simulate   # демо на синтетике
```

Открыть **http://127.0.0.1:8000** — по нарисованной «двери» ходят
синтетические люди, счётчики живут. Остановка: `Ctrl+C`.

## Команды

| Команда | Что делает |
|---|---|
| `python -m people_counter run` | запуск сервиса (`config/app.yaml`) |
| `python -m people_counter simulate` | демо на синтетической сцене |
| `python -m people_counter check` | диагностика: ffmpeg, GPU, рантаймы, модели, RTSP, БД |
| `python -m people_counter bench-detector` | замер FPS детектора на этой машине |

## API

| Эндпоинт | Ответ |
|---|---|
| `GET /` | дашборд |
| `GET /occupancy` | текущее значение + счётчики за день + состояние аудита |
| `GET /events?limit&since&camera` | события enter/exit из SQLite |
| `GET /health` | статус компонентов: FPS декода/трекинга, треки, леджер |
| `GET /cameras`, `GET /cameras/{id}/snapshot.jpg` | камеры и живые снапшоты |
| `GET /stream` | SSE: события, occupancy, статистика |
| `GET /metrics` | Prometheus |
| `GET /docs` | Swagger |

## Подключение реальных камер

Кратко (подробный порядок ввода в строй — в
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

1. Установить [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) в PATH; поднять
   [go2rtc](https://github.com/AlexxIT/go2rtc) как RTSP-прослойку
   (`deploy/go2rtc.yaml`).
2. Получить веса детектора: `scripts/export_onnx.py` (стартово COCO-веса,
   затем файнтюн на головах — `scripts/prepare_crowdhuman.py` +
   `scripts/train_heads.py`); на целевой машине собрать TensorRT-engine:
   `scripts/build_engine.py`.
3. Описать камеры в `config/cameras/*.yaml` (пример — `gate-1.yaml`),
   разметить зоны A/B по живому снапшоту: `scripts/calibrate_zones.py`.
4. Проверить готовность `python -m people_counter check` и запустить `run`.
5. Установить службой Windows: `deploy/install_service.ps1` (NSSM).

## Конфигурация

`config/app.yaml` — сервис, детектор, трекер, ReID, аудит;
`config/cameras/*.yaml` — по файлу на камеру (URL, зоны, пороги FSM).
Главные ручки:

| Параметр | Смысл |
|---|---|
| `detector.backend` | `tensorrt` (основной) / `onnxruntime` (fallback) / `mock` |
| `detector.input_size` | 560; поднять до 616/672, если головы в кадре < ~12 px |
| `tracker.backend` | `botsort` / `bytetrack` / `simple` (без torch) |
| `fsm.k_frames` | сколько кадров подряд подтверждают зону (антидребезг) |
| `audit.enabled` | автокоррекция occupancy по обзорным камерам |
| `pipeline.mode` | `multiprocess` (production) / `single` (отладка) |

## Как это работает

Однострочная версия: NVDEC-декод → батч-инференс детектора голов на GPU →
CPU-трекинг → FSM по паре зон с антидребезгом → классификация полной
траектории на событие → SQLite/леджер/occupancy → FastAPI.

Схема конвейера, процессная модель, бюджет VRAM, порядок ввода в строй и
таблица рисков — в **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Лицензия

MIT. Модель RF-DETR — Apache 2.0.
