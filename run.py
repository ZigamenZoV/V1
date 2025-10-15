import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
from pathlib import Path

# Попытка импорта селектора источника
try:
    from video_source_selector import select_video_source

    HAS_SELECTOR = True
except ImportError:
    HAS_SELECTOR = False

from people_counter import (
    YOLODetector, DeepSORTTracker, LineCounter,
    Visualizer, Logger, Config, FrameStats
)


class PeopleCounterSystem:
    def __init__(self):
        self.config = Config()
        self.config.validate()

        # Проверяем сохраненные настройки видео
        self.video_source = self.load_video_settings()

        # Инициализация компонентов
        print("Инициализация компонентов системы...")
        self.detector = YOLODetector(self.config)
        self.tracker = DeepSORTTracker(self.config)
        self.counter = LineCounter(self.config)
        self.visualizer = Visualizer(self.config)
        self.logger = Logger(self.config)

        self.cap = None
        self.frame_count = 0
        self.mirror_mode = self.config.MIRROR_IMAGE
        self.running = True
        self.last_fps_time = time.time()
        self.fps_counter = 0

    def load_video_settings(self):
        """Загрузка сохраненных настроек видео"""
        settings_file = Path("video_settings.json")

        if settings_file.exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                print(f"📁 Загружены настройки: {settings['video_source']}")
                return settings['video_source']
            except Exception as e:
                print(f"⚠️  Ошибка загрузки настроек: {e}")

        # Если настроек нет, используем графический выбор
        if HAS_SELECTOR:
            print("🎯 Запуск выбора источника видео...")
            return select_video_source()
        else:
            print("⚠️  Селектор не доступен, используется камера по умолчанию")
            return 0

    def initialize_video(self):
        """Инициализация видеопотока с выбранным источником"""
        print(f"🎥 Инициализация видеопотока: {self.video_source}")

        # Устанавливаем выбранный источник в конфиг
        self.config.VIDEO_SOURCE = self.video_source

        self.cap = cv2.VideoCapture(self.video_source)

        if not self.cap.isOpened():
            print(f"❌ Не удалось открыть видеопоток: {self.video_source}")

            # Пробуем fallback на камеру по умолчанию
            print("🔄 Пробуем камеру по умолчанию...")
            self.cap = cv2.VideoCapture(0)
            self.config.VIDEO_SOURCE = 0

            if not self.cap.isOpened():
                raise ValueError("Не удалось открыть ни один видеопоток!")

        # Настройка параметров видео
        self.cap.set(cv2.CAP_PROP_FPS, self.config.TARGET_FPS)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"✅ Видеопоток открыт: {width}x{height} @ {fps:.1f} FPS")
        print(f"📹 Источник: {self.video_source}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Полная обработка кадра: детекция -> трекинг -> подсчет -> визуализация
        """
        start_time = time.time()

        # Проверка валидности входного кадра
        if frame is None or frame.size == 0:
            print("⚠️ Получен пустой кадр")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Применение зеркального отражения если включено
        if self.mirror_mode:
            frame = cv2.flip(frame, 1)

        # 1. Детекция объектов
        detections, detection_time = self.detector.detect(frame)

        # 2. Трекинг объектов
        tracked_detections, track_infos = self.tracker.update(detections, frame)

        # 3. Подсчет пересечений линии
        count_stats = self.counter.update(tracked_detections, {})

        # 4. Визуализация результатов
        processed_frame = frame.copy()

        # Отрисовка детекций и треков
        processed_frame = self.visualizer.draw_detections(processed_frame, tracked_detections)

        # Отрисовка траекторий движения
        processed_frame = self.visualizer.draw_trajectories(processed_frame, track_infos)

        # Отрисовка линии подсчета
        processed_frame = self.visualizer.draw_line(processed_frame)

        # Расчет статистики кадра
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        frame_stats = FrameStats(
            frame_number=self.frame_count,
            processing_time=processing_time,
            detections_count=len(detections),
            tracks_count=len(track_infos),
            fps=fps
        )

        # Отрисовка информационной панели
        processed_frame = self.visualizer.draw_info_panel(processed_frame, frame_stats, count_stats)

        # Отрисовка подсказок по управлению
        processed_frame = self.visualizer.draw_controls_info(processed_frame)

        # Логирование статистики
        self.logger.log_frame(frame_stats, count_stats)

        # Обновление FPS для визуализатора
        self.visualizer.update_fps(processing_time)

        return processed_frame

    def run(self):
        """Главный цикл работы системы"""
        try:
            self.initialize_video()
            print("\n" + "=" * 50)
            print("🚀 Система подсчета людей запущена!")
            print("=" * 50)
            print("Управление:")
            print("  Q - Выход")
            print("  R - Сброс счетчиков")
            print("  M - Переключение зеркального режима")
            print("  WASD - Перемещение линии подсчета")
            print("  L - Экспорт статистики")
            print("  C - Показать текущую статистику")
            print("  S - Изменить источник видео (перезапуск)")
            print("-" * 50)

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Не удалось получить кадр")
                    break

                processed_frame = self.process_frame(frame)

                # Проверка что обработанный кадр валидный
                if processed_frame is not None and processed_frame.size > 0:
                    cv2.imshow('People Counter System', processed_frame)
                else:
                    print("⚠️ Пропуск отображения невалидного кадра")

                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self.handle_keyboard(key)

                self.frame_count += 1

        except KeyboardInterrupt:
            print("\n⏹️  Программа прервана пользователем")
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def handle_keyboard(self, key: int):
        """Обработка нажатий клавиш с добавлением смены источника"""
        if key == ord('q'):
            self.running = False
            print("⏹️  Завершение работы...")
        elif key == ord('r'):
            self.counter.reset()
            print("🔄 Счетчики сброшены")
        elif key == ord('m'):
            self.mirror_mode = not self.mirror_mode
            print(f"🔄 Зеркальный режим: {'включен' if self.mirror_mode else 'выключен'}")
        elif key == ord('s'):  # Новая клавиша - смена источника
            print("🔄 Запрос на смену источника видео...")
            self.running = False
        elif key == ord('w'):
            self.counter.move_line(0, -10)
            print("⬆️ Линия перемещена вверх")
        elif key == ord('s'):
            self.counter.move_line(0, 10)
            print("⬇️ Линия перемещена вниз")
        elif key == ord('a'):
            self.counter.move_line(-10, 0)
            print("⬅️ Линия перемещена влево")
        elif key == ord('d'):
            self.counter.move_line(10, 0)
            print("➡️ Линия перемещена вправо")
        elif key == ord('l'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"statistics_{timestamp}.json"
            self.logger.export_summary(export_path)
            print(f"📊 Статистика экспортирована в {export_path}")
        elif key == ord('c'):
            stats = self.counter.stats
            print(f"📊 Текущая статистика: Вошло {stats['in']}, Вышло {stats['out']}, Внутри {stats['inside']}")

    def cleanup(self):
        """Освобождение ресурсов"""
        print("🧹 Освобождение ресурсов...")
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Ресурсы освобождены")


def main():
    """Точка входа с поддержкой повторного запуска при смене источника"""
    first_run = True

    while True:
        try:
            if first_run:
                print("🎯 People Counter System v2.0")
                print("📹 Теперь с выбором источника видео!")
                first_run = False

            system = PeopleCounterSystem()
            system.run()

            # Если пользователь нажал 'S' для смены источника
            if hasattr(system, 'running') and not system.running:
                print("\n" + "=" * 50)
                print("🔄 Перезапуск с новым источником видео...")
                print("=" * 50)
                continue
            else:
                break

        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка запуска системы: {e}")
            print("Проверьте:")
            print("1. Наличие камеры/файла")
            print("2. Установленные зависимости")
            print("3. Доступность выбранного источника")
            break


if __name__ == "__main__":
    main()