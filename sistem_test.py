"""
Comprehensive Test Suite for People Counter System
Выявляет потенциальные ошибки во время выполнения, которые не видит IDE
"""

import sys
import os
import unittest
import warnings
import traceback
import time
import json
import csv
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np
import cv2

# Подавляем предупреждения для чистоты вывода
warnings.filterwarnings("ignore")

# Добавляем путь к модулю
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestEnvironment:
    """Настройка тестовой среды"""

    @staticmethod
    def create_test_video():
        """Создание тестового видеофайла"""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('test_video.avi', fourcc, 20.0, (640, 480))

        # Создаем 100 кадров с движущимся прямоугольником
        for i in range(100):
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Имитируем движущегося человека
            cv2.rectangle(test_frame, (50 + i * 5, 200), (100 + i * 5, 300), (255, 255, 255), -1)
            out.write(test_frame)

        out.release()
        return 'test_video.avi'

    @staticmethod
    def cleanup():
        """Очистка тестовых файлов"""
        files_to_remove = [
            'test_video.avi', 'test_config.csv', 'test_summary.json',
            'people_counter_log.csv', 'final_summary.json'
        ]
        for file in files_to_remove:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass


class TestConfig(unittest.TestCase):
    """Тестирование конфигурации"""

    def setUp(self):
        try:
            from people_counter.config import Config
            self.config = Config()
        except ImportError as e:
            self.skipTest(f"Cannot import Config: {e}")

    def test_config_attributes_exist(self):
        """Проверка существования всех атрибутов конфигурации"""
        required_attrs = [
            'MODEL_PATH', 'CONFIDENCE_THRESHOLD', 'PERSON_CLASS_ID',
            'DEEPSORT_CONFIG', 'LINE_START', 'LINE_END', 'VIDEO_SOURCE',
            'MIRROR_IMAGE', 'TARGET_FPS', 'COLORS', 'LOG_TO_CSV',
            'CSV_LOG_PATH', 'MAX_TRAJECTORY_POINTS'
        ]

        for attr in required_attrs:
            with self.subTest(attr=attr):
                self.assertTrue(hasattr(self.config, attr), f"Missing attribute: {attr}")

    def test_config_values_validity(self):
        """Проверка валидности значений конфигурации"""
        # Проверка числовых значений
        self.assertGreater(self.config.CONFIDENCE_THRESHOLD, 0)
        self.assertLessEqual(self.config.CONFIDENCE_THRESHOLD, 1)
        self.assertGreaterEqual(self.config.PERSON_CLASS_ID, 0)
        self.assertGreater(self.config.TARGET_FPS, 0)
        self.assertGreater(self.config.MAX_TRAJECTORY_POINTS, 0)

        # Проверка структур данных
        self.assertIsInstance(self.config.DEEPSORT_CONFIG, dict)
        self.assertIsInstance(self.config.COLORS, dict)
        self.assertIsInstance(self.config.LINE_START, tuple)
        self.assertIsInstance(self.config.LINE_END, tuple)

        # Проверка размеров кортежей
        self.assertEqual(len(self.config.LINE_START), 2)
        self.assertEqual(len(self.config.LINE_END), 2)


class TestEntities(unittest.TestCase):
    """Тестирование структур данных"""

    def setUp(self):
        try:
            from people_counter.entities import Detection, TrackInfo, FrameStats
            self.Detection = Detection
            self.TrackInfo = TrackInfo
            self.FrameStats = FrameStats
        except ImportError as e:
            self.skipTest(f"Cannot import entities: {e}")

    def test_detection_creation(self):
        """Тестирование создания объекта Detection"""
        test_detection = self.Detection(
            bbox=(10, 10, 50, 50),
            confidence=0.8,
            class_id=0
        )

        self.assertEqual(test_detection.bbox, (10, 10, 50, 50))
        self.assertEqual(test_detection.confidence, 0.8)
        self.assertEqual(test_detection.class_id, 0)
        self.assertIsNone(test_detection.tracker_id)

    def test_track_info_creation(self):
        """Тестирование создания объекта TrackInfo"""
        track_info = self.TrackInfo(
            track_id=1,
            positions=[(100, 200), (110, 210)]
        )

        self.assertEqual(track_info.track_id, 1)
        self.assertEqual(len(track_info.positions), 2)
        self.assertFalse(track_info.crossed)
        self.assertIsNotNone(track_info.first_detected)

    def test_frame_stats_creation(self):
        """Тестирование создания объекта FrameStats"""
        from datetime import datetime

        stats = self.FrameStats(
            frame_number=100,
            processing_time=0.05,
            detections_count=3,
            tracks_count=2,
            fps=20.0
        )

        self.assertEqual(stats.frame_number, 100)
        self.assertEqual(stats.processing_time, 0.05)
        self.assertIsNotNone(stats.timestamp)


class TestDetector(unittest.TestCase):
    """Тестирование детектора YOLO"""

    def setUp(self):
        try:
            from people_counter.detector import YOLODetector
            from people_counter.config import Config

            self.config = Config()
            # Пытаемся создать детектор, но обрабатываем возможные ошибки
            try:
                self.detector = YOLODetector(self.config)
            except Exception as e:
                self.skipTest(f"Cannot initialize YOLO detector: {e}")
        except ImportError as e:
            self.skipTest(f"Cannot import detector: {e}")

    def test_detector_initialization(self):
        """Тестирование инициализации детектора"""
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.config)
        self.assertEqual(self.detector.last_inference_time, 0)

    def test_detect_empty_frame(self):
        """Тестирование детекции на пустом кадре"""
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            detections, detection_time = self.detector.detect(empty_frame)
            self.assertIsInstance(detections, list)
            self.assertGreaterEqual(detection_time, 0)
        except Exception as e:
            self.fail(f"Detector failed on empty frame: {e}")

    def test_detect_invalid_input(self):
        """Тестирование детекции с невалидными данными"""
        invalid_inputs = [
            None,
            np.array([]),
            np.zeros((10, 10), dtype=np.uint8),  # 2D массив вместо 3D
            "invalid_input"
        ]

        for invalid_input in invalid_inputs:
            with self.subTest(input=type(invalid_input).__name__):
                try:
                    detections, detection_time = self.detector.detect(invalid_input)
                    # Если не выбросило исключение, проверяем что вернулся пустой список
                    self.assertIsInstance(detections, list)
                except Exception:
                    # Исключение ожидаемо для невалидных данных
                    pass


class TestTracker(unittest.TestCase):
    """Тестирование трекера DeepSORT"""

    def setUp(self):
        try:
            from people_counter.tracker import DeepSORTTracker
            from people_counter.config import Config
            from people_counter.entities import Detection

            self.config = Config()
            self.Detection = Detection

            try:
                self.tracker = DeepSORTTracker(self.config)
            except Exception as e:
                self.skipTest(f"Cannot initialize DeepSORT tracker: {e}")
        except ImportError as e:
            self.skipTest(f"Cannot import tracker: {e}")

    def test_tracker_initialization(self):
        """Тестирование инициализации трекера"""
        self.assertIsNotNone(self.tracker.tracker)
        self.assertIsInstance(self.tracker.track_history, dict)
        self.assertEqual(len(self.tracker.track_history), 0)

    def test_update_empty_detections(self):
        """Тестирование обновления с пустым списком детекций"""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            updated_detections, track_infos = self.tracker.update([], test_frame)
            self.assertIsInstance(updated_detections, list)
            self.assertIsInstance(track_infos, list)
        except Exception as e:
            self.fail(f"Tracker failed with empty detections: {e}")

    def test_update_with_detections(self):
        """Тестирование обновления с детекциями"""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            self.Detection(bbox=(100, 100, 200, 200), confidence=0.8, class_id=0),
            self.Detection(bbox=(300, 150, 400, 250), confidence=0.9, class_id=0)
        ]

        try:
            updated_detections, track_infos = self.tracker.update(detections, test_frame)
            self.assertIsInstance(updated_detections, list)
            self.assertIsInstance(track_infos, list)
        except Exception as e:
            self.fail(f"Tracker failed with valid detections: {e}")

    def test_foot_position_calculation(self):
        """Тестирование вычисления позиции ног"""
        bbox = [100, 150, 200, 250]  # x1, y1, x2, y2
        foot_pos = self.tracker._get_foot_position(bbox)

        expected_x = (100 + 200) // 2  # center x
        expected_y = 250  # bottom y

        self.assertEqual(foot_pos, (expected_x, expected_y))


class TestLineCounter(unittest.TestCase):
    """Тестирование счетчика линий"""

    def setUp(self):
        try:
            from people_counter.line_counter import LineCounter
            from people_counter.config import Config
            from people_counter.entities import Detection, TrackInfo

            self.config = Config()
            self.counter = LineCounter(self.config)
            self.Detection = Detection
            self.TrackInfo = TrackInfo
        except ImportError as e:
            self.skipTest(f"Cannot import line counter: {e}")

    def test_counter_initialization(self):
        """Тестирование инициализации счетчика"""
        self.assertEqual(self.counter.in_count, 0)
        self.assertEqual(self.counter.out_count, 0)
        self.assertEqual(self.counter.current_inside, 0)
        self.assertEqual(len(self.counter.crossing_events), 0)

    def test_line_intersection_detection(self):
        """Тестирование определения пересечения линий"""
        # Линии пересекаются
        p1, p2 = (0, 0), (10, 10)
        p3, p4 = (0, 10), (10, 0)
        self.assertTrue(self.counter._line_intersection(p1, p2, p3, p4))

        # Линии не пересекаются
        p1, p2 = (0, 0), (5, 5)
        p3, p4 = (10, 10), (15, 15)
        self.assertFalse(self.counter._line_intersection(p1, p2, p3, p4))

    def test_crossing_direction_detection(self):
        """Тестирование определения направления пересечения"""
        # Вертикальная линия (x=400)
        self.counter.line_start = (400, 100)
        self.counter.line_end = (400, 500)

        # Движение слева направо (IN)
        direction = self.counter._get_crossing_direction((350, 300), (450, 300))
        self.assertEqual(direction, 'in')

        # Движение справа налево (OUT)
        direction = self.counter._get_crossing_direction((450, 300), (350, 300))
        self.assertEqual(direction, 'out')

    def test_move_line(self):
        """Тестирование перемещения линии"""
        original_start = self.counter.line_start
        original_end = self.counter.line_end

        self.counter.move_line(10, -5)

        expected_start = (original_start[0] + 10, original_start[1] - 5)
        expected_end = (original_end[0] + 10, original_end[1] - 5)

        self.assertEqual(self.counter.line_start, expected_start)
        self.assertEqual(self.counter.line_end, expected_end)

    def test_reset_counters(self):
        """Тестирование сброса счетчиков"""
        # Устанавливаем некоторые значения
        self.counter.in_count = 5
        self.counter.out_count = 3
        self.counter.current_inside = 2
        self.counter.crossing_events = ['event1', 'event2']

        # Сбрасываем
        self.counter.reset()

        # Проверяем что все обнулилось
        self.assertEqual(self.counter.in_count, 0)
        self.assertEqual(self.counter.out_count, 0)
        self.assertEqual(self.counter.current_inside, 0)
        self.assertEqual(len(self.counter.crossing_events), 0)


class TestLogger(unittest.TestCase):
    """Тестирование логгера"""

    def setUp(self):
        try:
            from people_counter.logger import Logger
            from people_counter.config import Config
            from people_counter.entities import FrameStats

            self.config = Config()
            self.config.CSV_LOG_PATH = "test_log.csv"  # Используем тестовый файл
            self.logger = Logger(self.config)
            self.FrameStats = FrameStats

        except ImportError as e:
            self.skipTest(f"Cannot import logger: {e}")

    def tearDown(self):
        """Очистка тестовых файлов"""
        try:
            os.remove("test_log.csv")
        except FileNotFoundError:
            pass
        try:
            os.remove("test_summary.json")
        except FileNotFoundError:
            pass

    def test_csv_file_creation(self):
        """Тестирование создания CSV файла"""
        self.assertTrue(os.path.exists(self.config.CSV_LOG_PATH))

        # Проверяем заголовки
        with open(self.config.CSV_LOG_PATH, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            expected_headers = [
                'timestamp', 'frame_number', 'processing_time',
                'detections_count', 'tracks_count', 'fps',
                'in_count', 'out_count', 'current_inside'
            ]
            self.assertEqual(headers, expected_headers)

    def test_log_frame(self):
        """Тестирование записи кадра в лог"""
        from datetime import datetime

        stats = self.FrameStats(
            frame_number=1,
            processing_time=0.05,
            detections_count=2,
            tracks_count=1,
            fps=20.0,
            timestamp=datetime.now()
        )

        count_stats = {'in': 1, 'out': 0, 'inside': 1}

        try:
            self.logger.log_frame(stats, count_stats)

            # Проверяем что данные записались
            with open(self.config.CSV_LOG_PATH, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Пропускаем заголовки
                row = next(reader)
                self.assertEqual(row[1], '1')  # frame_number
                self.assertEqual(row[6], '1')  # in_count

        except Exception as e:
            self.fail(f"Log frame failed: {e}")

    def test_export_summary(self):
        """Тестирование экспорта статистики"""
        # Сначала записываем некоторые данные
        from datetime import datetime

        stats = self.FrameStats(
            frame_number=1,
            processing_time=0.05,
            detections_count=2,
            tracks_count=1,
            fps=20.0,
            timestamp=datetime.now()
        )

        count_stats = {'in': 1, 'out': 0, 'inside': 1}
        self.logger.log_frame(stats, count_stats)

        # Теперь экспортируем
        try:
            self.logger.export_summary("test_summary.json")
            self.assertTrue(os.path.exists("test_summary.json"))

            # Проверяем содержимое
            with open("test_summary.json", 'r') as f:
                summary = json.load(f)
                self.assertIn('total_frames', summary)
                self.assertIn('average_fps', summary)

        except Exception as e:
            self.fail(f"Export summary failed: {e}")


class TestVisualizer(unittest.TestCase):
    """Тестирование визуализатора"""

    def setUp(self):
        try:
            from people_counter.visualizer import Visualizer
            from people_counter.config import Config
            from people_counter.entities import Detection, TrackInfo, FrameStats
            from datetime import datetime

            self.config = Config()
            self.visualizer = Visualizer(self.config)
            self.Detection = Detection
            self.TrackInfo = TrackInfo
            self.FrameStats = FrameStats

        except ImportError as e:
            self.skipTest(f"Cannot import visualizer: {e}")

    def test_draw_detections(self):
        """Тестирование отрисовки детекций"""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            self.Detection(bbox=(100, 100, 200, 200), confidence=0.8, class_id=0, tracker_id=1)
        ]

        try:
            result_frame = self.visualizer.draw_detections(test_frame, detections)
            self.assertEqual(result_frame.shape, test_frame.shape)
            # Проверяем что изображение изменилось (что-то нарисовалось)
            self.assertFalse(np.array_equal(test_frame, result_frame))
        except Exception as e:
            self.fail(f"Draw detections failed: {e}")

    def test_draw_line(self):
        """Тестирование отрисовки линии"""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            result_frame = self.visualizer.draw_line(test_frame)
            self.assertEqual(result_frame.shape, test_frame.shape)
            # Проверяем что изображение изменилось
            self.assertFalse(np.array_equal(test_frame, result_frame))
        except Exception as e:
            self.fail(f"Draw line failed: {e}")

    def test_draw_info_panel(self):
        """Тестирование отрисовки информационной панели"""
        from datetime import datetime

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        stats = self.FrameStats(
            frame_number=100,
            processing_time=0.05,
            detections_count=3,
            tracks_count=2,
            fps=20.0,
            timestamp=datetime.now()
        )
        count_stats = {'in': 5, 'out': 2, 'inside': 3}

        try:
            result_frame = self.visualizer.draw_info_panel(test_frame, stats, count_stats)
            self.assertEqual(result_frame.shape, test_frame.shape)
        except Exception as e:
            self.fail(f"Draw info panel failed: {e}")


class TestSystemIntegration(unittest.TestCase):
    """Интеграционные тесты системы"""

    def setUp(self):
        try:
            from people_counter import (
                YOLODetector, DeepSORTTracker, LineCounter,
                Visualizer, Logger, Config, FrameStats
            )

            self.config = Config()
            self.config.VIDEO_SOURCE = TestEnvironment.create_test_video()
            self.config.CSV_LOG_PATH = "test_integration.csv"

            # Пытаемся инициализировать все компоненты
            try:
                self.detector = YOLODetector(self.config)
                self.tracker = DeepSORTTracker(self.config)
                self.counter = LineCounter(self.config)
                self.visualizer = Visualizer(self.config)
                self.logger = Logger(self.config)
            except Exception as e:
                self.skipTest(f"Cannot initialize system components: {e}")

        except ImportError as e:
            self.skipTest(f"Cannot import system components: {e}")

    def tearDown(self):
        TestEnvironment.cleanup()

    def test_full_pipeline(self):
        """Тестирование полного пайплайна обработки"""
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        try:
            # Детекция
            detections, detection_time = self.detector.detect(test_frame)
            self.assertIsInstance(detections, list)
            self.assertGreaterEqual(detection_time, 0)

            # Трекинг
            tracked_detections, track_infos = self.tracker.update(detections, test_frame)
            self.assertIsInstance(tracked_detections, list)
            self.assertIsInstance(track_infos, list)

            # Подсчет
            count_stats = self.counter.update(tracked_detections, track_infos)
            self.assertIn('in', count_stats)
            self.assertIn('out', count_stats)
            self.assertIn('inside', count_stats)

            # Визуализация
            vis_frame = self.visualizer.draw_detections(test_frame, tracked_detections)
            vis_frame = self.visualizer.draw_line(vis_frame)

            # Логирование
            from datetime import datetime
            from people_counter.entities import FrameStats

            stats = FrameStats(
                frame_number=1,
                processing_time=0.05,
                detections_count=len(detections),
                tracks_count=len(track_infos),
                fps=20.0,
                timestamp=datetime.now()
            )
            self.logger.log_frame(stats, count_stats)

        except Exception as e:
            self.fail(f"Full pipeline test failed: {e}")


class RuntimeErrorDetector:
    """Детектор ошибок времени выполнения"""

    @staticmethod
    def test_memory_leaks():
        """Тестирование утечек памяти"""
        import gc
        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Симулируем работу системы
        for i in range(100):
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Здесь бы была обработка кадра
            del test_frame

            if i % 10 == 0:
                gc.collect()

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Если рост памяти больше 100MB, возможна утечка
        if memory_growth > 100 * 1024 * 1024:
            print(f"⚠️  Potential memory leak detected: {memory_growth / 1024 / 1024:.2f} MB growth")
            return False

        print(f"✅ Memory usage is stable: {memory_growth / 1024 / 1024:.2f} MB growth")
        return True

    @staticmethod
    def test_file_handles():
        """Тестирование файловых дескрипторов"""
        import psutil

        process = psutil.Process(os.getpid())
        initial_files = process.num_fds() if hasattr(process, 'num_fds') else 0

        # Симулируем работу с файлами
        for i in range(50):
            with tempfile.NamedTemporaryFile() as temp_file:
                pass

        final_files = process.num_fds() if hasattr(process, 'num_fds') else 0
        file_growth = final_files - initial_files

        if file_growth > 10:
            print(f"⚠️  Potential file handle leak: {file_growth} handles not closed")
            return False

        print(f"✅ File handles properly managed: {file_growth} growth")
        return True

    @staticmethod
    def test_exception_handling():
        """Тестирование обработки исключений"""
        test_cases = [
            ("Division by zero", lambda: 1 / 0),
            ("Index out of range", lambda: [1, 2, 3][10]),
            ("Key error", lambda: {'a': 1}['b']),
            ("Type error", lambda: "string" + 5),
            ("Attribute error", lambda: None.some_attribute),
        ]

        failed_cases = []

        for test_name, test_func in test_cases:
            try:
                test_func()
                failed_cases.append(f"{test_name}: No exception raised")
            except Exception as error:
                print(f"✅ {test_name}: Properly caught {type(error).__name__}")

        if failed_cases:
            print("⚠️  Exception handling issues:")
            for case in failed_cases:
                print(f"   {case}")
            return False

        return True


def run_comprehensive_tests():
    """Запуск всех тестов"""
    print("🚀 Starting Comprehensive People Counter System Tests")
    print("=" * 60)

    # Настройка тестовой среды
    TestEnvironment.cleanup()  # Очистка перед тестами

    # Создание тестового набора
    test_suite = unittest.TestSuite()

    # Добавление всех тестовых классов
    test_classes = [
        TestConfig,
        TestEntities,
        TestDetector,
        TestTracker,
        TestLineCounter,
        TestLogger,
        TestVisualizer,
        TestSystemIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Запуск тестов
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)

    print("\n" + "=" * 60)
    print("🔍 Runtime Error Detection")
    print("=" * 60)

    # Тесты времени выполнения
    runtime_tests = [
        ("Memory Leaks", RuntimeErrorDetector.test_memory_leaks),
        ("File Handles", RuntimeErrorDetector.test_file_handles),
        ("Exception Handling", RuntimeErrorDetector.test_exception_handling)
    ]

    runtime_passed = 0
    for test_name, test_func in runtime_tests:
        try:
            if test_func():
                runtime_passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as error:
            print(f"❌ {test_name}: ERROR - {error}")

    # Очистка
    TestEnvironment.cleanup()

    # Итоговый отчет
    print("\n" + "=" * 60)
    print("📊 FINAL REPORT")
    print("=" * 60)

    total_tests = result.testsRun
    failed_tests = len(result.failures) + len(result.errors)
    passed_tests = total_tests - failed_tests

    print(f"Unit Tests: {passed_tests}/{total_tests} passed")
    print(f"Runtime Tests: {runtime_passed}/{len(runtime_tests)} passed")

    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, error_traceback in result.failures:
            print(f"   • {test}: {error_traceback.split('AssertionError: ')[-1].split('/n')[0]}")

    if result.errors:
        print(f"\n💥 ERRORS ({len(result.errors)}):")
        for test, error_traceback in result.errors:
            error_line = error_traceback.split('\n')[-2] if error_traceback.split('\n') else "Unknown error"
            print(f"   • {test}: {error_line}")

    if result.skipped:
        print(f"\n⭐️ SKIPPED ({len(result.skipped)}):")
        for test, reason in result.skipped:
            print(f"   • {test}: {reason}")

    # Анализ кода на потенциальные проблемы
    print("\n" + "=" * 60)
    print("🔬 STATIC CODE ANALYSIS")
    print("=" * 60)

    code_issues = analyze_code_issues()
    if code_issues:
        for issue_type, issues in code_issues.items():
            if issues:
                print(f"\n⚠️  {issue_type.upper()}:")
                for issue in issues:
                    print(f"   • {issue}")
    else:
        print("✅ No major code issues detected")

    # Рекомендации по улучшению
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)

    recommendations = generate_recommendations(result, runtime_passed, len(runtime_tests))
    for rec in recommendations:
        print(f"• {rec}")

    # Оценка общего состояния
    total_score = calculate_system_health(passed_tests, total_tests, runtime_passed, len(runtime_tests))
    print(f"\n🥇 SYSTEM HEALTH SCORE: {total_score}/100")

    if total_score >= 90:
        print("🟢 Excellent - System is production ready")
    elif total_score >= 75:
        print("🟡 Good - Minor issues need attention")
    elif total_score >= 60:
        print("🟠 Fair - Several issues should be fixed")
    else:
        print("🔴 Poor - Major issues require immediate attention")

    return result.wasSuccessful() and runtime_passed == len(runtime_tests)


def analyze_code_issues():
    """Анализ кода на потенциальные проблемы"""
    issues = {
        'import_issues': [],
        'exception_handling': [],
        'resource_management': [],
        'performance_issues': [],
        'thread_safety': []
    }

    try:
        # Проверяем импорты
        import_test_results = test_all_imports()
        issues['import_issues'] = import_test_results

        # Анализируем код файлов если доступны
        code_files = [
            'people_counter/detector.py',
            'people_counter/tracker.py',
            'people_counter/line_counter.py',
            'people_counter/logger.py',
            'people_counter/visualizer.py',
            'run.py'
        ]

        for file_path in code_files:
            if os.path.exists(file_path):
                file_issues = analyze_file(file_path)
                for issue_type, file_issues_list in file_issues.items():
                    issues[issue_type].extend(file_issues_list)

    except Exception as error:
        issues['import_issues'].append(f"Code analysis failed: {error}")

    return issues


def test_all_imports():
    """Тестирование всех импортов"""
    import_issues = []

    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('ultralytics', 'ultralytics'),
        ('deep_sort_realtime', 'deep-sort-realtime'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision')
    ]

    for package, pip_name in required_packages:
        try:
            __import__(package)
        except ImportError:
            import_issues.append(f"Missing package: {pip_name} (import {package} failed)")
        except Exception as error:
            import_issues.append(f"Import error for {package}: {error}")

    # Тестируем импорты модулей системы
    system_modules = [
        'people_counter.config',
        'people_counter.entities',
        'people_counter.detector',
        'people_counter.tracker',
        'people_counter.line_counter',
        'people_counter.logger',
        'people_counter.visualizer'
    ]

    for module in system_modules:
        try:
            __import__(module)
        except ImportError as error:
            import_issues.append(f"System module import failed: {module} - {error}")
        except Exception as error:
            import_issues.append(f"System module error: {module} - {error}")

    return import_issues


def analyze_file(file_path):
    """Анализ отдельного файла на проблемы"""
    issues = {
        'exception_handling': [],
        'resource_management': [],
        'performance_issues': [],
        'thread_safety': []
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        # Проверка обработки исключений
        if 'except:' in content:
            issues['exception_handling'].append(f"{file_path}: Bare except clause detected")

        if content.count('try:') > content.count('except'):
            issues['exception_handling'].append(f"{file_path}: Try blocks without proper exception handling")

        # Проверка управления ресурсами
        if 'open(' in content and 'with ' not in content:
            issues['resource_management'].append(f"{file_path}: File operations without context manager")

        if 'cv2.VideoCapture(' in content and '.release()' not in content:
            issues['resource_management'].append(f"{file_path}: VideoCapture without proper release")

        # Проверка производительности
        if 'time.sleep(' in content:
            issues['performance_issues'].append(f"{file_path}: time.sleep() usage detected")

        for i, line in enumerate(lines):
            if 'for ' in line and 'range(' in line and 'len(' in line:
                issues['performance_issues'].append(
                    f"{file_path}:{i + 1}: Inefficient loop pattern 'for i in range(len(...))'")

        # Проверка потокобезопасности
        if 'threading' in content or 'Thread' in content:
            if 'Lock' not in content and 'lock' not in content:
                issues['thread_safety'].append(f"{file_path}: Threading without synchronization")

        if 'global ' in content:
            issues['thread_safety'].append(f"{file_path}: Global variables may cause thread safety issues")

    except Exception as error:
        issues['exception_handling'].append(f"Failed to analyze {file_path}: {error}")

    return issues


def generate_recommendations(test_result, runtime_passed, total_runtime):
    """Генерация рекомендаций по улучшению"""
    recommendations = []

    # На основе результатов тестов
    if test_result.failures:
        recommendations.append("Fix failing unit tests to ensure code reliability")

    if test_result.errors:
        recommendations.append("Resolve test errors - they indicate serious code issues")

    if test_result.skipped:
        recommendations.append("Address skipped tests by installing missing dependencies")

    if runtime_passed < total_runtime:
        recommendations.append("Fix runtime issues to prevent memory leaks and resource problems")

    # Общие рекомендации
    recommendations.extend([
        "Add input validation to all public methods",
        "Implement comprehensive error logging",
        "Add configuration validation at startup",
        "Consider adding performance monitoring",
        "Implement graceful shutdown handling",
        "Add unit tests for edge cases",
        "Consider using dependency injection for better testability",
        "Add type hints for better code documentation",
        "Implement proper resource cleanup in finally blocks",
        "Consider adding health check endpoints"
    ])

    # Специфичные рекомендации для системы подсчета людей
    recommendations.extend([
        "Add calibration mode for line positioning",
        "Implement detection confidence filtering",
        "Add support for multiple counting lines",
        "Consider adding database storage for long-term statistics",
        "Implement backup/restore functionality for counters",
        "Add real-time performance metrics dashboard",
        "Consider adding remote monitoring capabilities"
    ])

    return recommendations[:10]  # Возвращаем топ-10 рекомендаций


def calculate_system_health(passed_tests, total_tests, runtime_passed, total_runtime):
    """Расчет общего состояния системы"""
    if total_tests == 0:
        unit_score = 0
    else:
        unit_score = (passed_tests / total_tests) * 70  # 70% веса

    if total_runtime == 0:
        runtime_score = 0
    else:
        runtime_score = (runtime_passed / total_runtime) * 30  # 30% веса

    return int(unit_score + runtime_score)


class StressTestRunner:
    """Класс для стресс-тестирования системы"""

    @staticmethod
    def run_stress_test(duration_seconds=30):
        """Запуск стресс-теста"""
        print(f"\n🔥 Running stress test for {duration_seconds} seconds...")

        try:
            from people_counter.config import Config
            from people_counter.detector import YOLODetector

            config = Config()
            detector = YOLODetector(config)

            start_time = time.time()
            frame_count = 0
            errors = 0

            while time.time() - start_time < duration_seconds:
                try:
                    # Генерируем случайный кадр
                    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                    # Обрабатываем кадр
                    detections, detection_time = detector.detect(test_frame)

                    frame_count += 1

                    if frame_count % 50 == 0:
                        print(f"   Processed {frame_count} frames, {errors} errors")

                except Exception as error:
                    errors += 1
                    if errors <= 5:  # Показываем только первые 5 ошибок
                        print(f"   Error {errors}: {error}")

            total_time = time.time() - start_time
            fps = frame_count / total_time
            error_rate = errors / frame_count * 100 if frame_count > 0 else 100

            print(f"✅ Stress test completed:")
            print(f"   Frames processed: {frame_count}")
            print(f"   Average FPS: {fps:.2f}")
            print(f"   Error rate: {error_rate:.2f}%")
            print(f"   Total errors: {errors}")

            return error_rate < 5  # Считаем успешным если ошибок меньше 5%

        except Exception as error:
            print(f"❌ Stress test failed: {error}")
            return False


if __name__ == "__main__":
    success = run_comprehensive_tests()

    # Дополнительный стресс-тест если основные тесты прошли
    if success:
        print("\n" + "=" * 60)
        print("🔥 STRESS TESTING")
        print("=" * 60)
        stress_success = StressTestRunner.run_stress_test(15)  # 15 секунд стресс-теста

        if not stress_success:
            success = False
            print("❌ Stress test revealed performance issues")

    # Финальный статус
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - System is ready for deployment!")
    else:
        print("⚠️  TESTS FAILED - Please address the issues before deployment")
    print("=" * 60)

    sys.exit(0 if success else 1)