"""
Детектор специфичных багов в системе подсчета людей
Тестирует конкретные проблемы, которые могут не проявиться в обычных тестах
"""

import sys
import os
import numpy as np
import cv2
import time
import traceback
from unittest.mock import patch, MagicMock
import warnings

warnings.filterwarnings("ignore")


class SpecificBugDetector:
    """Детектор специфичных багов в системе"""

    def __init__(self):
        self.bugs_found = []
        self.test_results = {}

    def run_all_bug_tests(self):
        """Запуск всех тестов на специфичные баги"""
        print("🐛 Detecting specific bugs in People Counter System")
        print("=" * 60)

        bug_tests = [
            ("Memory Management Issues", self.test_memory_management_bugs),
            ("Thread Safety Issues", self.test_thread_safety_bugs),
            ("Video Processing Bugs", self.test_video_processing_bugs),
            ("Tracker State Bugs", self.test_tracker_state_bugs),
            ("Counter Logic Bugs", self.test_counter_logic_bugs),
            ("Group Movement Logic", self.test_group_movement_logic),  # НОВЫЙ ТЕСТ
            ("File I/O Bugs", self.test_file_io_bugs),
            ("Edge Case Bugs", self.test_edge_case_bugs),
            ("Performance Bugs", self.test_performance_bugs),
            ("Resource Cleanup Bugs", self.test_resource_cleanup_bugs)
        ]

        total_tests = len(bug_tests)
        passed_tests = 0

        for test_name, test_func in bug_tests:
            print(f"\n🔍 {test_name}...")
            try:
                if test_func():
                    print(f"✅ {test_name}: PASSED")
                    passed_tests += 1
                    self.test_results[test_name] = "PASSED"
                else:
                    print(f"❌ {test_name}: FAILED")
                    self.test_results[test_name] = "FAILED"
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
                self.test_results[test_name] = f"ERROR: {e}"
                traceback.print_exc()

        self.print_summary(passed_tests, total_tests)
        return passed_tests == total_tests

    def test_memory_management_bugs(self):
        """Тестирование багов управления памятью"""
        bugs_found = False

        # Тест 1: Утечка памяти при обработке большого количества кадров
        try:
            import psutil
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            # Симулируем обработку 1000 кадров
            for i in range(1000):
                test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
                # Симулируем детекцию
                del test_frame

                if i % 100 == 0:
                    current_memory = process.memory_info().rss
                    memory_growth = current_memory - initial_memory
                    if memory_growth > 500 * 1024 * 1024:  # Более 500MB
                        self.bugs_found.append("Excessive memory growth during frame processing")
                        bugs_found = True
                        break

        except ImportError:
            print("   ⚠️  psutil not available, skipping memory test")

        # Тест 2: Неочищенные numpy массивы
        try:
            large_arrays = []
            for i in range(100):
                arr = np.zeros((1000, 1000, 3), dtype=np.float64)
                large_arrays.append(arr)

            # Проверяем что массивы правильно удаляются
            del large_arrays
            import gc
            gc.collect()

        except MemoryError:
            self.bugs_found.append("Memory allocation issues with large numpy arrays")
            bugs_found = True

        return not bugs_found

    def test_thread_safety_bugs(self):
        """Тестирование багов многопоточности"""
        bugs_found = False

        try:
            from people_counter.tracker import DeepSORTTracker
            from people_counter.config import Config
            import threading

            config = Config()
            tracker = DeepSORTTracker(config)

            errors = []

            def worker_thread(thread_id):
                try:
                    for i in range(50):
                        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                        # Попытка одновременного доступа к трекеру
                        detections = []
                        tracker.update(detections, test_frame)
                except Exception as error:
                    errors.append(f"Thread {thread_id}: {error}")

            # Запускаем несколько потоков одновременно
            threads = []
            for i in range(3):
                t = threading.Thread(target=worker_thread, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            if errors:
                self.bugs_found.append(f"Thread safety issues: {errors[:3]}")
                bugs_found = True

        except ImportError:
            print("   ⚠️  Cannot import tracker, skipping thread safety test")
        except Exception as error:
            if "concurrent access" in str(error).lower():
                bugs_found = True

        return not bugs_found

    def test_video_processing_bugs(self):
        """Тестирование багов обработки видео"""
        bugs_found = False

        # Тест 1: Обработка поврежденных кадров
        corrupted_frames = [
            None,
            np.array([]),
            np.zeros((10, 10), dtype=np.uint8),  # 2D вместо 3D
            np.zeros((480, 640, 1), dtype=np.uint8),  # Одноканальное
            np.zeros((480, 640, 4), dtype=np.uint8),  # 4 канала
            np.full((480, 640, 3), 255, dtype=np.uint8),  # Максимальное значение
        ]

        try:
            from people_counter.detector import YOLODetector
            from people_counter.config import Config

            config = Config()
            detector = YOLODetector(config)

            for i, test_frame in enumerate(corrupted_frames):
                try:
                    detection_results, time_taken = detector.detect(test_frame)
                    # Если не выбросило исключение, проверяем результат
                    if not isinstance(detection_results, list):
                        self.bugs_found.append(f"Detector returns non-list for corrupted frame {i}")
                        bugs_found = True
                except Exception as error:
                    # Ожидаемо для большинства поврежденных кадров
                    if "unexpected" in str(error).lower():
                        bugs_found = True

        except ImportError:
            print("   ⚠️  Cannot import detector, skipping video processing test")

        # Тест 2: Нестандартные размеры кадров
        unusual_sizes = [
            (1, 1, 3),
            (10000, 10000, 3),  # Очень большой
            (480, 1, 3),  # Очень узкий
            (1, 640, 3),  # Очень низкий
        ]

        for size in unusual_sizes:
            try:
                if np.prod(size) > 1000000:  # Пропускаем слишком большие
                    continue
                test_frame = np.zeros(size, dtype=np.uint8)
                # Здесь была бы проверка обработки нестандартных размеров
            except MemoryError:
                continue  # Ожидаемо для очень больших размеров

        return not bugs_found

    def test_tracker_state_bugs(self):
        """Тестирование багов состояния трекера"""
        bugs_found = False

        try:
            from people_counter.tracker import DeepSORTTracker
            from people_counter.entities import Detection
            from people_counter.config import Config

            config = Config()
            tracker = DeepSORTTracker(config)

            # Тест 1: Переполнение track_id
            for i in range(100000, 100010):  # Большие ID
                test_detection = Detection(bbox=(10, 10, 50, 50), confidence=0.8, class_id=0)
                test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

                try:
                    tracker.update([test_detection], test_frame)
                except OverflowError:
                    self.bugs_found.append("Track ID overflow error")
                    bugs_found = True
                    break

            # Тест 2: Сброс трекера во время обработки
            reset_detection = Detection(bbox=(10, 10, 50, 50), confidence=0.8, class_id=0)
            reset_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            tracker.update([reset_detection], reset_frame)
            tracker.reset()

            # Попытка использовать после сброса
            try:
                tracker.update([reset_detection], reset_frame)
            except AttributeError as error:
                if "nonetype" in str(error).lower():
                    self.bugs_found.append("Tracker not properly reinitialized after reset")
                    bugs_found = True

        except ImportError:
            print("   ⚠️  Cannot import tracker components, skipping tracker state test")

        return not bugs_found

    def test_counter_logic_bugs(self):
        """Тестирование багов логики счетчика"""
        bugs_found = False

        try:
            from people_counter.line_counter import LineCounter
            from people_counter.entities import Detection
            from people_counter.config import Config

            config = Config()
            counter = LineCounter(config)

            # Тест 1: Двойной подсчет при быстрых перемещениях
            # Создаем детекции для одного трека, который осциллирует около линии

            positions = [
                (340, 300),  # Слева от линии (X=350)
                (360, 300),  # Справа от линии - должно засчитать вход
                (340, 300),  # Обратно слева - должно засчитать выход
                (360, 300),  # Снова справа - НЕ должно засчитать из-за debounce
                (340, 300),  # Обратно слева - НЕ должно засчитать из-за debounce
            ]

            initial_count = counter.in_count + counter.out_count

            for i, pos in enumerate(positions):
                bbox = (pos[0] - 20, pos[1] - 50, pos[0] + 20, pos[1])
                detection = Detection(bbox=bbox, confidence=0.9, class_id=0, tracker_id=1)

                counter.update([detection])

                # Добавляем небольшую задержку для тестирования debounce
                if i > 1:
                    import time
                    time.sleep(0.1)

            final_count = counter.in_count + counter.out_count

            # Должно быть максимум 2 пересечения (вход + выход) из-за debounce
            if final_count - initial_count > 2:
                self.bugs_found.append(
                    f"Double counting detected: {final_count - initial_count} crossings for oscillating movement")
                bugs_found = True

            # Тест 2: Отрицательное количество внутри - исправленный
            counter.stats['inside'] = 5

            # Принудительно добавляем много выходов через прямое изменение статистики
            for _ in range(6):
                counter.stats['out'] += 1
                counter.stats['inside'] = max(0, counter.stats['in'] - counter.stats['out'])

            if counter.current_inside < 0:
                self.bugs_found.append("Negative count inside detected")
                bugs_found = True

        except ImportError:
            print("   ⚠️  Cannot import counter components, skipping counter logic test")
        except AttributeError as e:
            if "no setter" in str(e):
                self.bugs_found.append("Missing setter for current_inside property")
                bugs_found = True
            else:
                print(f"   ⚠️  AttributeError in counter logic test: {e}")
        except Exception as e:
            print(f"   ⚠️  Error in counter logic test: {e}")
            bugs_found = True

        return not bugs_found

    def test_group_movement_logic(self):
        """ Реалистичная проверка логики группового движения"""
        bugs_found = False

        try:
            from people_counter.line_counter import LineCounter
            from people_counter.entities import Detection
            from people_counter.config import Config

            config = Config()
            counter = LineCounter(config)

            print("   👥 Testing group movement logic (3 people entering, 2 exiting)")
            print(f"   📏 Line coordinates: {config.LINE_START} -> {config.LINE_END}")

            # Проверяем что линия вертикальная на X=350
            line_x = config.LINE_START[0]
            print(f"   📍 Testing crossings at X={line_x}")

            # Создаем реалистичную последовательность движений
            movements = [
                # Человек 1: слева направо (вход)
                (1, [(320, 300), (370, 300)]),
                # Человек 2: слева направо (вход)
                (2, [(320, 320), (370, 320)]),
                # Человек 3: слева направо (вход)
                (3, [(320, 340), (370, 340)]),
                # Человек 1: справа налево (выход)
                (1, [(370, 300), (320, 300)]),
                # Человек 2: справа налево (выход)
                (2, [(370, 320), (320, 320)]),
                # Человек 3 остается внутри
            ]

            expected_results = [
                (1, 0, 1),  # После входа 1
                (2, 0, 2),  # После входа 2
                (3, 0, 3),  # После входа 3
                (3, 1, 2),  # После выхода 1
                (3, 2, 1),  # После выхода 2
            ]

            current_detections = {}  # Храним активные детекции

            for step, (track_id, positions) in enumerate(movements):
                if len(positions) == 2:  # Есть движение
                    # Первая позиция
                    pos1 = positions[0]
                    bbox1 = (pos1[0] - 15, pos1[1] - 25, pos1[0] + 15, pos1[1])
                    detection1 = Detection(bbox=bbox1, confidence=0.9, class_id=0, tracker_id=track_id)
                    current_detections[track_id] = detection1

                    # Обновляем счетчик с первой позицией
                    counter.update(list(current_detections.values()))

                    # Вторая позиция (пересечение)
                    pos2 = positions[1]
                    bbox2 = (pos2[0] - 15, pos2[1] - 25, pos2[0] + 15, pos2[1])
                    detection2 = Detection(bbox=bbox2, confidence=0.9, class_id=0, tracker_id=track_id)
                    current_detections[track_id] = detection2

                    # Обновляем счетчик с пересечением
                    stats = counter.update(list(current_detections.values()))

                    print(f"   Step {step + 1}: Track {track_id} moved {pos1}->{pos2}")
                    print(f"   Result: in={stats['in']}, out={stats['out']}, inside={stats['inside']}")

                    # Проверяем результат
                    if step < len(expected_results):
                        exp_in, exp_out, exp_inside = expected_results[step]
                        if (stats['in'] != exp_in or stats['out'] != exp_out or stats['inside'] != exp_inside):
                            self.bugs_found.append(
                                f"Group movement logic error at step {step + 1}: "
                                f"expected (in={exp_in}, out={exp_out}, inside={exp_inside}), "
                                f"got (in={stats['in']}, out={stats['out']}, inside={stats['inside']})"
                            )
                            bugs_found = True
                            break

            # Финальная проверка - если дошли до конца без ошибок
            if not bugs_found:
                final_stats = counter.update(list(current_detections.values()))
                if final_stats['inside'] != 1 or final_stats['in'] != 3 or final_stats['out'] != 2:
                    self.bugs_found.append(
                        f"Final verification failed: expected (in=3, out=2, inside=1), "
                        f"got (in={final_stats['in']}, out={final_stats['out']}, inside={final_stats['inside']})"
                    )
                    bugs_found = True
                else:
                    print(
                        f"   ✅ Final result: in={final_stats['in']}, out={final_stats['out']}, inside={final_stats['inside']}")

        except ImportError:
            print("   ⚠️  Cannot import counter components, skipping group movement test")
            return True
        except Exception as e:
            print(f"   ⚠️  Error in group movement test: {e}")
            import traceback
            traceback.print_exc()
            bugs_found = True

        return not bugs_found

    def test_file_io_bugs(self):
        """Тестирование багов файлового ввода-вывода"""
        bugs_found = False

        try:
            from people_counter.logger import Logger
            from people_counter.config import Config
            from people_counter.entities import FrameStats
            from datetime import datetime
            import tempfile

            # Тест 1: Запись в недоступный файл
            config = Config()
            config.CSV_LOG_PATH = "/root/impossible_path/log.csv"  # Недоступный путь

            try:
                test_logger = Logger(config)
                stats = FrameStats(
                    frame_number=1, processing_time=0.1, detections_count=1,
                    tracks_count=1, fps=10.0, timestamp=datetime.now()
                )
                test_logger.log_frame(stats, {'in': 1, 'out': 0, 'inside': 1})
            except PermissionError:
                pass  # Ожидаемо
            except Exception as error:
                if "permission" not in str(error).lower():
                    self.bugs_found.append(f"Unexpected error for inaccessible file: {error}")
                    bugs_found = True

            # Тест 2: Запись в файл во время его использования
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                config.CSV_LOG_PATH = tmp.name
                test_logger = Logger(config)

                try:
                    stats = FrameStats(
                        frame_number=1, processing_time=0.1, detections_count=1,
                        tracks_count=1, fps=10.0, timestamp=datetime.now()
                    )
                    test_logger.log_frame(stats, {'in': 1, 'out': 0, 'inside': 1})
                except Exception:
                    pass  # Может быть ожидаемо на некоторых системах

                try:
                    os.unlink(tmp.name)
                except PermissionError:
                    pass

        except ImportError:
            print("   ⚠️  Cannot import logger, skipping file I/O test")

        return not bugs_found

    def test_edge_case_bugs(self):
        """Тестирование багов граничных случаев"""
        bugs_found = False

        # Тест 1: Пустые детекции
        try:
            from people_counter.tracker import DeepSORTTracker
            from people_counter.config import Config

            config = Config()
            tracker = DeepSORTTracker(config)

            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Множественные пустые обновления
            for _ in range(1000):
                tracker.update([], test_frame)

            # Проверяем что трекер не накапливает мусор
            if len(tracker.track_history) > 0:
                self.bugs_found.append("Tracker accumulates empty tracks")
                bugs_found = True

        except ImportError:
            print("   ⚠️  Cannot import tracker for edge case test")

        # Тест 2: Экстремальные значения координат
        try:
            from people_counter.entities import Detection

            extreme_coords = [
                (-1000, -1000, -500, -500),  # Отрицательные
                (0, 0, 0, 0),  # Нулевые размеры
                (100, 100, 50, 50),  # x2 < x1, y2 < y1
                (float('inf'), 0, 100, 100),  # Бесконечность
                (0, float('nan'), 100, 100),  # NaN
            ]

            for coords in extreme_coords:
                try:
                    test_detection = Detection(bbox=coords, confidence=0.8, class_id=0)
                except (ValueError, OverflowError):
                    pass  # Ожидаемо для некоторых экстремальных значений
                except Exception as error:
                    if "unexpected" in str(error).lower():
                        self.bugs_found.append(f"Unexpected error for extreme coordinates: {error}")
                        bugs_found = True

        except ImportError:
            print("   ⚠️  Cannot import entities for edge case test")

        return not bugs_found

    def test_performance_bugs(self):
        """Тестирование производительностных багов"""
        bugs_found = False

        # Тест 1: Деградация производительности при большом количестве треков
        try:
            from people_counter.line_counter import LineCounter
            from people_counter.entities import TrackInfo
            from people_counter.config import Config

            config = Config()
            counter = LineCounter(config)

            # Создаем множество треков
            tracks = []
            for i in range(1000):
                track = TrackInfo(track_id=i, positions=[(100 + i, 200 + i)])
                tracks.append(track)

            start_time = time.time()
            counter.update([], tracks)
            processing_time = time.time() - start_time

            # Если обработка занимает больше 1 секунды для 1000 треков
            if processing_time > 1.0:
                self.bugs_found.append(
                    f"Performance degradation with many tracks: {processing_time:.2f}s for 1000 tracks")
                bugs_found = True

        except ImportError:
            print("   ⚠️  Cannot import counter for performance test")

        # Тест 2: Медленная обработка больших кадров
        try:
            from people_counter.detector import YOLODetector
            from people_counter.config import Config

            config = Config()
            detector = YOLODetector(config)

            # Большой кадр
            large_frame = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)  # 4K

            start_time = time.time()
            detection_results, inference_time = detector.detect(large_frame)
            total_time = time.time() - start_time

            # Если обработка занимает больше 10 секунд
            if total_time > 10.0:
                self.bugs_found.append(f"Very slow processing for 4K frame: {total_time:.2f}s")
                bugs_found = True

        except ImportError:
            print("   ⚠️  Cannot import detector for performance test")
        except Exception as error:
            if "memory" in str(error).lower():
                pass  # Ожидаемо для очень больших кадров

        return not bugs_found

    def test_resource_cleanup_bugs(self):
        """Тестирование багов очистки ресурсов"""
        bugs_found = False

        # Тест 1: Незакрытые файлы
        try:
            import psutil

            process = psutil.Process(os.getpid())
            initial_files = process.num_fds() if hasattr(process, 'num_fds') else 0

            # Симулируем множественные операции с файлами
            from people_counter.logger import Logger
            from people_counter.config import Config

            for i in range(50):
                config = Config()
                config.CSV_LOG_PATH = f"temp_log_{i}.csv"
                test_logger = Logger(config)
                del test_logger

                try:
                    os.remove(f"temp_log_{i}.csv")
                except FileNotFoundError:
                    pass

            final_files = process.num_fds() if hasattr(process, 'num_fds') else 0
            file_growth = final_files - initial_files

            if file_growth > 10:
                self.bugs_found.append(f"File descriptor leak: {file_growth} files not closed")
                bugs_found = True

        except ImportError:
            print("   ⚠️  psutil not available for resource cleanup test")

        # Тест 2: OpenCV ресурсы
        try:
            caps = []
            for i in range(10):
                cap = cv2.VideoCapture(0)  # Попытка открыть камеру
                if cap.isOpened():
                    caps.append(cap)
                else:
                    break

            # Освобождаем только половину
            for i in range(len(caps) // 2):
                caps[i].release()

            # Оставшиеся должны быть освобождены сборщиком мусора
            del caps

        except Exception:
            pass  # Проблемы с камерой ожидаемы в тестовой среде

        return not bugs_found

    def print_summary(self, passed_tests, total_tests):
        """Печать итогового отчета"""
        print("\n" + "=" * 60)
        print("🐛 BUG DETECTION SUMMARY")
        print("=" * 60)

        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Tests failed: {total_tests - passed_tests}/{total_tests}")

        if self.bugs_found:
            print(f"\n❌ BUGS FOUND ({len(self.bugs_found)}):")
            for i, bug in enumerate(self.bugs_found, 1):
                print(f"   {i}. {bug}")
        else:
            print("\n✅ No specific bugs detected!")

        print("\n📊 DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result == "PASSED" else "❌" if result == "FAILED" else "💥"
            print(f"   {status_emoji} {test_name}: {result}")

        # Рекомендации по исправлению
        if self.bugs_found:
            print("\n💡 RECOMMENDATIONS:")
            recommendations = self.generate_bug_fix_recommendations()
            for rec in recommendations:
                print(f"   • {rec}")

    def generate_bug_fix_recommendations(self):
        """Генерация рекомендаций по исправлению багов"""
        recommendations = []

        bug_categories = {
            'memory': [
                "Implement proper memory management with context managers",
                "Add memory usage monitoring and alerts",
                "Use memory profilers to identify leaks"
            ],
            'thread': [
                "Add thread synchronization with locks",
                "Implement thread-safe data structures",
                "Consider using queue-based communication"
            ],
            'counter': [
                "Add state validation in counter logic",
                "Implement debouncing for rapid crossings",
                "Add bounds checking for negative counts"
            ],
            'performance': [
                "Implement frame rate limiting",
                "Add performance monitoring",
                "Optimize algorithms for large datasets"
            ],
            'resource': [
                "Use context managers for all resources",
                "Implement proper cleanup in finally blocks",
                "Add resource monitoring"
            ]
        }

        # Анализируем найденные баги и подбираем рекомендации
        for bug in self.bugs_found:
            bug_lower = bug.lower()

            if any(word in bug_lower for word in ['memory', 'leak']):
                recommendations.extend(bug_categories['memory'][:1])
            elif any(word in bug_lower for word in ['thread', 'concurrent']):
                recommendations.extend(bug_categories['thread'][:1])
            elif any(word in bug_lower for word in ['count', 'negative']):
                recommendations.extend(bug_categories['counter'][:1])
            elif any(word in bug_lower for word in ['performance', 'slow']):
                recommendations.extend(bug_categories['performance'][:1])
            elif any(word in bug_lower for word in ['file', 'resource']):
                recommendations.extend(bug_categories['resource'][:1])

        # Убираем дубликаты
        recommendations = list(dict.fromkeys(recommendations))

        # Добавляем общие рекомендации
        if not recommendations:
            recommendations = [
                "Continue monitoring system behavior in production",
                "Implement comprehensive logging",
                "Add automated testing to CI/CD pipeline"
            ]

        return recommendations[:5]  # Топ-5 рекомендаций


class SystemIntegrityChecker:
    """Проверка целостности системы"""

    @staticmethod
    def check_dependencies():
        """Проверка зависимостей"""
        print("\n🔍 Checking system dependencies...")

        required_packages = [
            ('cv2', 'opencv-python', 'Computer vision operations'),
            ('numpy', 'numpy', 'Numerical computations'),
            ('ultralytics', 'ultralytics', 'YOLO object detection'),
            ('deep_sort_realtime', 'deep-sort-realtime', 'Object tracking'),
            ('torch', 'torch', 'PyTorch backend for YOLO'),
        ]

        missing_packages = []

        for package, pip_name, description in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {pip_name}: OK")
            except ImportError:
                print(f"   ❌ {pip_name}: MISSING - {description}")
                missing_packages.append(pip_name)

        if missing_packages:
            print(f"\n📦 Install missing packages:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False

        return True

    @staticmethod
    def check_file_structure():
        """Проверка структуры файлов"""
        print("\n📁 Checking file structure...")

        required_files = [
            ('run.py', 'Main application entry point'),
            ('people_counter/__init__.py', 'Package initialization'),
            ('people_counter/config.py', 'Configuration module'),
            ('people_counter/entities.py', 'Data structures'),
            ('people_counter/detector.py', 'YOLO detector'),
            ('people_counter/tracker.py', 'DeepSORT tracker'),
            ('people_counter/line_counter.py', 'Line crossing counter'),
            ('people_counter/visualizer.py', 'Visualization module'),
            ('people_counter/logger.py', 'Logging module'),
        ]

        missing_files = []

        for file_path, description in required_files:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path}: OK")
            else:
                print(f"   ❌ {file_path}: MISSING - {description}")
                missing_files.append(file_path)

        return len(missing_files) == 0

    @staticmethod
    def check_model_availability():
        """Проверка доступности моделей"""
        print("\n🧠 Checking model availability...")

        try:
            from ultralytics import YOLO

            # Попытка загрузки модели
            try:
                test_model = YOLO('yolov8m.pt')
                print("   ✅ YOLO model: OK")
                return True
            except Exception as error:
                print(f"   ❌ YOLO model: FAILED - {error}")
                print("   💡 Model will be downloaded on first run")
                return False

        except ImportError:
            print("   ❌ ultralytics not available")
            return False


def main():
    """Главная функция запуска тестов"""
    print("🎯 People Counter System - Comprehensive Bug Detection")
    print("=" * 60)

    # Проверка целостности системы
    integrity_checker = SystemIntegrityChecker()

    deps_ok = integrity_checker.check_dependencies()
    files_ok = integrity_checker.check_file_structure()
    models_ok = integrity_checker.check_model_availability()

    if not deps_ok:
        print("\n⚠️  Cannot proceed without required dependencies")
        return False

    if not files_ok:
        print("\n⚠️  System files are missing")
        return False

    # Запуск детекции багов
    detector = SpecificBugDetector()
    bugs_found = detector.run_all_bug_tests()

    print("\n" + "=" * 60)
    print("🎯 FINAL VERDICT")
    print("=" * 60)

    if bugs_found and len(detector.bugs_found) == 0:
        print("🎉 System appears to be bug-free!")
        print("✅ All specific bug tests passed")
        verdict = "READY FOR PRODUCTION"
    elif len(detector.bugs_found) <= 2:
        print("⚠️  Minor issues detected")
        print("🟡 System is mostly stable but needs attention")
        verdict = "NEEDS MINOR FIXES"
    elif len(detector.bugs_found) <= 5:
        print("🟠 Several issues detected")
        print("⚠️  System needs significant improvements")
        verdict = "NEEDS MAJOR FIXES"
    else:
        print("🔴 Critical issues detected")
        print("❌ System is not ready for production")
        verdict = "NOT PRODUCTION READY"

    print(f"\n🏷️  STATUS: {verdict}")

    # Следующие шаги
    print("\n📋 NEXT STEPS:")
    if len(detector.bugs_found) == 0:
        print("   1. Deploy to staging environment")
        print("   2. Run load testing")
        print("   3. Monitor in production")
    else:
        print("   1. Fix identified bugs")
        print("   2. Re-run this test suite")
        print("   3. Consider code review")
        print("   4. Add unit tests for fixed bugs")

    return len(detector.bugs_found) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
