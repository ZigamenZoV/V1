"""
Графический интерфейс для выбора источника видео
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import sys
from pathlib import Path


class VideoSourceSelector:
    def __init__(self):
        self.selected_source = None
        self.selected_value = 0  # Значение по умолчанию

        try:
            self.root = tk.Tk()
            self.root.title("Выбор источника видео - People Counter")
            self.root.geometry("600x500")
            self.root.resizable(False, False)

            # Центрирование окна
            self.center_window()
            self.setup_ui()

        except Exception as e:
            print(f"Ошибка инициализации GUI: {e}")
            # Fallback - возвращаем камеру по умолчанию
            self.selected_value = 0

    def center_window(self):
        """Центрирование окна на экране"""
        try:
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'{width}x{height}+{x}+{y}')
        except Exception as e:
            print(f"Предупреждение: не удалось центрировать окно - {e}")

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        try:
            # Заголовок
            title_label = tk.Label(
                self.root,
                text="Настройка источника видео",
                font=("Arial", 16, "bold"),
                pady=20
            )
            title_label.pack()

            # Описание
            desc_label = tk.Label(
                self.root,
                text="Выберите источник видео для системы подсчета людей:",
                font=("Arial", 10),
                pady=10
            )
            desc_label.pack()

            # Фрейм с вариантами выбора
            options_frame = ttk.LabelFrame(self.root, text="Доступные источники", padding=15)
            options_frame.pack(padx=20, pady=10, fill="both", expand=True)

            # Переменная для радиокнопок
            self.source_var = tk.StringVar(value="camera")

            # Вариант 1: Веб-камеры
            self.camera_var = tk.StringVar(value="0")
            camera_frame = ttk.Frame(options_frame)
            camera_frame.pack(fill="x", pady=5)

            tk.Radiobutton(
                camera_frame,
                text="📷 Веб-камера:",
                variable=self.source_var,
                value="camera"
            ).pack(side="left")

            # Автоопределение камер
            available_cameras = self.detect_available_cameras()
            camera_combobox = ttk.Combobox(
                camera_frame,
                textvariable=self.camera_var,
                values=available_cameras,
                state="readonly",
                width=30
            )
            camera_combobox.pack(side="left", padx=10)
            if available_cameras:
                camera_combobox.set(available_cameras[0])

            # Вариант 2: Видеофайл
            self.file_path = tk.StringVar()
            file_frame = ttk.Frame(options_frame)
            file_frame.pack(fill="x", pady=5)

            tk.Radiobutton(
                file_frame,
                text="📁 Видеофайл",
                variable=self.source_var,
                value="file"
            ).pack(side="left")

            file_entry = tk.Entry(file_frame, textvariable=self.file_path, width=30)
            file_entry.pack(side="left", padx=5)

            tk.Button(
                file_frame,
                text="Обзор...",
                command=self.browse_file
            ).pack(side="left")

            # Вариант 3: IP-камера
            self.ip_url = tk.StringVar(value="rtsp://")
            ip_frame = ttk.Frame(options_frame)
            ip_frame.pack(fill="x", pady=5)

            tk.Radiobutton(
                ip_frame,
                text="🌐 IP-камера",
                variable=self.source_var,
                value="ip"
            ).pack(side="left")

            ip_entry = tk.Entry(ip_frame, textvariable=self.ip_url, width=40)
            ip_entry.pack(side="left", padx=5)

            # Кнопки управления
            button_frame = ttk.Frame(self.root)
            button_frame.pack(pady=20)

            tk.Button(
                button_frame,
                text="✅ Запустить систему",
                command=self.launch_system,
                bg="green",
                fg="white",
                font=("Arial", 10, "bold"),
                padx=20
            ).pack(side="left", padx=10)

            tk.Button(
                button_frame,
                text="❌ Выход",
                command=self.close_app,
                bg="red",
                fg="white",
                padx=20
            ).pack(side="left", padx=10)

        except Exception as e:
            print(f"Ошибка настройки UI: {e}")
            # При ошибке UI устанавливаем значение по умолчанию
            self.selected_value = 0

    def detect_available_cameras(self):
        """Автоматическое обнаружение доступных камер"""
        cameras = []
        print("🔍 Поиск доступных камер...")

        try:
            # Проверяем первые 5 индексов камер (уменьшено для скорости)
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_name = f"Камера {i}"
                        # Пытаемся получить разрешение
                        try:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if width > 0 and height > 0:
                                camera_name += f" ({width}x{height})"
                        except:
                            pass

                        cameras.append(f"{camera_name} (ID: {i})")
                        print(f"✅ Найдена {camera_name}")
                    cap.release()

        except Exception as e:
            print(f"Ошибка поиска камер: {e}")

        if not cameras:
            cameras = ["Камера 0 (по умолчанию) (ID: 0)"]
            print("⚠️ Камеры не найдены, используется по умолчанию")

        return cameras

    def browse_file(self):
        """Выбор видеофайла через диалог"""
        try:
            file_types = [
                ("Видео файлы", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Все файлы", "*.*")
            ]

            filename = filedialog.askopenfilename(
                title="Выберите видеофайл",
                filetypes=file_types
            )

            if filename:
                self.file_path.set(filename)
                self.source_var.set("file")

        except Exception as e:
            print(f"Ошибка выбора файла: {e}")
            messagebox.showerror("Ошибка", f"Не удалось открыть диалог выбора файла: {e}")

    def validate_selection(self):
        """Проверка выбранного источника"""
        try:
            source_type = self.source_var.get()

            if source_type == "camera":
                return True

            elif source_type == "file":
                if not self.file_path.get():
                    messagebox.showerror("Ошибка", "Выберите видеофайл!")
                    return False
                if not os.path.exists(self.file_path.get()):
                    messagebox.showerror("Ошибка", "Файл не существует!")
                    return False
                return True

            elif source_type == "ip":
                url = self.ip_url.get()
                if not url.startswith(('rtsp://', 'http://', 'https://')):
                    messagebox.showerror("Ошибка", "Введите корректный URL (rtsp://, http:// или https://)")
                    return False
                return True

            else:
                messagebox.showerror("Ошибка", "Выберите источник видео!")
                return False

        except Exception as e:
            print(f"Ошибка валидации: {e}")
            return False

    def get_video_source(self):
        """Получение выбранного источника в формате для OpenCV"""
        try:
            source_type = self.source_var.get()

            if source_type == "camera":
                # Извлекаем ID камеры из строки
                camera_str = self.camera_var.get()
                if "(ID: " in camera_str:
                    camera_id = int(camera_str.split("(ID: ")[1].replace(")", ""))
                    return camera_id
                return 0  # По умолчанию

            elif source_type == "file":
                return self.file_path.get()

            elif source_type == "ip":
                return self.ip_url.get()

        except Exception as e:
            print(f"Ошибка получения источника: {e}")

        return 0  # fallback

    def launch_system(self):
        """Запуск основной системы"""
        try:
            if not self.validate_selection():
                return

            self.selected_value = self.get_video_source()
            self.save_settings()

            if hasattr(self, 'root'):
                self.root.destroy()  # Закрываем окно выбора

        except Exception as e:
            print(f"Ошибка запуска: {e}")
            self.selected_value = 0  # fallback

    def close_app(self):
        """Закрытие приложения"""
        try:
            if hasattr(self, 'root'):
                self.root.quit()
                self.root.destroy()
            sys.exit(0)
        except:
            sys.exit(0)

    def save_settings(self):
        """Сохранение настроек в файл"""
        try:
            settings = {
                'video_source': self.selected_value,
                'source_type': self.source_var.get() if hasattr(self, 'source_var') else 'camera',
                'last_used': str(Path(__file__).parent / "last_settings.json")
            }

            import json
            config_path = Path(__file__).parent / "video_settings.json"

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)

            print(f"💾 Настройки сохранены: {self.selected_value}")

        except Exception as e:
            print(f"⚠️ Не удалось сохранить настройки: {e}")

    def run(self):
        """Запуск интерфейса выбора"""
        try:
            if hasattr(self, 'root'):
                self.root.mainloop()
            return self.selected_value
        except Exception as e:
            print(f"Ошибка запуска GUI: {e}")
            return 0


def select_video_source():
    """Функция для выбора источника видео с обработкой ошибок"""
    try:
        selector = VideoSourceSelector()
        return selector.run()
    except Exception as e:
        print(f"Критическая ошибка селектора: {e}")
        print("Возврат к камере по умолчанию (ID: 0)")
        return 0


# Проверка доступности tkinter при импорте модуля
def check_tkinter_available():
    """Проверка доступности tkinter"""
    try:
        import tkinter
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    if not check_tkinter_available():
        print("❌ tkinter недоступен. Установите: pip install tk")
        print("Используется камера по умолчанию (ID: 0)")
        source = 0
    else:
        source = select_video_source()

    print(f"Выбран источник: {source}")