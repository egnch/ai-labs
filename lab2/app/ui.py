import tkinter as tk
from tkinter import messagebox
import threading

from .inference import Inference


class LoadingFrame(tk.Frame):
    """
    Окно загрузки модели
    """

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self) -> None:
        """
        Создание виджетов
        """

        self.loading_label = tk.Label(self, text="Загружаем модель...")
        self.loading_label.pack(expand=True)


class SettingsFrame(tk.Toplevel):
    """
    Окно настроек генерации
    """

    def __init__(self, master: tk.Frame, inference: Inference):
        super().__init__(master)
        self.title("Настройки генерации")
        self.resizable(False, False)
        self.inference = inference
        self.create_widgets()

    def create_widgets(self) -> None:
        """
        Создание виджетов
        """

        tk.Label(self, text="Максимальное количество новых токенов:").grid(
            row=0, column=0, padx=10, pady=5, sticky=tk.W
        )
        self.max_new_tokens_var = tk.IntVar(value=self.inference.max_new_tokens)
        tk.Entry(self, textvariable=self.max_new_tokens_var).grid(
            row=0, column=1, padx=10, pady=5
        )

        tk.Label(self, text="No repeat n-gram size:").grid(
            row=1, column=0, padx=10, pady=5, sticky=tk.W
        )
        self.no_repeat_ngram_size_var = tk.IntVar(
            value=self.inference.no_repeat_ngram_size
        )
        tk.Entry(self, textvariable=self.no_repeat_ngram_size_var).grid(
            row=1, column=1, padx=10, pady=5
        )

        tk.Label(self, text="Repetition penalty:").grid(
            row=2, column=0, padx=10, pady=5, sticky=tk.W
        )
        self.repetition_penalty_var = tk.DoubleVar(
            value=self.inference.repetition_penalty
        )
        tk.Entry(self, textvariable=self.repetition_penalty_var).grid(
            row=2, column=1, padx=10, pady=5
        )

        tk.Label(self, text="Температура:").grid(
            row=3, column=0, padx=10, pady=5, sticky=tk.W
        )
        self.temperature_var = tk.DoubleVar(value=self.inference.temperature)
        tk.Entry(self, textvariable=self.temperature_var).grid(
            row=3, column=1, padx=10, pady=5
        )

        tk.Button(self, text="Применить", command=self.apply_settings).grid(
            row=4, column=0, columnspan=2, pady=10
        )

    def apply_settings(self) -> None:
        """
        Применение настроек
        """

        try:
            self.inference.max_new_tokens = self.max_new_tokens_var.get()
            self.inference.no_repeat_ngram_size = self.no_repeat_ngram_size_var.get()
            self.inference.repetition_penalty = self.repetition_penalty_var.get()
            self.inference.temperature = self.temperature_var.get()
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Неверный ввод: {e}")


class MainFrame(tk.Frame):
    """
    Окно генерации текста
    """

    model_output: tk.Text
    prompt_input: tk.Entry
    generate_button: tk.Button
    settings_button: tk.Button

    def __init__(self, master: tk.Tk, inference: Inference):
        super().__init__(master)
        self.inference = inference
        self.create_widgets()

    def create_widgets(self) -> None:
        """
        Создание виджетов
        """

        self.columnconfigure(0, weight=16)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self.model_output = tk.Text(self, height=17, state=tk.DISABLED)
        self.model_output.grid(row=0, columnspan=3, sticky=tk.NSEW, pady=(0, 5))

        self.prompt_input = tk.Entry(self)
        self.prompt_input.grid(row=1, column=0, sticky=tk.NSEW)

        self.generate_button = tk.Button(
            self, text="Отправить", command=self.start_generate_text
        )
        self.generate_button.grid(row=1, column=1, sticky=tk.NSEW, padx=(5, 0))

        self.settings_button = tk.Button(
            self, text="Настройки", command=self.open_settings
        )
        self.settings_button.grid(row=1, column=2, sticky=tk.NSEW, padx=(5, 0))

        self.prompt_input.bind("<Return>", self.start_generate_text)
        self.prompt_input.focus()

    def open_settings(self) -> None:
        """
        Открытие окна настроек
        """

        SettingsFrame(self, self.inference)

    def start_generate_text(self, _: tk.Event = None) -> None:
        """
        Запуск генерации текста
        """

        prompt = self.prompt_input.get()
        if not prompt.strip():
            return

        self.generate_button.config(state=tk.DISABLED)
        self.model_output.config(state=tk.NORMAL)
        self.model_output.delete("1.0", tk.END)
        self.model_output.insert("1.0", "Генерация...\n")
        self.model_output.config(state=tk.DISABLED)

        thread = threading.Thread(target=self.generate_text, args=(prompt,))
        thread.start()

    def generate_text(self, prompt: str) -> None:
        """
        Генерация текста
        
        :param prompt: Промпт
        """

        try:
            result = self.inference.generate_text(prompt)
            self.master.after(0, self.update_output, result)
        except Exception as e:
            self.master.after(0, self.show_error, str(e))

    def update_output(self, text: str) -> None:
        """
        Обновление сгенерированного текста

        :param text: Сгенерированный текст
        """

        self.model_output.config(state=tk.NORMAL)
        self.model_output.delete("1.0", tk.END)
        self.model_output.insert("1.0", text)
        self.model_output.config(state=tk.DISABLED)
        self.generate_button.config(state=tk.NORMAL)

    def show_error(self, message: str) -> None:
        """
        Показ ошибки если не удалось сгенерировать текст

        :param message: Сообщение об ошибке
        """

        self.generate_button.config(state=tk.NORMAL)
        messagebox.showerror("Ошибка", f"Не удалось сгенерировать текст:\n{message}")


class App(tk.Tk):
    inference: Inference

    loading_frame: LoadingFrame
    main_frame: MainFrame

    def __init__(self):
        super().__init__()
        self.title("Лабораторная работа №2")
        self.geometry("600x390")
        self.resizable(False, False)

        self.loading_frame = LoadingFrame(self)
        self.loading_frame.pack(side="top", fill="both", expand=True)

        self.inference = Inference()

        threading.Thread(target=self.load_models).start()

    def load_models(self) -> None:
        """
        Загрузка моделей
        """

        try:
            self.inference.load_models()
            self.after(0, self.on_models_loaded)
        except Exception as e:
            self.after(0, self.on_loading_error, str(e))

    def on_models_loaded(self) -> None:
        """
        Обработка завершения загрузки моделей
        """

        self.loading_frame.destroy()
        self.main_frame = MainFrame(self, self.inference)
        self.main_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

    def on_loading_error(self, message: str) -> None:
        """
        Обработка ошибки загрузки моделей

        :param message: Сообщение об ошибке
        """

        self.loading_frame.destroy()
        messagebox.showerror("Ошибка", f"Не удалось загрузить модель:\n{message}")
        self.destroy()
