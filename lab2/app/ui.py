import tkinter as tk
from tkinter import messagebox
import threading

from .inference import Inference


class LoadingFrame(tk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        self.loading_label = tk.Label(self, text="Загружаем модель...")
        self.loading_label.pack(expand=True)


class MainFrame(tk.Frame):
    model_output: tk.Text
    prompt_input: tk.Entry
    generate_button: tk.Button

    def __init__(self, master: tk.Tk, inference: Inference):
        super().__init__(master)
        self.inference = inference
        self.create_widgets()

    def create_widgets(self):
        self.columnconfigure(0, weight=16)
        self.columnconfigure(1, weight=1)

        self.model_output = tk.Text(self, height=17, state=tk.DISABLED)
        self.model_output.grid(row=0, columnspan=2, sticky=tk.NSEW)

        self.prompt_input = tk.Entry(self)
        self.prompt_input.grid(row=1, column=0, sticky=tk.NSEW)

        self.generate_button = tk.Button(self, text="Отправить", command=self.start_generate_text)
        self.generate_button.grid(row=1, column=1, sticky=tk.NSEW)

        self.prompt_input.bind("<Return>", self.start_generate_text)
        self.prompt_input.focus()

    def start_generate_text(self, _: tk.Event = None) -> None:
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
        try:
            result = self.inference.generate_text(prompt)
            self.master.after(0, self.update_output, result)
        except Exception as e:
            self.master.after(0, self.show_error, str(e))

    def update_output(self, text: str) -> None:
        self.model_output.config(state=tk.NORMAL)
        self.model_output.delete("1.0", tk.END)
        self.model_output.insert("1.0", text)
        self.model_output.config(state=tk.DISABLED)
        self.generate_button.config(state=tk.NORMAL)

    def show_error(self, message: str) -> None:
        self.generate_button.config(state=tk.NORMAL)
        messagebox.showerror("Ошибка", f"Не удалось сгенерировать текст:\n{message}")


class App(tk.Tk):
    inference: Inference

    loading_frame: LoadingFrame
    main_frame: MainFrame

    def __init__(self):
        super().__init__()
        self.title("Лабораторная работа №2")
        self.geometry("600x385")
        self.resizable(False, False)

        self.loading_frame = LoadingFrame(self)
        self.loading_frame.pack(side="top", fill="both", expand=True)

        self.inference = Inference()

        threading.Thread(target=self.load_models).start()

    def load_models(self):
        try:
            self.inference.load_models()
            self.after(0, self.on_models_loaded)
        except Exception as e:
            self.after(0, self.on_loading_error, str(e))

    def on_models_loaded(self):
        self.loading_frame.destroy()
        self.main_frame = MainFrame(self, self.inference)
        self.main_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

    def on_loading_error(self, message: str):
        self.loading_frame.destroy()
        messagebox.showerror("Ошибка", f"Не удалось загрузить модель:\n{message}")
        self.destroy()
