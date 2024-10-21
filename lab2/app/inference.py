from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .settings import MODEL_NAME

import torch


class Inference:
    """
    Класс для генерации текста при помощи ruGPT3
    """

    loaded: bool = False

    max_new_tokens: int = 100
    no_repeat_ngram_size: int = 2
    repetition_penalty: float = 1.1
    temperature: float = 0.7

    model: GPT2LMHeadModel | None = None
    tokenizer: GPT2Tokenizer | None = None

    def load_models(self) -> None:
        """
        Загрузка моделей
        """

        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.loaded = True

    def generate_text(self, prompt: str) -> str:
        """
        Генерация текста

        :param prompt: Строка, на основе которой будет сгенерирован текст
        :return: Сгенерированный текст
        """

        if not self.loaded:
            raise RuntimeError("Модели не загружены.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
