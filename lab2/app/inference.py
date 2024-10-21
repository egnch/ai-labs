from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


class Inference:
    loaded: bool = False

    max_new_tokens: int = 100
    no_repeat_ngram_size: int = 2
    repetition_penalty: float = 1.1
    temperature: float = 0.7

    model: GPT2LMHeadModel | None = None
    tokenizer: GPT2Tokenizer | None = None

    def load_models(self) -> None:
        self.model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.loaded = True

    def generate_text(self, prompt: str) -> str:
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
                temperature=self.temperature,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
