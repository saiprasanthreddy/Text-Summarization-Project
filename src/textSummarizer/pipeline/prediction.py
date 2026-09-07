from textSummarizer.config.configuration import ConfigurationManager
from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@lru_cache(maxsize=1)
def _load_model_and_tokenizer(model_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.model, self.tokenizer, self.device = _load_model_and_tokenizer(
            str(self.config.model_path),
            str(self.config.tokenizer_path),
        )

    def predict(self,text):
        print("Dialogue:")
        print(text)

        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                length_penalty=0.8,
                num_beams=4,
                max_length=128,
            )

        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\nModel Summary:")
        print(output)

        return output