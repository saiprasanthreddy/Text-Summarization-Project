from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from rouge_score import rouge_scorer, scoring
import torch
import pandas as pd
from tqdm import tqdm
from textSummarizer.entity import ModelEvaluationConfig



class ModelEvaluation:
    def __init__(self, config: "ModelEvaluationConfig"):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Split a dataset column into batches for model inference."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i:i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        column_text="article",
        column_summary="highlights",
        max_input_length=256,
        max_output_length=32,
        num_beams=1,
    ):
        article_batches = self.generate_batch_sized_chunks(dataset[column_text], batch_size)
        target_batches = self.generate_batch_sized_chunks(dataset[column_summary], batch_size)
        aggregator = scoring.BootstrapAggregator()

        model.eval()
        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches),
            total=(len(dataset[column_text]) + batch_size - 1) // batch_size,
        ):
            inputs = tokenizer(
                article_batch,
                max_length=max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                summaries = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    num_beams=num_beams,
                    max_length=max_output_length,
                    do_sample=False,
                )

            decoded_summaries = [
                tokenizer.decode(
                    summary,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).replace("<n>", " ")
                for summary in summaries
            ]

            for prediction, reference in zip(decoded_summaries, target_batch):
                aggregator.add_scores(metric.score(reference, prediction))

        return aggregator.aggregate()

    def evaluate(self, sample_count=1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = rouge_scorer.RougeScorer(rouge_names, use_stemmer=True)

        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt["test"][:sample_count],
            rouge_metric,
            model_pegasus,
            tokenizer,
            batch_size=4,
            column_text="dialogue",
            column_summary="summary",
        )

        rouge_dict = {name: score[name].mid.fmeasure for name in rouge_names}
        pd.DataFrame(rouge_dict, index=["pegasus"]).to_csv(
            self.config.metric_file_name,
            index=False,
        )
        return rouge_dict