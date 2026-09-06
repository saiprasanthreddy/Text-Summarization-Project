from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch
import os



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        #loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        train_dataset = dataset_samsum_pt["train"].select(range(min(200, len(dataset_samsum_pt["train"]))))
        eval_dataset = dataset_samsum_pt["validation"].select(range(min(50, len(dataset_samsum_pt["validation"]))))

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            max_steps=50,
            warmup_steps=0,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="no",
            save_strategy="no",
            gradient_accumulation_steps=1,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        trainer = Trainer(model=model_pegasus, args=trainer_args,
                  processing_class=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset)
        
        trainer.train()

        ## Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))