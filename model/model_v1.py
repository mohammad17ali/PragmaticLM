import os
import numpy as np
import transformers
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# Load model from HF transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

# Dataset
ds = load_dataset("msamogh/indirect-requests")

def prepare_dataset(ds):
  new_ds = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])

  new_ds = new_ds.remove_columns(
      [col for col in new_ds.column_names if col not in ["utterance", "situation"]]
      )

  data = {
      'input_text': ["Restructure Prompt: " + example['utterance'] for example in new_ds],
      'target_text': [example['situation'] for example in new_ds]
  }
  
  df = pd.DataFrame(data)
  
  train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
  
  train_dataset = Dataset.from_pandas(train_df)
  val_dataset = Dataset.from_pandas(val_df)
  
  return train_dataset, val_dataset

def preprocess_function(examples):
    inputs = tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(examples['target_text'], padding="max_length", truncation=True, max_length=128)
    
    # dealing w pad token id
    targets["input_ids"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in target] 
        for target in targets["input_ids"]
    ]
    
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset, val_dataset = prepare_dataset(ds)
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# trainning args
training_args = TrainingArguments(
    run_name = 'pragmaticLM',
    output_dir="./results",
    eval_strategy="epoch",
    report_to = 'none',
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=3,
    #load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim="adamw_torch",
    max_steps=1000,
)

class CustomTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            encoder_params = []
            decoder_params = []
            
            for name, param in self.model.named_parameters():
                if name.startswith('encoder'):
                    encoder_params.append(param)
                elif name.startswith('decoder') or name.startswith('lm_head'):
                    decoder_params.append(param)
            
            optimizer_grouped_parameters = [
                {'params': encoder_params, 'lr': 1e-5},
                {'params': decoder_params, 'lr': 3e-5}
            ]
            
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        return self.optimizer

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

#fine-tuning
trainer.train()

# save pragmaticLM_v1
model.save_pretrained("./prompt-restructuring-t5")
tokenizer.save_pretrained("./prompt-restructuring-t5")

# inference
def restructure_prompt(input_prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_text = f"Restructure Prompt: {input_prompt}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    
    model.to(device)

    output = model.generate(
        inputs.input_ids,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# example 1
test_prompt = "Could you help me find a medical professional who specializes in women's health issues?"
restructured = restructure_prompt(test_prompt)
print(f"Original: {test_prompt}")
print(f"Restructured: {restructured}")
