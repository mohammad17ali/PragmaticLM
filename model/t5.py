import os
import numpy as np
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load model from HF transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

text = "translate English to German: The house is wonderful."

inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

# Generate an output to see the format
outputs = model.generate(inputs.input_ids)
decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
out_tokens = tokenizer.convert_ids_to_tokens(outputs[0])
print(outputs[0])
