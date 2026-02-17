import os
# %%
from datetime import datetime
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Current time: {current_time_str}")
# %%
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from collections import Counter

import pandas as pd
import numpy as np
import torch
import wandb

from datasets import (
    load_dataset, 
    Audio
    # load_from_disk, 
    # DatasetDict, 
    # concatenate_datasets, 
)

# %%

from transformers import (
    AutoModelForAudioClassification, 
    AutoFeatureExtractor, 
    Wav2Vec2Config,
    AutoConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed
)

from huggingface_hub import login

# import Hugging Face libraries
import evaluate

# %%
# check if there GPU
print("Check if GPU available:")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.get_device_name(): {torch.cuda.get_device_name()}")



# %%
# login to Hugging Face
hf_key = os.environ.get("HF_KEY")
login(token=hf_key)

# %%
# login to WANDB
wandb_key = os.environ.get("WANDB_KEY")
wandb.login(key=wandb_key)

# %%
model_id = "facebook/mms-300m"
#model_id = "utter-project/mHuBERT-147"
#model_id = "facebook/wav2vec2-xls-r-300m"


# %%
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, 
    do_normalize=True,
    return_attention_mask=True,
)

# %%
dataset = load_dataset("badrex/nnti-dataset-full")

# %%
# check the strucutre of the dataset object 
print(f"dataset['train']: {dataset['train']}")

# %%
# check the strucutre of one training sample (before decoding)
print(f"dataset['train'][0]: {dataset['train'][0]}")

# %%
# shuffle the dataset
train_ds = dataset['train'].shuffle(seed=42)
valid_ds = dataset['validation'].shuffle(seed=42)

# resample to 16kHz
train_ds = train_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))
valid_ds = valid_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))


# %%
# based on the model typel, set input features key
if model_id == "facebook/w2v-bert-2.0":
    input_features_key = "input_features"
else:
    input_features_key = "input_values"

# %%
max_duration = 7 # in seconds

# %%
# get the set of languages
LABELS = train_ds.unique('language')

sorted_labels = sorted(l.upper() for l in LABELS) 
print(f"Languages: {sorted_labels}")

str_to_int = {
    s: i for i, s in enumerate(LABELS)
}


# %%
def preprocess_function(examples):

    audio_arrays = [x["array"] for x in examples["audio_filepath"]]

    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * max_duration), 
        return_attention_mask=True,
    )

    inputs["label"] = [str_to_int[x] for x in examples["language"]]
    
    # convert input_features_key contains numerical arrays
    inputs[input_features_key] = [
        np.array(x) for x in inputs[input_features_key]
    ]

    inputs["length"] = [len(f) for f in inputs[input_features_key]]

    return inputs

# %%
keep_cols = ['speaker_id', 'language']

# %% [markdown]
# ## encode the train and valid splits 

# %%
train_ds_encoded = train_ds.map(
    preprocess_function, 
    remove_columns=[c for c in train_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=32,
    #num_proc=8,
)

# %%
valid_ds_encoded = valid_ds.map(
    preprocess_function, 
    remove_columns=[c for c in valid_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=32,
    #num_proc=8,
)

# %%
int_to_str = {
    i: s for s, i in str_to_int.items()
}

num_labels = len(int_to_str)

# %%
config = AutoConfig.from_pretrained(model_id)

config.num_labels=num_labels
config.label2id=str_to_int
config.id2label=int_to_str

do_apply_dropout = False 

# check if dropout is enabled
if do_apply_dropout:
    config.hidden_dropout = 0.1           # Dropout for hidden states
    config.attention_dropout = 0.1        # Dropout in attention layers
    config.activation_dropout = 0.1       # Dropout after activation functions
    config.feat_proj_dropout = 0.1   

# %%
# spoken language ID (SLID) model
slid_model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    config=config,
)

# %%
# create collator for padding
class AudioDataCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # prepare the batch dict in the format expected by the feature extractor
        batch = {
            input_features_key: [f[input_features_key] for f in features],
            "attention_mask": [f["attention_mask"] for f in features]
        }
        
        # use the feature extractor's native padding
        batch = self.feature_extractor.pad(
            batch,
            padding=True,
            return_tensors="pt"
        )
        
        # add labels
        batch["labels"] = torch.tensor(
            [f["label"] for f in features], 
            dtype=torch.long
        )
        
        return batch


# %%
data_collator = AudioDataCollator(feature_extractor)

# %%
batch_size = 8
gradient_accumulation_steps = 2
num_train_epochs = 3
lr = 0.00001

# %%
wandb.init(project="Indic-SLID", name=f"SLID_{model_id}_{lr}_{current_time_str}")

# %%
training_args = TrainingArguments(
    group_by_length=False,
    #run_name='SLID_1', 
    report_to="wandb",  # enable logging to W&B
    logging_steps=1,  # how often to log to W&B
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,    
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps", 
    save_steps=100,
    learning_rate=lr,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,  # True if your metric should be maximized (like accuracy)
    save_total_limit=2,  # Keep only the best model
    fp16=True,
    push_to_hub=False,
)

# %%
# load evaluation metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(
        predictions=predictions, 
        references=eval_pred.label_ids
    )


# %%
trainer = Trainer(
    slid_model,
    training_args,
    train_dataset=train_ds_encoded,
    eval_dataset=valid_ds_encoded,
    processing_class=feature_extractor,
    data_collator=data_collator,  
    compute_metrics=compute_metrics,
)

# %%
print("Train loop starting...")
trainer.train()

# %%
# push model to hub 
# slid_model.push_to_hub(
#     "your-hf-account/indic-language-identification"
# )

# %%
print("Final evaluation starting...")
trainer.evaluate()


# save model to disk 
save_dir = "./indic-SLID/inprogress"
slid_model.save_pretrained(save_dir)
