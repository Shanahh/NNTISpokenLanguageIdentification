# %%
import os
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
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from datasets import (
    load_dataset, 
    Audio
    # load_from_disk, 
    # DatasetDict, 
    # concatenate_datasets, 
)

# %%

from transformers import (
    AutoModel,
    AutoFeatureExtractor, 
    Wav2Vec2Config,
    AutoConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback
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
if hf_key:
    login(token=hf_key)

# %%
# login to WANDB
wandb_key = os.environ.get("WANDB_KEY")
if wandb_key:
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
LABELS = sorted(train_ds.unique('language'))

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
    batch_size=8,
    #num_proc=8,
)

# %%
valid_ds_encoded = valid_ds.map(
    preprocess_function, 
    remove_columns=[c for c in valid_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=8,
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

do_apply_dropout = True

# check if dropout is enabled
if do_apply_dropout:
    config.hidden_dropout = 0.1           # Dropout for hidden states
    config.attention_dropout = 0.1        # Dropout in attention layers
    config.activation_dropout = 0.1       # Dropout after activation functions
    config.feat_proj_dropout = 0.1

# %%
class MMSForCentroid(nn.Module):
    def __init__(self, model_id, config):
        super().__init__()
        self.mms = AutoModel.from_pretrained(model_id, config=config)
        # parameters that stay consistent across batches (calculate prototypes over all batches)
        self.prototypes = nn.Parameter(torch.randn(config.num_labels, config.hidden_size) * 0.01)

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.mms(input_values=input_values, attention_mask=attention_mask)
        # hidden_states shape: [batch, sequence_length, hidden_size] -> mean over time dimension
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # squared second norm of distance
        distances = torch.cdist(embeddings, self.prototypes, p=2).pow(2)
        return {"logits": -distances, "last_hidden_state": outputs.last_hidden_state}

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        self.mms.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

# %%
# spoken language ID (SLID) model
slid_model = MMSForCentroid(model_id, config)

# %%
class CentroidTrainer(Trainer):
    """
    Centroid-based Classification
    It computes the mean embedding for each class in the batch, then calculates the centroids and uses Euclidean distance for classification to the nearest one
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # distance calculation is in the model (forward)
        outputs = model(input_values=inputs.get("input_values"), attention_mask=inputs.get("attention_mask"))
        logits = outputs["logits"]

        # cross entropy for -distances
        loss = F.cross_entropy(logits, labels)

        if return_outputs:
            return loss, {"logits": logits}
        return loss
        
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
gradient_accumulation_steps = 8
num_train_epochs_stage1 = 5
num_train_epochs_stage2 = 20
lr_stage1 = 1e-5
lr_stage2 = 0.00001

# %%
wandb.init(project="Indic-SLID", name=f"SLID_{model_id}_{lr_stage2}_{current_time_str}")

if wandb.config is not None:
    if "batch_size" in wandb.config: batch_size = int(wandb.config["batch_size"])
    if "gradient_accumulation_steps" in wandb.config: gradient_accumulation_steps = int(
        wandb.config["gradient_accumulation_steps"])
    if "num_train_epochs_stage1" in wandb.config: num_train_epochs_stage1 = int(wandb.config["num_train_epochs_stage1"])
    if "num_train_epochs_stage2" in wandb.config: num_train_epochs_stage2 = int(wandb.config["num_train_epochs_stage2"])
    if "lr_stage1" in wandb.config: lr_stage1 = float(wandb.config["lr_stage1"])
    if "lr_stage2" in wandb.config: lr_stage2 = float(wandb.config["lr_stage2"])
    if "max_duration" in wandb.config: max_duration = int(wandb.config["max_duration"])
    if "warmup_ratio" in wandb.config: pass
    if "weight_decay" in wandb.config: pass

# %%
# load evaluation metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    out = accuracy_metric.compute(
        predictions=predictions,
        references=eval_pred.label_ids
    )
    out.update(
        f1_metric.compute(
            predictions=predictions,
            references=eval_pred.label_ids,
            average="macro"
        )
    )
    return out


# %%
def _get_encoder(m):
    for a in ("wav2vec2", "hubert", "w2v2_bert", "encoder", "model"):
        if hasattr(m, a):
            return getattr(m, a)
    return None


def freeze_encoder(m):
    enc = _get_encoder(m)
    if enc is None:
        return
    for p in enc.parameters():
        p.requires_grad = False


def unfreeze_encoder(m):
    enc = _get_encoder(m)
    if enc is None:
        return
    for p in enc.parameters():
        p.requires_grad = True


# %%
freeze_encoder(slid_model)

# %%
training_args_stage1 = TrainingArguments(
    group_by_length=False,
    report_to="wandb",
    logging_steps=15,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    learning_rate=lr_stage1,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs_stage1,
    weight_decay=float(getattr(wandb.config, "weight_decay", 0.01)) if wandb.config is not None else 0.01,
    warmup_ratio=float(getattr(wandb.config, "warmup_ratio", 0.05)) if wandb.config is not None else 0.05,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    max_grad_norm=1.0,
    push_to_hub=False,
    eval_accumulation_steps=15
)

# %%
trainer_stage1 = CentroidTrainer(
    slid_model,
    training_args_stage1,
    train_dataset=train_ds_encoded,
    eval_dataset=valid_ds_encoded,
    processing_class=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# %%
print("Train loop starting (stage 1: head-only)")
trainer_stage1.train()

# %%
unfreeze_encoder(slid_model)

# %%
training_args_stage2 = TrainingArguments(
    report_to="wandb",
    logging_steps=10,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    learning_rate=lr_stage2,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs_stage2,
    weight_decay=float(getattr(wandb.config, "weight_decay", 0.01)) if wandb.config is not None else 0.01,
    warmup_ratio=float(getattr(wandb.config, "warmup_ratio", 0.05)) if wandb.config is not None else 0.05,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    max_grad_norm=1.0,
    push_to_hub=False,
    eval_accumulation_steps=15
)

training_args_stage2.output_dir = "./slid_stage2_results"

# %%
trainer = CentroidTrainer(
    slid_model,
    training_args_stage2,
    train_dataset=train_ds_encoded,
    eval_dataset=valid_ds_encoded,
    processing_class=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

# %%
print("Train loop starting (stage 2: full finetune)")
trainer.train(resume_from_checkpoint=False)

# %%
def plot_embeddings(model, dataset, str_to_int, num_samples=500):
    model_device = next(model.parameters()).device
    model.eval()
    embeddings = []
    labels = []

    # smaller set for visualization
    subset = dataset.select(range(min(num_samples, len(dataset))))
    dataloader = torch.utils.data.DataLoader(subset, batch_size=8, collate_fn=data_collator)

    with torch.no_grad(): # save memory and speed up inference
        for batch in dataloader:
            inputs = {k: v.to(model_device) for k, v in batch.items() if k != "labels"}
            # decide on encoder
            if hasattr(model, "mms"):
                backbone = model.mms
            elif hasattr(model, "wav2vec2"):
                backbone = model.wav2vec2
            else:
                backbone = model
            output = backbone(**inputs) # forward pass
            # hidden_states shape: [batch, sequence_length, hidden_size] -> mean over time dimension
            embedding = output.last_hidden_state.mean(dim=1)

            embeddings.append(embedding.cpu())
            labels.append(batch["labels"])
    # combine batches in large arrays
    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    # 2D
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings) #fits into higher-dim and transforms into lower-dim

    plt.figure(figsize=(12, 8))
    for i, lang in enumerate(str_to_int.keys()):
        mask = labels == i
        plt.scatter(reduced[mask, 0], reduced[mask, 1], label=lang, alpha=0.6)

    plt.legend()
    plt.title("t-SNE Visualization of Language Embeddings (Centroid-based)")
    plt.savefig("tsne_embeddings.png")
    wandb.log({"plots/tsne": wandb.Image("tsne_embeddings.png")})
    plt.show()


def plot_confusion_matrix(trainer, dataset, labels):
    output = trainer.predict(dataset)
    # for each input chooses the predicted language based on the highest score
    predictions = np.argmax(output.predictions, axis=-1)

    true_labels = np.array(dataset["label"])
    conf_matrix = confusion_matrix(true_labels, predictions)

    fig, ax = plt.subplots(figsize=(16, 16))  # Large size for 22 labels
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)

    # cmap="Blues" -> darker blue means == samples in that cell
    # xticks_rotation='vertical' -> prevents long language names from overlapping
    # values_format='d' -> integers in cells
    disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical', values_format='d')
    plt.title("Confusion Matrix: Task 2")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    wandb.log({"plots/confusion_matrix": wandb.Image("confusion_matrix.png")})
    plt.show()

# %%
# push model to hub
# slid_model.push_to_hub(
#     "your-hf-account/indic-language-identification"
# )

# %%
print("Final evaluation starting...")
trainer.evaluate()
plot_embeddings(slid_model, valid_ds_encoded, str_to_int)
plot_confusion_matrix(trainer, valid_ds_encoded, LABELS)
# save model to disk
save_dir = "./indic-SLID/inprogress"
slid_model.save_pretrained(save_dir)