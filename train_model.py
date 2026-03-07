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
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.optim import AdamW

import random
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT

import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="facebook/mms-300m", choices=["utter-project/mHuBERT-147", "facebook/wav2vec2-xls-r-300m", "facebook/w2v-bert-2.0", "facebook/mms-300m"])

parser.add_argument("--enable_augmentation", action="store_true")
parser.add_argument("--augment_prob", type=float, default=0.8)

parser.add_argument("--enable_gain", action="store_true")
parser.add_argument("--enable_time_shift", action="store_true")
parser.add_argument("--enable_speed_perturb", action="store_true")
parser.add_argument("--enable_pitch_shift", action="store_true")
parser.add_argument("--enable_noise", action="store_true")

args, _ = parser.parse_known_args()

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
model_id = args.model_id


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
max_duration = 4 # in seconds

# %%
# Augmentation toggles
ENABLE_AUGMENTATION = False
AUGMENT_PROB = 0.3  # probability to apply augmentation per sample
SR_AUG = 16000

AUG_PITCH_MIN = -2.0  # semitones
AUG_PITCH_MAX = 2.0
AUG_SPEED_MIN = 0.9  # speed perturb rate
AUG_SPEED_MAX = 1.1
AUG_SHIFT_MS = 50  # max time shift (ms)
AUG_SNR_MIN = 10.0  # dB
AUG_SNR_MAX = 25.0

ENABLE_GAIN = False
ENABLE_TIME_SHIFT = False
ENABLE_SPEED_PERTURB = False
ENABLE_PITCH_SHIFT = False
ENABLE_NOISE = False

if args.enable_augmentation:
    ENABLE_AUGMENTATION = True

AUGMENT_PROB = float(args.augment_prob)

any_specific = any([
    args.enable_gain,
    args.enable_time_shift,
    args.enable_speed_perturb,
    args.enable_pitch_shift,
    args.enable_noise,
])

if ENABLE_AUGMENTATION and any_specific:
    ENABLE_GAIN = bool(args.enable_gain)
    ENABLE_TIME_SHIFT = bool(args.enable_time_shift)
    ENABLE_SPEED_PERTURB = bool(args.enable_speed_perturb)
    ENABLE_PITCH_SHIFT = bool(args.enable_pitch_shift)
    ENABLE_NOISE = bool(args.enable_noise)


# Peak-normalize (clamp) an audio waveform to prevent clipping after augmentations.
# Inputs:
#   x: 1D torch.Tensor of shape [T] containing the audio waveform samples.
#   peak: float, target maximum absolute amplitude after normalization.
# Output:
#   1D torch.Tensor with the same shape as x, scaled only if max(|x|) > peak.
def _clamp_audio(x, peak=0.99):
    m = x.abs().max().clamp(min=1e-6)
    return (x / m) * peak if m > peak else x


# Apply a random circular time shift to an audio waveform to reduce sensitivity to alignment.
# Inputs:
#   x: 1D torch.Tensor of shape [T] containing the audio waveform samples.
#   max_shift_ms: int/float, maximum time shift in milliseconds (both directions).
#   sr: int, sampling rate in Hz used to convert milliseconds to samples.
# Output:
#   1D torch.Tensor with the same shape as x, circularly shifted by a random amount.
def random_time_shift(x, max_shift_ms=AUG_SHIFT_MS, sr=SR_AUG):
    max_shift = int(sr * max_shift_ms / 1000)
    if max_shift <= 0:
        return x
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(x, shifts=shift)


# Apply a random gain (volume change) in decibels to simulate recording-level variation.
# Inputs:
#   x: 1D torch.Tensor of shape [T] containing the audio waveform samples.
#   min_db: float, minimum gain in dB.
#   max_db: float, maximum gain in dB.
# Output:
#   1D torch.Tensor with the same shape as x, scaled by a random gain and then clamped.
def random_gain(x, min_db=-6.0, max_db=6.0):
    db = random.uniform(min_db, max_db)
    g = 10 ** (db / 20)
    return _clamp_audio(x * g)


# Add Gaussian noise at a randomly sampled SNR to simulate background noise/channel noise.
# Inputs:
#   x: 1D torch.Tensor of shape [T] containing the audio waveform samples.
#   snr_db_min: float, minimum signal-to-noise ratio in dB.
#   snr_db_max: float, maximum signal-to-noise ratio in dB.
# Output:
#   1D torch.Tensor with the same shape as x, with added noise at the selected SNR and clamped.
def add_noise(x, snr_db_min=AUG_SNR_MIN, snr_db_max=AUG_SNR_MAX):
    snr_db = random.uniform(snr_db_min, snr_db_max)
    sig_power = x.pow(2).mean().clamp(min=1e-9)
    noise = torch.randn_like(x)
    noise_power = noise.pow(2).mean().clamp(min=1e-9)
    k = torch.sqrt(sig_power / (10 ** (snr_db / 10) * noise_power))
    y = x + k * noise
    return _clamp_audio(y)


# Perform speed perturbation by resampling to a random rate and back to the original sampling rate.
# This changes speaking rate/tempo slightly while keeping the sampling rate consistent for the model.
# Inputs:
#   x: 1D torch.Tensor of shape [T] containing the audio waveform samples.
#   sr: int, original sampling rate in Hz.
#   min_rate: float, minimum speed perturbation factor (<1.0 slows down).
#   max_rate: float, maximum speed perturbation factor (>1.0 speeds up).
# Output:
#   1D torch.Tensor containing the speed-perturbed waveform (length may change slightly).
def speed_perturb_resample(x, sr=SR_AUG, min_rate=AUG_SPEED_MIN, max_rate=AUG_SPEED_MAX):
    rate = random.uniform(min_rate, max_rate)
    if abs(rate - 1.0) < 1e-3:
        return x
    src_sr = int(sr * rate)
    y = AT.Resample(orig_freq=sr, new_freq=src_sr)(x.unsqueeze(0)).squeeze(0)
    y = AT.Resample(orig_freq=src_sr, new_freq=sr)(y.unsqueeze(0)).squeeze(0)
    return y


# Apply a random pitch shift (in semitones) to reduce dependence on speaker pitch characteristics.
# Inputs:
#   x: 1D torch.Tensor of shape [T] containing the audio waveform samples.
#   sr: int, sampling rate in Hz (required by the pitch shift operation).
#   min_semitones: float, minimum pitch shift in semitones (negative lowers pitch).
#   max_semitones: float, maximum pitch shift in semitones (positive raises pitch).
# Output:
#   1D torch.Tensor with the same shape as x, pitch-shifted and clamped.
def pitch_shift(x, sr=SR_AUG, min_semitones=AUG_PITCH_MIN, max_semitones=AUG_PITCH_MAX):
    n_steps = random.uniform(min_semitones, max_semitones)
    if abs(n_steps) < 1e-3:
        return x
    y = AF.pitch_shift(x.unsqueeze(0), sample_rate=sr, n_steps=n_steps).squeeze(0)
    return _clamp_audio(y)


# Apply a randomized sequence of waveform augmentations with overall probability AUGMENT_PROB.
# Intended to improve robustness by simulating variations in loudness, alignment, speaking rate,
# pitch, and background noise, while preserving the language label.
# Inputs:
#   x: 1D torch.Tensor of shape [T] containing the audio waveform samples.
# Output:
#   1D torch.Tensor containing the augmented waveform (or the original if not applied).
def apply_random_augmentation(x):
    if (not ENABLE_AUGMENTATION) or (random.random() > AUGMENT_PROB):
        return x
    if ENABLE_GAIN and random.random() < 0.1:
        x = random_gain(x)
    if ENABLE_TIME_SHIFT and random.random() < 0.2:
        x = random_time_shift(x)
    if ENABLE_SPEED_PERTURB and random.random() < 0.1:
        x = speed_perturb_resample(x)
    if ENABLE_PITCH_SHIFT and random.random() < 0.1:
        x = pitch_shift(x)
    if ENABLE_NOISE and random.random() < 0.2:
        x = add_noise(x)
    return x


# %%
# get the set of languages
LABELS = sorted(train_ds.unique('language'))

sorted_labels = sorted(l.upper() for l in LABELS)
print(f"Languages: {sorted_labels}")

str_to_int = {
    s: i for i, s in enumerate(LABELS)
}


# %%
# not covered in the documentation as this is a small adjustment to randomly
# take part of audio if it is larger than max duration so if recordings have similar beginning we "added" some diversity
def random_crop(audio, max_samples):
    # randomly choose part of audio if > max duration
    if len(audio) <= max_samples:
        return audio
    start = random.randint(0, len(audio) - max_samples)
    return audio[start:start + max_samples]

def preprocess_function(examples):
    max_samples = int(feature_extractor.sampling_rate * max_duration) # maximum allowed sample count

    audio_arrays = [
        random_crop(x["array"], max_samples)
        for x in examples["audio_filepath"]
    ]

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
keep_cols = ['speaker_id', 'language', 'audio_filepath']

# %% [markdown]
# ## encode the train and valid splits

# %%
train_ds_encoded = train_ds.map(
    preprocess_function, 
    remove_columns=[c for c in train_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=16,
    num_proc=8,
)

# %%
valid_ds_encoded = valid_ds.map(
    preprocess_function, 
    remove_columns=[c for c in valid_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=16,
    num_proc=8,
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
    config.hidden_dropout = 0.2           # Dropout for hidden states
    config.attention_dropout = 0.2        # Dropout in attention layers
    config.activation_dropout = 0.2       # Dropout after activation functions
    config.feat_proj_dropout = 0.2

# %%
class MMSForCentroid(nn.Module):
    """
        Custom Spoken Language Identification model with MMS backbone
    """
    def __init__(self, model_id, config):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(model_id, config=config)
        self.encoder.feature_extractor._freeze_parameters()
        self.embedding = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.LayerNorm(512)
        )
        self.centroids = nn.Parameter(torch.zeros(config.num_labels, 512))

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)

        hidden_state = outputs.last_hidden_state # [batch_size, seq_len, hidden_dim]

        if attention_mask is not None:
            batch_size, seq_len, _ = hidden_state.shape
            # shrink our input mask to match hiiden seq_len
            # average only over the actual speech frames (no padding)

            subsampled_mask = attention_mask[:, ::attention_mask.shape[1] // seq_len]
            # just if float math is off
            subsampled_mask = subsampled_mask[:, :seq_len].float().unsqueeze(-1)

            hidden_state = hidden_state * subsampled_mask
            pooled = hidden_state.sum(dim=1) / subsampled_mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_state.mean(dim=1)

        embedding = self.embedding(pooled)
        dists = torch.cdist(embedding, self.centroids, p=2).pow(2) #squared euclid dist
        logits = -dists

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}

    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        self.encoder.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

# %%
# spoken language ID (SLID) model
slid_model = MMSForCentroid(model_id, config)

# %%
class CentroidTrainer(Trainer):
    """
    Trainer for centroid-based classifier with learnable centroids
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # more like extract loss
        labels = inputs["labels"]
        outputs = model(input_values=inputs["input_values"], attention_mask=inputs["attention_mask"], labels=labels)
        loss = outputs["loss"]

        if return_outputs:
            return loss, outputs
        return loss

# %%
# create collator for padding
class AudioDataCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # prepare the batch dict in the format expected by the feature extractor
        # Use raw audio for on-the-fly augmentation + extraction if available
        use_raw_audio = (
                "audio_filepath" in features[0]
                and isinstance(features[0]["audio_filepath"], dict)
                and "array" in features[0]["audio_filepath"]
        )

        if use_raw_audio:
            audio_arrays = [f["audio_filepath"]["array"] for f in features]
            audio_tensors = [torch.tensor(a, dtype=torch.float32) for a in audio_arrays]

            audio_tensors = [apply_random_augmentation(x) for x in audio_tensors]
            audio_arrays = [x.numpy() for x in audio_tensors]

            inputs = self.feature_extractor(
                audio_arrays,
                sampling_rate=self.feature_extractor.sampling_rate,
                truncation=True,
                max_length=int(self.feature_extractor.sampling_rate * max_duration),
                return_attention_mask=True,
            )

            batch = {
                input_features_key: inputs[input_features_key],
                "attention_mask": inputs["attention_mask"]
            }
        else:
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
batch_size = 16
gradient_accumulation_steps = 2
num_train_epochs = 15
lr = 1e-5

# %%
wandb.init(project="Indic-SLID", name=f"SLID_{model_id}_{lr}_{current_time_str}")

if wandb.config is not None:
    if "batch_size" in wandb.config: batch_size = int(wandb.config["batch_size"])
    if "gradient_accumulation_steps" in wandb.config: gradient_accumulation_steps = int(
        wandb.config["gradient_accumulation_steps"])
    if "num_train_epochs" in wandb.config: num_train_epochs = int(wandb.config["num_train_epochs"])
    if "lr" in wandb.config: lr = float(wandb.config["lr"])
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
training_args = TrainingArguments(
    group_by_length=False,
    report_to="wandb",
    logging_steps=10,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=500,
    learning_rate=lr,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    weight_decay=float(getattr(wandb.config, "weight_decay", 0.05)) if wandb.config is not None else 0.05,
    warmup_ratio=float(getattr(wandb.config, "warmup_ratio", 0.05)) if wandb.config is not None else 0.05,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    max_grad_norm=1.0,
    push_to_hub=False,
    eval_accumulation_steps=15,
    warmup_steps=200,
    label_smoothing_factor=0.05
)

# %%

trainer = CentroidTrainer(
    model=slid_model,
    args=training_args,
    train_dataset=train_ds_encoded,
    eval_dataset=valid_ds_encoded,
    processing_class=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# %%
print("Train loop starting...")
trainer.train()

# %%
def plot_embeddings(model, dataset, str_to_int, num_samples=500):
    model_device = next(model.parameters()).device
    model.eval()
    embeddings = []
    labels = []

    # smaller set for visualization
    subset = dataset.select(range(min(num_samples, len(dataset))))
    dataloader = torch.utils.data.DataLoader(subset, batch_size=16, collate_fn=data_collator)

    with torch.no_grad():  # save memory and speed up inference
        for batch in dataloader:
            inputs = {k: v.to(model_device) for k, v in batch.items() if k != "labels"}
            output = model.encoder(**inputs)
            hidden = output.last_hidden_state
            if "attention_mask" in inputs:
                mask = inputs["attention_mask"]
                seq_len = hidden.shape[1]
                step = max(1, mask.shape[1] // seq_len)
                mask = mask[:, ::step][:, :seq_len].unsqueeze(-1)

                hidden = hidden * mask
                pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden.mean(dim=1)

            embedding = model.embedding(pooled)
            embedding = F.normalize(embedding, dim=-1)

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
    plt.title("t-SNE Visualization of Language Embeddings")
    plt.savefig("tsne_embeddings.png")
    wandb.log({"plots/tsne": wandb.Image("tsne_embeddings.png")})
    plt.show()


def plot_confusion_matrix(trainer, dataset, labels):
    output = trainer.predict(dataset)
    # for each input chooses the predicted language based on the highest score
    predictions = np.argmax(output.predictions, axis=-1)

    true_labels = np.array(dataset["label"])
    conf_matrix = confusion_matrix(true_labels, predictions)

    fig, ax = plt.subplots(figsize=(16, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)

    # cmap="Blues" -> darker blue means == samples in that cell
    # xticks_rotation='vertical' -> prevents long language names from overlapping
    # values_format='d' -> integers in cells
    disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical', values_format='d')
    plt.title("Confusion Matrix")
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