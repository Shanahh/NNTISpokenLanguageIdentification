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
    AutoModelForAudioClassification, 
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
max_duration = 7 # in seconds

# %%
# Augmentation toggles
ENABLE_AUGMENTATION = False
AUGMENT_PROB = 0.8  # probability to apply augmentation per sample
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
    if ENABLE_GAIN and random.random() < 0.5:
        x = random_gain(x)
    if ENABLE_TIME_SHIFT and random.random() < 0.7:
        x = random_time_shift(x)
    if ENABLE_SPEED_PERTURB and random.random() < 0.7:
        x = speed_perturb_resample(x)
    if ENABLE_PITCH_SHIFT and random.random() < 0.5:
        x = pitch_shift(x)
    if ENABLE_NOISE and random.random() < 0.5:
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
keep_cols = ['speaker_id', 'language', 'audio_filepath']

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
batch_size = 8
gradient_accumulation_steps = 2
num_train_epochs = 20
lr = 0.00002

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
    logging_steps=25,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=lr,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    weight_decay=float(getattr(wandb.config, "weight_decay", 0.01)) if wandb.config is not None else 0.01,
    warmup_ratio=float(getattr(wandb.config, "warmup_ratio", 0.05)) if wandb.config is not None else 0.05,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    max_grad_norm=1.0,
    push_to_hub=False,
)

# %%
trainer = Trainer(
    slid_model,
    training_args,
    train_dataset=train_ds_encoded,
    eval_dataset=valid_ds_encoded,
    tokenizer=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# %%
print("Train loop starting...")
trainer.train()

# %%
print("Final evaluation starting...")
trainer.evaluate()

# save model to disk
save_dir = "./indic-SLID/inprogress"
slid_model.save_pretrained(save_dir)