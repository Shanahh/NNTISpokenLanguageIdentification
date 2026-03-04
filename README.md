# NNTISpokenLanguageIdentification

## Python version
Python 3.11 (latest 3.11.x)

## Environment setup

### venv + pip
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```
or alternatively use the docker image provided by the Tutors.

## Credentials
Set these environment variables before running:
```bash
export HF_KEY="your_hf_token"
export WANDB_KEY="your_wandb_api_key"
```

## Running training

Reproduce baseline:
```bash
python repro_baseline.py
```

Default run (uses `facebook/mms-300m`):
```bash
python train_model.py
```

Select a pretrained model:
```bash
python train_model.py --model_id facebook/wav2vec2-xls-r-300m
```

## Data augmentation (waveform)

Waveform augmentation is OFF by default. It is only applied when `--enable_augmentation` is passed.

Enable waveform augmentation:
```bash
python train_model.py --enable_augmentation
```

Control probability of applying augmentation per sample:
```bash
python train_model.py --enable_augmentation --augment_prob 0.8
```

### Enable specific augmentations
If you pass any of the specific `--enable_*` flags, only those enabled transforms will be used (the others are disabled). Example: enable augmentation but only use pitch shift + noise:
```bash
python train_model.py --enable_augmentation --enable_pitch_shift --enable_noise
```

Available flags:
- `--enable_gain`
- `--enable_time_shift`
- `--enable_speed_perturb`
- `--enable_pitch_shift`
- `--enable_noise`

## Outputs
- Logs are sent to Weights & Biases (W&B).
- The trained model is saved to `./indic-SLID/inprogress`.
