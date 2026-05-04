# CDVLM: Cross-Domain Vision-Language Model

**Paper**: Multimodal Vision-Language Models for Cross-Domain Visual Understanding

## Architecture

```
Image → [ViT-B/16] → visual_features → [DIFA] → aligned_visual ─┐
                                                                  ├→ [Contrastive Loss]
Text  → [BERT]     → text_features    → [DIFA] → aligned_text   ─┘
                                                                  ├→ [Classification Head]
                                                                  └→ [Cross-Domain Contrastive]
```

## Loss Function

**L = L_clip + λ₁ · L_align + λ₂ · L_cd + L_cls**

- `L_clip`: CLIP-style vision-text contrastive loss
- `L_align`: Domain-invariant feature alignment (MMD + adversarial)
- `L_cd`: Cross-domain contrastive learning
- `L_cls`: Classification loss

## Datasets

- **Natural**: MS-COCO
- **Driving**: Cityscapes
- **Medical**: ChestX-ray14

## Project Structure

```
cdvlm/
├── config.py               # Configuration
├── models/
│   ├── __init__.py
│   ├── vision_encoder.py   # ViT-B/16 implementation
│   ├── text_encoder.py     # BERT text encoder
│   ├── difa_module.py      # Domain-Invariant Feature Alignment
│   ├── contrastive_loss.py # Cross-domain contrastive learning
│   └── cdvlm.py            # Main model (CDVLM)
├── data/
│   ├── __init__.py
│   └── dataset.py          # Datasets, transforms, samplers
├── utils/
│   ├── __init__.py
│   └── trainer.py          # Training loop, EMA, scheduler
├── train.py                # Main training script
├── evaluate.py             # Evaluation script
├── transfer_experiment.py  # Domain transfer experiments
├── baselines.py            # CLIP, DANN, MMD baselines
├── evaluator.py            # Evaluation utilities
├── visualization.py        # Plotting utilities
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train.py --epochs 30 --batch_size 256 --lr 1e-4

# Debug mode (fast test)
python train.py --debug --epochs 2

# Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pt --visualize

# Domain transfer experiments
python transfer_experiment.py --checkpoint checkpoints/best_model.pt
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Vision Encoder | ViT-B/16 |
| Text Encoder | BERT-base |
| Embedding Dim | 256 |
| Batch Size | 256 |
| Learning Rate | 1e-4 |
| Epochs | 30 |
| Optimizer | AdamW |
| Scheduler | Cosine |
| λ₁ (align) | 0.5 |
| λ₂ (contrastive) | 3.0 |
| Temperature | 0.07 |
| Warmup Steps | 1000 |
| Gradient Clip | 1.0 |
