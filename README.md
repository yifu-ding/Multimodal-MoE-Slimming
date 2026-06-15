# Multimodal MoE Slimming

This repository provides a lightweight framework for slimming multimodal Mixture-of-Experts models. The goal of this project is to reduce redundant expert capacity in multimodal MoE models by considering how different experts respond to different modalities. 

## Overview

The repository focuses on structural slimming for multimodal MoE models. 
In multimodal models, experts may not contribute uniformly to all input types. Some experts are more useful for visual information, while others are more useful for textual information. This repository explores this non-uniform expert behavior and uses it to identify and remove redundant expert dimensions while preserving the model's multimodal capability. Instead of treating all experts and modalities equally, it analyzes expert behavior under multimodal inputs and allocates compression decisions accordingly.

The general workflow includes:

1. **Collect multimodal calibration signals**
   Run the model on representative multimodal samples and collect expert-level and channel-level statistics.

2. **Analyze modality-dependent expert behavior**
   Estimate how different experts respond to visual and textual information.

3. **Generate slimming decisions**
   Determine which parts of the expert modules are less critical and can be removed.

4. **Build a compact model**
   Apply structural slimming to obtain a smaller multimodal MoE checkpoint.

5. **Evaluate the compact model**
   Test the slimmed model on downstream multimodal understanding tasks.

## Repository Structure

```text
.
├── scripts/              # Running scripts for calibration, slimming, and evaluation
├── src/                  # Core implementation
├── configs/              # Example configuration files
├── data/                 # Calibration data utilities
├── eval/                 # Evaluation utilities
├── examples/             # Example commands
└── README.md
```

The exact directory names may vary depending on the released version.

## Installation

```bash
git clone https://github.com/yifu-ding/Multimodal-MoE-Slimming.git
cd Multimodal-MoE-Slimming

conda create -n mm-moe-slim python=3.10
conda activate mm-moe-slim

pip install -r requirements.txt
```

Recommended dependencies include:

```bash
pip install torch transformers accelerate datasets
```

Additional dependencies may be required depending on the target model and evaluation benchmarks.

## Usage

### 1. Prepare calibration data

Prepare a small set of representative multimodal samples. The calibration data should cover both visual and textual inputs so that the expert behavior can be estimated under realistic multimodal usage.

### 2. Run calibration

```bash
python scripts/calibrate.py \
    --model_path /path/to/model \
    --data_path /path/to/calibration/data \
    --output_path outputs/calibration
```

### 3. Generate slimming configuration

```bash
python scripts/generate_slimming_config.py \
    --calibration_path outputs/calibration \
    --pruning_ratio 0.5 \
    --output_path outputs/slimming_config.json
```

### 4. Apply slimming

```bash
python scripts/apply_slimming.py \
    --model_path /path/to/model \
    --config_path outputs/slimming_config.json \
    --output_path outputs/slimmed_model
```

### 5. Evaluate the slimmed model

```bash
python scripts/evaluate.py \
    --model_path outputs/slimmed_model \
    --benchmark /path/to/benchmark
```

## Notes

* This repository is intended for research and experimental use.
* The best compression ratio depends on the target model, calibration data, and downstream tasks.
* For stable results, the calibration set should include diverse multimodal examples.
