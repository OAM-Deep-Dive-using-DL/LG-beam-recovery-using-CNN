# Quick Start Guide

<cite>
**Referenced Files in This Document**
- [README.md](file://README.md)
- [requirements.txt](file://requirements.txt)
- [models/CNN Trials/requirements.txt](file://models/CNN Trials/requirements.txt)
- [models/CNN Trials/src/data_gen/generate_dataset.py](file://models/CNN Trials/src/data_gen/generate_dataset.py)
- [models/CNN Trials/src/training/train.py](file://models/CNN Trials/src/training/train.py)
- [models/CNN Trials/src/evaluation/evaluate.py](file://models/CNN Trials/src/evaluation/evaluate.py)
- [models/CNN Trials/src/utils/dataset.py](file://models/CNN Trials/src/utils/dataset.py)
- [models/CNN Trials/src/models/model.py](file://models/CNN Trials/src/models/model.py)
- [models/CNN Trials/src/models/resnet_cbam.py](file://models/CNN Trials/src/models/resnet_cbam.py)
- [models/CNN Trials/src/models/attention.py](file://models/CNN Trials/src/models/attention.py)
- [models/CNN Trials/src/utils/device_utils.py](file://models/CNN Trials/src/utils/device_utils.py)
- [models/CNN Trials/data/configs/config.json](file://models/CNN Trials/data/configs/config.json)
- [models/CNN Trials/outputs/reports/Throughput_Analysis.md](file://models/CNN Trials/outputs/reports/Throughput_Analysis.md)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [30-Second Demo](#30-second-demo)
4. [Expected Outputs and Visualizations](#expected-outputs-and-visualizations)
5. [Hardware Requirements and Performance](#hardware-requirements-and-performance)
6. [Troubleshooting](#troubleshooting)
7. [Optimization Tips](#optimization-tips)
8. [Next Steps](#next-steps)

## Introduction
This Quick Start Guide helps you install the project, generate sample data, train the CBAM-enhanced model, and evaluate results in under 30 seconds. It also covers hardware requirements, expected outputs, troubleshooting, and performance tips for rapid prototyping.

## Installation
Follow these steps to prepare your environment and install dependencies.

- Clone the repository and navigate to the project root.
- Install core dependencies using pip. The project requires scientific computing, deep learning, and visualization libraries.
- For Apple Silicon (M3), use the Apple Silicon-optimized requirements that enable MPS acceleration.

Key commands:
- Install dependencies for the main project.
- Install Apple Silicon-optimized dependencies for PyTorch with MPS support.

Notes:
- Ensure Python 3.8+ and compatible PyTorch 2.0+ are installed.
- On Apple Silicon, PyTorch will automatically use MPS when available.

**Section sources**
- [README.md](file://README.md#L77-L86)
- [requirements.txt](file://requirements.txt#L1-L11)
- [models/CNN Trials/requirements.txt](file://models/CNN Trials/requirements.txt#L1-L33)

## 30-Second Demo
Run the following commands to generate a small dataset, train the model, and evaluate results.

- Change to the CNN Trials directory.
- Generate a small demo dataset with a specified number of samples and a dataset name.
- Train the model for a small number of epochs using the CBAM-enhanced backbone.
- Evaluate the trained model on the same dataset to produce metrics and plots.

Command sequence:
- Generate sample dataset.
- Train the model with a small epoch count for quick iteration.
- Evaluate the model to compute SER/BER and produce visualizations.

What you will see:
- Progress bars during data generation and training.
- Validation loss and learning rate updates printed during training.
- Evaluation metrics and saved plots upon completion.

**Section sources**
- [README.md](file://README.md#L88-L100)
- [models/CNN Trials/src/data_gen/generate_dataset.py](file://models/CNN Trials/src/data_gen/generate_dataset.py#L165-L175)
- [models/CNN Trials/src/training/train.py](file://models/CNN Trials/src/training/train.py#L138-L149)
- [models/CNN Trials/src/evaluation/evaluate.py](file://models/CNN Trials/src/evaluation/evaluate.py#L306-L315)

## Expected Outputs and Visualizations
After running the evaluation, the following outputs are produced:

- Evaluation plots:
  - BER vs. turbulence strength curve.
  - Throughput vs. turbulence strength curve.
  - Combined BER and throughput plot.
  - Constellation diagram comparing true vs. recovered symbols.
- Metrics:
  - Overall Symbol Error Rate (SER) and Bit Error Rate (BER).
  - Per-Cn2 breakdown and diagnosis statistics (mean magnitude, phase bias, phase jitter).
- Numerical results:
  - A NumPy archive containing Cn2 values, BER, and throughput for downstream plotting.

Where to find them:
- Plots and numerical results are saved in the evaluation script’s working directory.

**Section sources**
- [models/CNN Trials/src/evaluation/evaluate.py](file://models/CNN Trials/src/evaluation/evaluate.py#L222-L304)
- [models/CNN Trials/src/evaluation/evaluate.py](file://models/CNN Trials/src/evaluation/evaluate.py#L284-L286)

## Hardware Requirements and Performance
Choose the best device for your system:

- NVIDIA GPU: CUDA is detected automatically. cuDNN benchmarking is enabled for performance.
- Apple Silicon (M1/M2/M3): MPS is preferred and used automatically when available.
- CPU: Fallback device if neither CUDA nor MPS is available.

Device selection logic:
- Prefer CUDA if available.
- Otherwise prefer MPS on Apple Silicon.
- Fall back to CPU otherwise.

Batch size and workers:
- Batch size is selected based on available memory (GPU or system RAM).
- Workers for data loading are tuned per device type.

System information:
- The device utilities module prints system and device details for debugging and optimization.

**Section sources**
- [models/CNN Trials/src/training/train.py](file://models/CNN Trials/src/training/train.py#L17-L18)
- [models/CNN Trials/src/evaluation/evaluate.py](file://models/CNN Trials/src/evaluation/evaluate.py#L81-L82)
- [models/CNN Trials/src/utils/device_utils.py](file://models/CNN Trials/src/utils/device_utils.py#L15-L46)
- [models/CNN Trials/src/utils/device_utils.py](file://models/CNN Trials/src/utils/device_utils.py#L106-L142)
- [models/CNN Trials/src/utils/device_utils.py](file://models/CNN Trials/src/utils/device_utils.py#L145-L167)

## Troubleshooting
Common issues and fixes:

- PyTorch device detection:
  - If CUDA is unavailable, the system falls back to MPS (Apple Silicon) or CPU. Verify device selection and adjust batch size accordingly.
- MPS on Apple Silicon:
  - Ensure PyTorch 2.0+ with MPS support is installed. The environment file specifies MPS-friendly versions.
- Insufficient memory:
  - Reduce batch size or disable multiprocessing workers. The device utilities provide recommended settings.
- Data loading:
  - The dataset loader loads the entire dataset into memory. Ensure sufficient RAM for the dataset size.
- Evaluation artifacts:
  - If evaluation fails to load the model checkpoint, confirm the best model file exists and matches the backbone name.

**Section sources**
- [models/CNN Trials/src/utils/device_utils.py](file://models/CNN Trials/src/utils/device_utils.py#L15-L46)
- [models/CNN Trials/src/utils/device_utils.py](file://models/CNN Trials/src/utils/device_utils.py#L106-L142)
- [models/CNN Trials/src/utils/dataset.py](file://models/CNN Trials/src/utils/dataset.py#L11-L22)
- [models/CNN Trials/src/evaluation/evaluate.py](file://models/CNN Trials/src/evaluation/evaluate.py#L88-L93)

## Optimization Tips
Speed up your experiments:

- Use a GPU (CUDA) for training and evaluation.
- On Apple Silicon, rely on MPS for acceleration.
- Reduce batch size if encountering memory pressure.
- Limit workers for data loading to reduce contention.
- For rapid iteration, reduce epochs and use a smaller dataset.
- Use the CBAM-enhanced backbone for improved convergence and resilience.

**Section sources**
- [models/CNN Trials/src/utils/device_utils.py](file://models/CNN Trials/src/utils/device_utils.py#L106-L142)
- [models/CNN Trials/src/utils/device_utils.py](file://models/CNN Trials/src/utils/device_utils.py#L145-L167)
- [README.md](file://README.md#L88-L100)

## Next Steps
After completing the quick start:

- Generate larger datasets for robust training.
- Train for more epochs with the full dataset.
- Compare architectures (vanilla ResNet-18 vs. ResNet-18 + CBAM).
- Explore throughput analysis and scalability to higher data rates.

**Section sources**
- [README.md](file://README.md#L229-L308)
- [models/CNN Trials/outputs/reports/Throughput_Analysis.md](file://models/CNN Trials/outputs/reports/Throughput_Analysis.md#L1-L55)