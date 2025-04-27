# Adversarial Defense using JPEG Compression & ViT-Guided Attention

## Overview

This project implements and evaluates adversarial defense strategies based on JPEG compression, inspired by the SHIELD paper ([arXiv:1802.06816](https://arxiv.org/abs/1802.06816)). It includes:

1.  **JPEG Vaccination:** Using pre-trained ResNet50 models fine-tuned on datasets compressed at various JPEG quality levels (e.g., Q50, Q60, Q70, Q80, Q90).
2.  **Ensemble Defense:** Combining predictions from multiple vaccinated models.
3.  **Defense Mechanisms:**
    *   Standard full-image JPEG compression.
    *   Stochastic Local Quantization (SLQ) from the SHIELD paper.
    *   **Novelty:** ViT-Attention-Guided Compression.
4.  **Evaluation:** Testing defenses against common adversarial attacks (FGSM, I-FGSM, C&W) in gray-box and black-box settings using `torchattacks`.

## Novel Contribution: ViT-Attention-Guided Compression

This project introduces a novel defense mechanism that uses attention maps extracted from a pre-trained Vision Transformer (ViT, e.g., DINOv2) to guide the JPEG compression process.

*   **Concept:** Identify salient image regions using ViT attention.
*   **Mechanism:** Apply **higher JPEG quality** to important regions (high attention) and **lower JPEG quality** to less important regions (low attention).
*   **Goal:** Adaptively remove high-frequency perturbations, especially in less critical areas, while better preserving essential features in salient regions compared to uniform or purely random compression.

## Requirements

*   Python 3.x
*   PyTorch
*   Torchattacks
*   Transformers (Hugging Face)
*   OpenCV-Python (`opencv-python`)
*   NumPy
*   Matplotlib
*   PIL (Pillow)

(Recommended: Create a `requirements.txt` file)

## Setup

1.  **Clone the repository:**
    ```
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:**
    ```
    pip install -r requirements.txt # Or install manually
    ```
3.  **Dataset:** Prepare your test image dataset (e.g., a subset of ImageNet validation) in a directory structure like:
    ```
    test_folder/
        class_001/
            image1.jpg
            image2.jpg
            ...
        class_002/
            image3.jpg
            ...
        ...
    ```
    Update the `test_dir` variable in the scripts to point to `test_folder`.
4.  **Models:** Place your pre-trained vaccinated ResNet50 model files (e.g., `resnet50_q50.pth`, `resnet50_q60.pth`, etc.) in the project directory or update the paths in `load_vaccinated_models`.

## Usage

1.  Configure the main evaluation script(s) (e.g., `run_gray_box_experiment`, `run_black_box_experiment`, `run_attention_guided_defense_experiment`) with the correct dataset path and desired model qualities.
2.  Execute the script(s):
    ```
    Run the jupyter notebook
    ```
3.  The script will:
    *   Load models and data.
    *   Generate adversarial examples.
    *   Apply defense mechanisms (JPEG, SLQ, ViT-Attention).
    *   Evaluate the ensemble model's accuracy under different conditions.
    *   Print accuracy results and save comparison plots (e.g., `black_box_defense_results.png`).

