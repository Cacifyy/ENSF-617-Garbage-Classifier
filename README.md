# Multimodal Garbage Classifier

## Overview
This project implements a multimodal deep learning model to classify garbage into four categories: **Black**, **Blue**, **Green**, and **TTR**. The model leverages both visual data (images) and textual context (derived from filenames).

## Architecture
- **Vision Branch:** Pre-trained **ResNet-18** (weights frozen) used as a feature extractor.
- **Text Branch:** **DistilBERT** encoder to process labels and descriptions extracted from filenames.
- **Fusion Head:** A multi-layer perceptron (MLP) that concatenates both 512D (image) and 768D (text) embeddings to make the final prediction.

## Performance
- **Final Validation Accuracy:** ~87.1%
- **Test Accuracy:** ~78.9%
- **Key Observation:** The model performs exceptionally well on Blue and Green bins but faces challenges distinguishing between TTR and Black bins.

## How to Run
1. Ensure the dataset is located in the correct file paths.
2. Run the training script.
3. Outputs (plots, best model checkpoint, and classification reports) will be saved in the `/outputs` directory.

## Requirements
- PyTorch
- Torchvision
- Transformers (HuggingFace)
- Scikit-learn
- Matplotlib
