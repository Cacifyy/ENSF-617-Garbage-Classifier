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

## Detail Training Result
Our model shows decent overall performance achieving 78.99% accuracy on the test set, along with validation accuracy stabilizing around 87%. The validation accuracy trend indicates that there is good convergence. We could potentially see if increasing the number of epochs would benefit the performance of the model, however the convergence may prove otherwise for our case. 
From the confusion matrix and classification report we see that the model best performs on Green and Blue waste, with recall being above 90%. This tells us that we can confidently rely on the model to identify those categories reliably. TTR has the highest precision (0.916), so when the model predicts TTR it is usually correct, but its lower recall (0.616) shows many TTR items are misclassified, often as Blue or Black. Additionally, Black waste has the weakest overall performance (F1 = 0.681), with moderate confusion across other categories. 
Overall, the model distinguishes recyclable and organic waste well but struggles more with separating Black and TTR classes.
