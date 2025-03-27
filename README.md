# MLP Classification on the Breast Cancer Wisconsin Dataset

This repository demonstrates a complete workflow for binary classification using **Multilayer Perceptron (MLP)** on the Breast Cancer Wisconsin dataset. It covers data loading, preprocessing, model training, evaluation, and multiple visualization techniques to ensure both clarity and reproducibility.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features & Visualizations](#features--visualizations)  
3. [Prerequisites & Installation](#prerequisites--installation)  
4. [Usage](#usage)  
5. [Repository Structure](#repository-structure)  
6. [Detailed Explanation](#detailed-explanation)  
7. [Results](#results)  
8. [Accessibility](#accessibility)  
9. [License](#license)  
10. [Clone the Repository](#clone-the-repository)  
11. [Additional Resources](#additional-resources)

---

## Project Overview

The **Breast Cancer Wisconsin dataset** is a widely used benchmark in medical machine learning, containing 30 features extracted from digitized images of breast tumors along with a binary target indicating whether a tumor is **malignant** or **benign**. This project applies a **Multilayer Perceptron (MLP)** to classify tumors with high accuracy. By following the provided code and instructions, you will learn how to:

- Load and preprocess a real-world medical dataset.
- Train and fine-tune an MLP for optimal performance.
- Evaluate model performance using classification reports, confusion matrices, and ROC curves.
- Visualize the training process and decision boundaries through various plots.

---

## Features & Visualizations

1. **Data Preprocessing**
   - Standardizes all features to zero mean and unit variance using StandardScaler.

2. **MLP Model Training**
   - Implements an MLP with two hidden layers (50 and 25 neurons) to capture complex patterns.

3. **Performance Metrics**
   - Generates a detailed classification report (precision, recall, F1-score) and confusion matrix.

4. **Rich Visualizations**
   - **Confusion Matrix:** Visualized using a custom colormap.
   - **ROC Curve:** Illustrates the trade-off between true positive and false positive rates.
   - **Training Loss Curve:** Displays convergence over iterations.
   - **PCA Decision Boundary:** Projects high-dimensional data into 2D for intuitive visualization of class separation.

---

## Prerequisites & Installation

- **Python 3.7+**
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

## Data Loading
We load the Breast Cancer dataset using `load_breast_cancer()` from scikit-learn. This function returns a feature matrix with 30 attributes and a binary target indicating whether a tumor is malignant or benign.

## Preprocessing
- **StandardScaler** is applied to normalize all features to zero mean and unit variance.
- A **train–test split** (70% training, 30% testing) ensures unbiased evaluation of the model.

## Model Training
- An **MLP with two hidden layers** (50 and 25 neurons) is configured and trained using the Adam optimizer and ReLU activation.
- Training logs display the loss decreasing from approximately 0.75 to under 0.01 over multiple iterations, indicating smooth convergence.

## Evaluation
- A **classification report** details precision, recall, and F1-scores for both malignant and benign classes.
- A **confusion matrix** is generated to visualize the distribution of correct and incorrect predictions.
- An **ROC curve** is plotted, showing an AUC of 1.00, which signifies excellent discriminative performance.
- A **training loss curve** is visualized to show how the model converges during training.
- A **PCA-based decision boundary plot** is produced to illustrate how the model separates classes in a 2D projection of the high-dimensional data.

## Results
Typical results (exact numbers may vary slightly):
- **Accuracy:** ~97%
- **Precision, Recall, F1:** High scores for both malignant and benign classes.
- **Loss Curve:** Demonstrates a consistent decline in loss over iterations.
- **Visualizations:** Clear separation between classes in PCA space, minimal misclassifications, and an ROC AUC of 1.00.

## Accessibility
- **Color Schemes:** Uses colorblind-friendly palettes (e.g., PuBu, Spectral) with large font sizes.
- **Clear Labeling:** All plots include descriptive titles, axis labels, and legends.
- **Screen Reader Compatibility:** Text descriptions ensure that screen readers can interpret the figures.
- **Transcripts/Closed Captions:** (Recommended if multimedia components accompany the tutorial.)

## License
This project is distributed under the **MIT License**. You are free to use, modify, and distribute the code for personal or commercial purposes, provided the original license text is included.

## Clone the Repository
```bash
git clone https://github.com/Nara999/MLP-BreastCancer.git

