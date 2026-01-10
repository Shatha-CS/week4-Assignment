# C1M1-Assignment — Regression with Feature Engineering:

In this assignment, we built a regression model using a neural network to predict delivery time based on multiple input features. We started by exploring and visualizing the data to understand the relationships between variables. Then, we applied feature engineering by adding meaningful features such as rush hour and weekend indicators to improve the model’s performance. After preparing the data for training, we built and trained a multi-input neural network and monitored the loss to ensure the model was learning correctly. The main goal of this assignment was to understand how thoughtful feature engineering can improve regression model performance beyond using raw data alone.

-----------------

# C1M2-Assignment — EMNIST Letter Detective:

In this assignment, we built an image classification model to recognize handwritten letters using the EMNIST dataset. The work involved exploring and preparing the image data by correcting orientation, normalizing pixel values, and converting images into tensors suitable for training. A neural network was then trained to classify letters based on image inputs, and the model was evaluated by decoding a handwritten message to assess its performance. The main goal of this assignment was to understand the complete workflow of image classification, from raw image preprocessing to training and evaluating a neural network on a more complex dataset.

-----------------

# week4-Assignment3-kaggle-Dataset:

In this assignment, we applied a Convolutional Neural Network (CNN) to perform image classification using a Kaggle dataset. The work involved loading and preprocessing image data, preparing it for training, and designing a CNN architecture capable of extracting spatial features from images. The model was then trained and evaluated to assess its ability to correctly classify images. The main goal of this assignment was to understand how CNNs work for image-based tasks and how they outperform traditional neural networks when dealing with visual data.

-----------------

#  C1M4-Assignment — Programming Assignment — Overcoming Overfitting: Building a Robust CNN:

In this assignment, we focused on addressing overfitting while training a Convolutional Neural Network (CNN). The work involved improving model generalization by applying proper training and validation strategies and monitoring performance metrics such as loss and accuracy. Through this process, we ensured that the model learned meaningful patterns rather than memorizing the training data. The main goal of this assignment was to understand how to build a more robust CNN that performs well on unseen data.

-----------------

# Weekly Project: Image Classification with Transfer Learning:

## Problem
- Task: Multiclass image classification  
- Dataset: Intel Image Classification (Natural Scene Images)  
- Classes: buildings, forest, glacier, mountain, sea, street  
- Decision enabled: Automatically classify natural scene images into semantic categories  

---

## Quick Start

### 1. Data Ingestion
- Dataset source: Kaggle — *Intel Image Classification*
- Loaded using `torchvision.datasets.ImageFolder`

**Dataset structure:**
- `seg_train/` → training images  
- `seg_test/` → validation & inference images  

---

### 2. Data Preparation
- Image resizing to **224 × 224**
- Data augmentation (training only):
  - Random horizontal flip  
  - Random rotation  
  - Color jitter  
- Normalization using **ImageNet statistics**
- Efficient batch loading using `DataLoader`

---

### 3. Model Building
- Base model: **ResNet-18 (pre-trained on ImageNet)**
- Modification:
  - Final fully connected layer replaced to output **6 classes**
- Loss function: **Cross-Entropy Loss**

---

### 4. Training

Two transfer learning strategies were implemented:

#### 4.1 Feature Extractor
- Convolutional layers frozen
- Only final classifier trained
- Faster training, fewer trainable parameters

#### 4.2 Fine-tuning
- All layers unfrozen
- Entire network trained with a smaller learning rate
- Better adaptation to the dataset

---

## Data
- Number of classes: **6**
- Training images: **14,034**
- Validation images: **3,000**
- Images resized during preprocessing

---

## Splits
- Training set: `seg_train`
- Validation / inference set: `seg_test`
- Split provided directly by the dataset

---

## Metrics (Validation Set)

| Approach           | Best Validation Accuracy | Final Validation Accuracy |
|-------------------|--------------------------|---------------------------|
| Feature Extractor | 0.9030                   | 0.9030                    |
| Fine-tuning       | 0.9350                   | 0.9220                    |

---

## Inference
- Inference performed on unseen images from `seg_test`
- Same preprocessing pipeline used as validation
- Model predicts correct scene categories with high confidence

---

## Notes
- Fine-tuning achieved higher peak accuracy
- Feature extraction provided faster and more stable training
- ONNX deployment step was optional and not required

