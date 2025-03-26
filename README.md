# Brain Tumor Segmentation and Classification  

This repository contains the implementation of **Brain Tumor Segmentation and Classification**, combining deep learning techniques for medical image analysis. The classification model follows the methodology outlined in [this tutorial playlist](https://www.youtube.com/watch?v=7DhQHwkPJzI&list=PL5foUFuneQnratPPuucpVxWl4RlqueP1u&index=5), while the segmentation model is based on the research paper ["3D MRI Brain Tumor Segmentation using a U-Net Based Deep Learning Model"](https://arxiv.org/pdf/1810.11654), with development currently in progress.

---

## 📌 Project Overview  
This project aims to:  
1. **Classify Brain Tumors** from MRI images into distinct categories.  
2. **Segment Brain Tumors** using deep learning models, improving accuracy in medical diagnosis.  

### 🧠 **Classification**  
The classification model predicts whether an MRI scan contains a tumor and categorizes it into types such as **glioma, meningioma, and pituitary tumors**.  

### 🩺 **Segmentation**  
The segmentation model detects and outlines tumor regions from MRI scans using a deep learning approach, focusing on pixel-wise classification for medical image analysis.  

---

## 📁 Dataset  
For classification, we use the **Brain Tumor MRI Dataset** from Kaggle, which consists of T1-weighted MRI images.  
For segmentation, the **BraTS (Brain Tumor Segmentation) dataset** is used, containing labeled tumor regions for supervised learning.  

📌 **Dataset Sources:**  
- **Classification:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  
- **Segmentation:** [BraTS Dataset](https://www.med.upenn.edu/cbica/brats.html)  

---

## 🚀 Model Development  

### 🏥 **1. Brain Tumor Classification Model**  

#### **📌 Model Architecture**
- **Backbone:** Pretrained CNN (VGG16, ResNet50, EfficientNet)
- **Input Shape:** 224x224 grayscale MRI images  
- **Layers:**  
  - Convolutional layers with ReLU activation  
  - MaxPooling layers for downsampling  
  - Fully connected dense layers  
  - Softmax activation for multi-class classification  

#### **📌 Training Process**  
1. **Data Preprocessing:**  
   - Convert grayscale images to RGB  
   - Resize images to (224x224)  
   - Normalize pixel values  
   - Augmentation (rotation, flipping, zoom)  
2. **Model Training:**  
   - Train on labeled MRI images  
   - Use **Cross-Entropy Loss**  
   - Optimizer: **Adam**  
   - Learning Rate: **0.0001**  
3. **Evaluation:**  
   - Accuracy, Precision, Recall, F1-score  

#### **📌 Implementation**
- Built using **TensorFlow/Keras**  
- Transfer learning applied for feature extraction  

---

### 🎯 **2. Brain Tumor Segmentation Model** (Under Development)  

#### **📌 Model Architecture**
- **Base Model:** 3D U-Net  
- **Input Shape:** 3D MRI scans  
- **Layers:**  
  - Encoder (CNN layers for feature extraction)  
  - Bottleneck (Dense layers with dropout)  
  - Decoder (Transpose convolution for upsampling)  
  - Final output: Pixel-wise segmentation map  

#### **📌 Training Process**  
1. **Data Preprocessing:**  
   - Convert MRI scans into 3D tensor format  
   - Normalize pixel intensities  
   - Augment dataset (random rotations, flips)  
2. **Model Training:**  
   - Loss Function: **Dice Loss + Binary Cross-Entropy**  
   - Optimizer: **Adam**  
   - Metrics: **IoU (Intersection over Union), Dice Score**  
3. **Evaluation:**  
   - Compare predicted masks with ground truth  
   - Compute segmentation accuracy  

#### **📌 Implementation**  
- Built using **PyTorch/Keras**  
- 3D convolution for improved spatial feature extraction  

---

## 🛠️ Installation  

Clone this repository:  
```bash
git clone https://github.com/yourusername/brain-tumor-segmentation-classification.git
cd brain-tumor-segmentation-classification
