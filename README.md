# 🍎 Fruit Classification using CNN (From Scratch)

This project implements a **Convolutional Neural Network (CNN) from scratch** to classify images of fruits into multiple categories using TensorFlow and Keras.

The goal of this project is **learning and understanding CNN fundamentals**, not achieving state-of-the-art performance.

Classes:
1. Apple
2. Avocado
3. Banana
4. Cherry
5. Kiwi
6. Mango
7. Orange
8. Pineapple
9. Strawberries
10. Watermelon
---

## 📌 Project Highlights
- CNN built **from scratch (no transfer learning)**
- Image classification using `flow_from_directory`
- Data augmentation to reduce overfitting
- Separate notebooks for exploration and clean training pipeline
- Keras Tuner used for basic hyperparameter tuning

---

## 📂 Project Structure
fruit-classification-cnn-from-scratch/
│
├── notebooks/
│ ├── 01_exploring_learning_classification.ipynb
│ └── 02_clean_training_pipeline.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
---

## 🧠 Dataset
- Taken from kaggle: https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class
- Dataset size is relatively small

⚠️ **Note:** Dataset is not included in this repository due to size constraints.

---

## 🏗️ Model Architecture
- Convolutional layers with ReLU activation
- MaxPooling layers
- Fully connected Dense layers
- Dropout used for regularization

Loss function: `categorical_crossentropy`  
Optimizer: `Adam`

---

## 📊 Results
| Metric | Value |
|------|------|
| Training Accuracy | ~62% |
| Validation Accuracy | ~66% |
| Hyperparameter tuning |~54% |

⚠️ Due to limited dataset size and training from scratch, the model may not generalize well to unseen images. 
    Even hyperparameter tuning has its limitations and cannot compensate for lack of data.

---

## ❗ Limitations
- Small dataset
- No transfer learning
- Limited generalization on real-world images

---

## 🚀 Future Improvements
- Use **transfer learning (MobileNet, ResNet)**
- Increase dataset size
- Apply class weighting
- Improve augmentation strategy
- Add test-time evaluation and confusion matrix

---

## 🛠️ How to Run
```bash
pip install -r requirements.txt
Open jupyter notebook
