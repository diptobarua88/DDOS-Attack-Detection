# 🛡️ DDoS Attack Detection Framework

This project presents an **end-to-end detection framework** for **Distributed Denial of Service (DDoS)** attacks using the **CICDDoS2019 dataset**.  
It integrates **Machine Learning**, **Multi-Layer Perceptrons (MLPs)**, **Deep Learning**, and a **Hybrid CNN+LSTM model** to effectively detect and interpret network-based DDoS attacks.

---

## 🚀 Project Overview

The goal of this project is to build a **robust, interpretable, and scalable detection system** capable of identifying different types of DDoS attacks in real-world network traffic.  
It combines **28 algorithms** (16 ML + 4 MLP + 4 DL + 1 Hybrid + 1 Transformer) with comprehensive **EDA**, **visualization**, **dimensionality reduction**, and **explainability analysis**.

---

## 🧩 Key Features

### 🔍 Data Preparation & Processing
- **Exploratory Data Analysis (EDA):**  
  Performed detailed statistical analysis and visual exploration.
- **Outlier Removal:**  
  Identified and removed abnormal data to enhance model reliability.
- **Data Balancing (SMOTE):**  
  Applied **Synthetic Minority Over-sampling Technique** to handle class imbalance.
- **Dimensionality Reduction Techniques:**  
  - **PCA (Principal Component Analysis)**  
  - **LDA (Linear Discriminant Analysis)**  
  - **t-SNE (t-Distributed Stochastic Neighbor Embedding)**  
  - **UMAP (Uniform Manifold Approximation and Projection)**  

### ⚙️ Model Categories

#### 🧠 1. Machine Learning Models (16)
- **LightGBM:** Gradient boosting with histogram-based fast learning, suitable for classification and ranking.  
- **XGBoost:** Optimized gradient boosting with regularization and pruning for high efficiency.  
- **Random Forest:** Ensemble of decision trees built from bootstrapped samples for improved accuracy.  
- **Gradient Boosting:** Sequential ensemble that minimizes residual errors of prior learners.  
- **Decision Tree:** Tree-structured classifier using decision nodes and leaves.  
- **Extra Trees:** Similar to Random Forest but adds extra randomness for better generalization.  
- **K-Nearest Neighbors (KNN):** Instance-based learning that classifies samples by nearest neighbors.  
- **Logistic Regression:** Linear classifier using the sigmoid function for binary outcomes.  
- **Support Vector Machine (SVM):** Maximizes the margin between classes via optimal hyperplane.  
- **AdaBoost:** Boosting method combining weak learners with adaptive sample weighting.  
- **Quadratic Discriminant Analysis (QDA):** Nonlinear generative classifier with class-specific covariance.  
- **Ridge Classifier:** Linear model using L2 regularization to prevent overfitting.  
- **Naive Bayes:** Probabilistic classifier assuming independence between features.  
- *(Additional tuned ensemble and boosting variants included in comparative analysis.)*

#### ⚡ 2. MLP Variants (4)
- **Fast MLP:** Optimized for fast training and inference via efficient activation and structure.  
- **Tiny / Mini MLP:** Compact architecture designed for low-resource inference environments.  
- **Quick MLP:** Lightweight model for real-time or low-latency applications.  
- **Ultra MLP:** Deeper or ultra-efficient architecture for specialized use cases and mobile inference.

#### 🧬 3. Deep Learning Models (4)
- **CNN:** 1D Convolutional layers + MaxPooling → Dense classifier for spatial feature extraction.  
- **LSTM:** Two LSTM layers + Dense classifier for temporal sequence modeling.  
- **GRU:** Two GRU layers + Dense classifier, optimized for faster training than LSTM.  
- **Transformer:** Multi-head attention encoder with positional embeddings for contextual learning.

#### 🔗 4. Hybrid Model
- **CNN + LSTM:** Combines convolutional feature extraction with LSTM-based temporal modeling to capture both spatial and sequential patterns in network traffic.

---

## 💡 Explainability
- **SHAP (SHapley Additive exPlanations):**  
  Used for feature-level interpretability to understand model decisions and feature contributions.

---

## 🧰 Tech Stack

- **Language:** Python  
- **Libraries:**  
  `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`,  
  `xgboost`, `lightgbm`, `tensorflow`, `keras`, `umap-learn`, `shap`

---

## 📂 Project Structure
DDoS_Detection_Framework/
│
├── data/
│ └── CICDDoS2019.csv
│
├── notebooks/
│ └── ddos_detection_framework.ipynb
│
├── models/
│ ├── ml_models/
│ ├── mlp_variants/
│ ├── dl_models/
│ └── hybrid_model/
│
├── visuals/
│ ├── correlation_heatmap.png
│ ├── pca_plot.png
│ ├── tsne_plot.png
│ ├── umap_visual.png
│ └── shap_summary.png
│
├── README.md
└── requirements.txt
