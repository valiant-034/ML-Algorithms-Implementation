# 🤖 Machine Learning Algorithms Implementation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Algorithms](https://img.shields.io/badge/Algorithms-From%20Scratch-success)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Welcome to the **ML Algorithms Implementation** repository! This project features ground-up Python implementations of fundamental Machine Learning algorithms, designed to provide a deep understanding of their inner workings without relying heavily on black-box libraries.

## 📖 Overview

This repository contains scripts and visualizations for three core machine learning tasks:
1.  **Clustering** (K-Means)
2.  **Probabilistic Classification** (Naive Bayes)
3.  **Tree-Based Classification** (Decision Trees)

## 📂 Repository Contents

### 1. K-Means Clustering (`q1_kmeans.py`)
*   **Implementation**: Custom K-Means algorithm analyzing centroids and cluster assignments.
*   **Visualizations**:
    *   `q1_partA_convergence.png`: Demonstrates algorithm convergence over iterations.
    *   `q1_partB_adversarial.png`: Showcases K-Means behavior on complex/adversarial data shapes.
    *   `q1_partC_initialization.png`: Compares different initialization strategies (e.g., Random vs. K-Means++).

### 2. Naive Bayes Classifier (`q2_naive_bayes.py`)
*   **Implementation**: A probabilistic classifier based on Bayes' theorem with independence assumptions.
*   **Visualizations**:
    *   `q2_partB_correlated.png`: Highlights performance impacts when features are highly correlated.
    *   `q2_partC_boundaries.png`: Visualizes the mathematical decision boundaries drawn by the classifier.

### 3. Decision Tree Classifier (`q3_decision_tree.py`)
*   **Implementation**: A recursive tree-building algorithm from scratch.
*   **Visualizations**:
    *   `q3_partB_gain_ratio.png`: Analyzes the effect of using Gain Ratio for splits.
    *   `q3_partC_overfitting.png`: Demonstrates tree overfitting with increasing depth.
    *   `q3_partD_greedy.png`: Explores the limitations of the greedy search approach in tree building.
    *   `q3_partE_noise.png`: Shows how noise in the dataset affects the final tree structure.

### Utilities
*   **`dataset_generator.py`**: A custom utility to generate synthetic datasets of varying complexities to test the robustness of the algorithms above.
*   **`ML_Assignment2_BSAI23034.pdf`**: The official assignment brief and requirements.

## 🚀 Key Takeaways

*   **From-Scratch Coding**: Enhances understanding of mathematical foundations (Distances, Probabilities, Information Gain).
*   **Visual Analytics**: Extensive plotting provides immediate, intuitive feedback on algorithmic behavior and pitfalls (like overfitting or poor initialization).

## 🛠️ Setup & Usage

To run the scripts locally:
1. Clone this repository: `git clone https://github.com/valiant-034/ML-Algorithms-Implementation.git`
2. Install required libraries: `pip install numpy matplotlib pandas scikit-learn`
3. Execute individual scripts, e.g., `python q1_kmeans.py`.

---
*Developed as part of the Machine Learning coursework.*
