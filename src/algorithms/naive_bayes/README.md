# Naive Bayes Classifier Implementation

This directory contains a comprehensive implementation of the Naive Bayes classifier using the Skin Segmentation Dataset for binary classification of skin vs non-skin pixels.

## Table of Contents
1. [Algorithm Overview](#algorithm-overview)
2. [How Naive Bayes Works](#how-naive-bayes-works)
3. [Dataset Description](#dataset-description)
4. [Implementation Details](#implementation-details)
5. [Usage](#usage)
6. [Results](#results)
7. [Dependencies](#dependencies)

## Algorithm Overview

**Naive Bayes** is a family of probabilistic algorithms based on Bayes' theorem with the "naive" assumption of conditional independence between features. Despite this simplifying assumption, Naive Bayes classifiers often perform surprisingly well in practice and are particularly effective for text classification and other high-dimensional problems.

### Key Characteristics:
- **Probabilistic**: Based on Bayes' theorem and probability distributions
- **Fast**: Very efficient training and prediction
- **Simple**: Easy to implement and understand
- **Robust**: Handles irrelevant features well
- **Requires small training data**: Performs well even with limited data

## How Naive Bayes Works

### Bayes' Theorem:
The foundation of Naive Bayes is Bayes' theorem:

$$P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$$

Where:
- $P(y|X)$ = Posterior probability of class $y$ given features $X$
- $P(X|y)$ = Likelihood of features $X$ given class $y$
- $P(y)$ = Prior probability of class $y$
- $P(X)$ = Evidence (marginal probability of features $X$)

### The "Naive" Assumption:
Naive Bayes assumes that all features are conditionally independent given the class:

$$P(X|y) = P(x_1|y) \times P(x_2|y) \times \ldots \times P(x_n|y) = \prod_{i=1}^{n} P(x_i|y)$$

### Classification Process:
1. **Training Phase:**
   - Calculate prior probabilities $P(y)$ for each class
   - Estimate likelihood $P(x_i|y)$ for each feature given each class
   - For continuous features (Gaussian NB): estimate mean and variance for each class

2. **Prediction Phase:**
   - For a new instance, calculate posterior probability for each class
   - Assign the class with the highest posterior probability

### Gaussian Naive Bayes:
For continuous features, we assume a Gaussian (normal) distribution:

$$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_{y}^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_{y}^2}\right)$$

Where $\mu_y$ and $\sigma_y$ are the mean and standard deviation of feature $x_i$ for class $y$.

### Advantages:
- **Fast training and prediction**: Linear time complexity
- **Good baseline**: Often provides surprisingly good results
- **Handles multi-class naturally**: Extends easily to multiple classes
- **Not sensitive to irrelevant features**: The independence assumption helps
- **Good with small datasets**: Requires less training data than discriminative models
- **No hyperparameters to tune**: Simple and straightforward

### Disadvantages:
- **Independence assumption**: Rarely true in real-world data
- **Categorical features**: Requires smoothing for unseen feature combinations
- **Poor probability estimates**: Good for classification, not probability estimation
- **Correlated features**: Performance degrades when features are highly correlated

## Dataset Description

This implementation uses the **Skin Segmentation Dataset** for computer vision applications.

### Dataset Characteristics:
- **Total samples**: 245,057 observations
- **Features**: 3 continuous variables representing RGB color channels
- **Target**: Binary classification (Skin vs Non-skin pixels)
  - Class 1: Skin pixels (human skin)
  - Class 0: Non-skin pixels (background/objects)

### Feature Descriptions:
| Feature   | Description                   | Range | Type      |
|-----------|-------------------------------|-------|-----------|
| `B`       | Blue color channel intensity  | 0-255 | Integer   |
| `G`       | Green color channel intensity | 0-255 | Integer   |
| `R`       | Red color channel intensity   | 0-255 | Integer   |

### Class Distribution:
- **Non-skin pixels**: 194,198 samples (~79.2%)
- **Skin pixels**: 50,859 samples (~20.8%)

The dataset shows significant class imbalance, which reflects real-world scenarios where skin pixels typically represent a smaller portion of natural images.

### Applications:
This dataset is used for skin detection research in:
- **Face detection and recognition systems**
- **Hand gesture recognition**
- **Content filtering systems**
- **Human-computer interaction**
- **Medical image analysis**
- **Surveillance systems**

### Data Collection Context:
The RGB values were collected from face images of people of different ages, races, and genders under various lighting conditions and backgrounds, ensuring model generalization across diverse scenarios.

## Implementation Details

### Data Preprocessing:
1. **Data Loading**: Load the dataset from tab-separated format
2. **Column Naming**: Apply descriptive names to RGB features
3. **Target Encoding**: Map classes (1 → 1 for skin, 2 → 0 for non-skin)
4. **Data Shuffling**: Randomize order for better train/validation/test splits
5. **Feature Scaling**: StandardScaler normalization (important for visualization)
6. **Oversampling**: RandomOverSampler on training data to address class imbalance

### Dataset Splits:
- **Training**: 70% (with oversampling to balance classes)
- **Validation**: 15%
- **Test**: 15%

### Model Configuration:
- **Algorithm**: scikit-learn's GaussianNB (Gaussian Naive Bayes)
- **Assumption**: Features follow Gaussian distribution within each class
- **No hyperparameters**: Naive Bayes is parameter-free

### Key Implementation Features:
- Comprehensive data visualization with KDE plots by class
- Modular scaling and oversampling function
- Detailed classification reports for both validation and test sets
- Class distribution analysis

### Why Gaussian Naive Bayes for this Dataset:
- **Continuous features**: RGB values are continuous, making Gaussian NB appropriate
- **Independent color channels**: RGB channels can be reasonably assumed independent
- **Large dataset**: Sufficient data to estimate Gaussian parameters reliably
- **Binary classification**: Well-suited for skin vs non-skin classification

## Usage

### Prerequisites:
Make sure you have the required dependencies installed (see [Dependencies](#dependencies) section).

### Running the Implementation:

1. **Navigate to the Naive Bayes directory:**
   ```bash
   cd src/algorithms/naive_bayes/
   ```

2. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook naive_bayes.ipynb
   ```

3. **Execute the cells sequentially** to:
   - Install and import required libraries
   - Load and preprocess the skin segmentation dataset
   - Visualize feature distributions by class
   - Split and scale the data
   - Train the Gaussian Naive Bayes model
   - Evaluate model performance

### Code Structure:
```
naive_bayes.ipynb
├── 01. Library Installation and Imports
├── 02. Data Loading and Preprocessing
├── 03. Data Visualization and Exploration
├── 04. Dataset Splitting and Scaling
├── 05. Naive Bayes Implementation
├── 06. Comprehensive Model Evaluation
│   ├── Basic Classification Metrics (Accuracy, Precision, Recall, F1)
│   ├── Advanced Metrics (Balanced Accuracy, MCC, Cohen's Kappa)
│   ├── Probability-based Metrics (ROC-AUC, PR-AUC, Log Loss)
│   ├── Visual Analysis (ROC Curves, PR Curves)
│   ├── Cross-Validation with Confidence Intervals
│   ├── Error Analysis and Confidence Assessment
│   └── Learning Curve Analysis
└── 07. Summary and Interpretation
```

### Expected Workflow:
1. **Data Exploration**: Understand RGB distribution patterns for skin vs non-skin
2. **Preprocessing**: Handle class imbalance and scale features
3. **Model Training**: Fit Gaussian distributions for each class and feature
4. **Evaluation**: Assess performance on validation and test sets

## Results

The implementation provides comprehensive evaluation metrics to thoroughly assess model performance:

### Basic Performance Metrics:
- **Accuracy**: Overall classification accuracy on validation and test sets
- **Precision**: Proportion of positive predictions that are correct (per class)
- **Recall (Sensitivity)**: Proportion of actual positives correctly identified (per class)
- **F1-Score**: Harmonic mean of precision and recall (per class)
- **Balanced Accuracy**: Accuracy adjusted for class imbalance
- **Matthews Correlation Coefficient (MCC)**: Balanced measure considering all confusion matrix elements

### Advanced Evaluation:
- **Cohen's Kappa**: Inter-rater reliability accounting for chance agreement
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Average Precision (PR-AUC)**: Area under Precision-Recall curve (better for imbalanced data)
- **Log Loss**: Probabilistic loss function measuring prediction uncertainty

### Visual Analysis:
- **ROC Curves**: True Positive Rate vs False Positive Rate for validation and test sets
- **Precision-Recall Curves**: Precision vs Recall trade-off visualization
- **Feature Distribution Plots**: KDE plots showing RGB distributions by class
- **Learning Curves**: Training and validation scores vs training set size

### Statistical Validation:
- **Cross-Validation**: 5-fold stratified cross-validation with confidence intervals
- **Error Analysis**: Detailed analysis of misclassified samples and confidence levels
- **Performance Stability**: Multiple metrics to ensure robust evaluation

### Expected Performance:
- Naive Bayes typically performs well on this dataset due to distinct RGB patterns between skin and non-skin pixels
- The class imbalance is addressed through oversampling during training and appropriate metric selection
- Cross-validation provides confidence intervals for performance estimates
- Feature scaling helps with visualization but doesn't affect Naive Bayes performance significantly

### Performance Characteristics:
- **High Speed**: Very fast training and prediction
- **Good Baseline**: Provides strong baseline performance for comparison with other algorithms
- **Interpretable**: Easy to understand which RGB ranges correspond to skin pixels
- **Robust Evaluation**: Multiple complementary metrics provide comprehensive assessment

## Dependencies

```python
# Core libraries
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine Learning
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0

# Additional metrics and evaluation
# (included in scikit-learn):
# - sklearn.metrics for comprehensive evaluation
# - sklearn.model_selection for cross-validation
# - sklearn.preprocessing for data scaling
```

### Installation:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

Or install from the notebook:
```python
%pip install imbalanced-learn matplotlib numpy pandas scikit-learn seaborn
```

---

## Educational Notes

This implementation serves as an excellent introduction to:
- **Probabilistic algorithms**: Understanding Bayes' theorem in practice
- **Assumption impact**: How the independence assumption affects performance
- **Class imbalance**: Handling unbalanced datasets in real applications
- **Computer vision basics**: Understanding color-based image segmentation
- **Model interpretability**: How probabilistic models provide insights

### Comparison with k-NN:
- **Speed**: Naive Bayes is much faster than k-NN for prediction
- **Memory**: Uses much less memory (only stores parameters, not all training data)
- **Assumptions**: Makes stronger assumptions about data distribution
- **Interpretability**: Provides probabilistic output and feature importance insights

The Naive Bayes classifier is particularly valuable for understanding probabilistic machine learning and serves as an excellent stepping stone to more complex algorithms.