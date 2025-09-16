# k-Nearest Neighbors (kNN) Implementation with Comprehensive Metrics

This directory contains a comprehensive implementation of the k-Nearest Neighbors algorithm using the MAGIC Gamma Telescope Dataset for binary classification, featuring extensive evaluation metrics and analysis techniques.

## Table of Contents
1. [Algorithm Overview](#algorithm-overview)
2. [How kNN Works](#how-knn-works)
3. [Dataset Description](#dataset-description)
4. [Implementation Details](#implementation-details)
5. [Usage](#usage)
6. [Results](#results)
7. [Dependencies](#dependencies)

## Algorithm Overview

The **k-Nearest Neighbors (kNN)** is a simple, non-parametric, lazy learning algorithm used for both classification and regression tasks. It's called "lazy" because it doesn't build an explicit model during training; instead, it stores all training data and makes predictions based on the similarity of new instances to stored instances.

### Key Characteristics:
- **Non-parametric**: Makes no assumptions about the underlying data distribution
- **Instance-based**: Uses specific training instances to make predictions
- **Lazy learning**: No training phase; all computation happens during prediction
- **Memory-based**: Requires storing all training data

## How kNN Works

### The Algorithm Steps:

1. **Store the training dataset** (features and labels)
2. **For each new prediction:**
   - Calculate the distance between the new point and all training points
   - Find the k closest neighbors
   - For classification: Take the majority vote among these k neighbors
   - For regression: Take the average of the k neighbors' values

### Distance Metrics:
The most common distance metrics used in kNN are:

- **Euclidean Distance** (default in most implementations):
  $$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

- **Manhattan Distance**:
  $$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

- **Minkowski Distance** (generalization of Euclidean and Manhattan):
  $$d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$$

### Choosing k:
- **Small k** (e.g., k=1): More sensitive to noise, complex decision boundaries
- **Large k**: Smoother decision boundaries, less sensitive to noise, but may ignore local patterns
- **Rule of thumb**: Start with k = √n (where n is the number of training samples)
- **Cross-validation**: Use validation data to find optimal k

### Advantages:
- Simple to understand and implement
- No assumptions about data distribution
- Works well with small datasets
- Can be used for both classification and regression
- Naturally handles multi-class problems

### Disadvantages:
- Computationally expensive for large datasets (O(n) for each prediction)
- Sensitive to irrelevant features and feature scaling
- Memory intensive (stores all training data)
- Performance degrades with high-dimensional data (curse of dimensionality)

## Dataset Description

This implementation uses the **MAGIC Gamma Telescope Dataset** from the UCI Machine Learning Repository.

### Dataset Characteristics:
- **Total samples**: 19,020 observations
- **Features**: 10 continuous variables measuring telescope imaging parameters
- **Target**: Binary classification (Gamma particles vs Hadrons)
  - Class 1: Gamma particles (signal) - astrophysical interest
  - Class 0: Hadrons (background noise) - cosmic ray showers

### Feature Descriptions:
| Feature       | Description                                                       | Unit  |
|---------------|-------------------------------------------------------------------|-------|
| `fLength`     | Major axis of ellipse                                             | mm    |
| `fWidth`      | Minor axis of ellipse                                             | mm    |
| `fSize`       | 10-log of sum of content of all pixels                            | #phot |
| `fConc`       | Ratio of sum of two highest pixels over fSize                     | ratio |
| `fConc1`      | Ratio of highest pixel over fSize                                 | ratio |
| `fAsym`       | Distance from highest pixel to center, projected onto major axis  | mm    |
| `fM3Long`     | 3rd root of third moment along major axis                         | mm    |
| `fM3Trans`    | 3rd root of third moment along minor axis                         | mm    |
| `fAlpha`      | Angle of major axis with vector to origin                         | deg   |
| `fDist`       | Distance from origin to center of ellipse                         | mm    |

## Implementation Details

### Data Preprocessing:
1. **Data Loading**: Load the dataset from CSV format
2. **Column Naming**: Apply descriptive names to features
3. **Target Encoding**: Map classes ('g' → 1, 'h' → 0)
4. **Data Shuffling**: Randomize order for better train/validation/test splits
5. **Feature Scaling**: StandardScaler normalization (mean = 0, std = 1)
6. **Oversampling**: RandomOverSampler on training data to handle class imbalance

### Dataset Splits:
- **Training**: 70% (with oversampling)
- **Validation**: 15%
- **Test**: 15%

### Model Configuration:
- **Algorithm**: scikit-learn's KNeighborsClassifier
- **Distance Metric**: Euclidean distance (default)
- **k values tested**: 1, 3, 5 neighbors (basic evaluation)
- **k optimization range**: 1-20 neighbors (systematic analysis)
- **Cross-validation**: 5-fold stratified cross-validation
- **Performance metrics**: 15+ different evaluation metrics

### Key Implementation Features:
- Comprehensive data visualization with histograms and correlation matrix
- Modular scaling and oversampling function
- Multiple k-value evaluation with systematic optimization
- Individual cells for each metric calculation
- Detailed classification reports for validation and test sets
- ROC and Precision-Recall curve visualizations
- Cross-validation analysis with confidence intervals
- Error analysis and prediction confidence evaluation
- Learning curve analysis for data sufficiency assessment
- Bias-variance trade-off visualization

## Usage

### Prerequisites:
Make sure you have the required dependencies installed (see [Dependencies](#dependencies) section).

### Running the Implementation:

1. **Navigate to the kNN directory:**
   ```bash
   cd src/algorithms/knn/
   ```

2. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook knn.ipynb
   ```

3. **Execute the cells sequentially** to:
   - Install and import required libraries (including advanced metrics)
   - Load and preprocess the dataset
   - Visualize feature distributions and correlations
   - Split and scale the data with oversampling
   - Train kNN models with different k values
   - Calculate comprehensive performance metrics
   - Generate ROC and Precision-Recall curves
   - Perform cross-validation analysis
   - Conduct error analysis and confidence evaluation
   - Analyze learning curves and k-value optimization
   - Visualize bias-variance trade-offs

### Code Structure:
```
knn.ipynb / METRICS_knn.ipynb
├── 01. Library Installation
├── 02. Library Imports (with additional metrics)
├── 03. Data Loading and Preprocessing
├── 04. Data Visualization
├── 05. Dataset Splitting and Scaling
├── 06. kNN Implementation and Evaluation
│   ├── Basic kNN with k=1,3,5
│   ├── Optimal k using rule of thumb
│   ├── Classification reports and confusion matrices
│   ├── Individual metric calculations (accuracy, precision, recall, F1)
│   ├── Advanced metrics (balanced accuracy, MCC, Cohen's kappa)
│   ├── Probability-based metrics (ROC AUC, PR AUC, log loss)
│   ├── ROC and Precision-Recall curve visualizations
│   ├── Cross-validation analysis (5-fold for multiple metrics)
│   ├── Error analysis and confidence evaluation
│   ├── Learning curve analysis
│   ├── K-value optimization (k=1 to k=20)
│   └── K-value analysis visualization
└── 07. Summary and Interpretation of Additional Metrics
```

## Results

The implementation provides a comprehensive evaluation of kNN performance using multiple metrics and analysis techniques:

### Basic Performance Metrics:
- **Accuracy**: Overall classification accuracy on validation and test sets
- **Precision**: Class-specific and weighted precision scores
- **Recall**: Class-specific and weighted recall scores  
- **F1-Score**: Class-specific and weighted F1-scores
- **Balanced Accuracy**: Accounts for class imbalance better than standard accuracy

### Advanced Metrics:
- **Matthews Correlation Coefficient (MCC)**: Correlation between predictions and reality (-1 to +1)
- **Cohen's Kappa**: Inter-rater reliability accounting for chance agreement
- **ROC AUC**: Area under ROC curve measuring discriminative ability
- **Average Precision (PR AUC)**: Area under Precision-Recall curve, ideal for imbalanced datasets
- **Log Loss**: Quantifies prediction uncertainty using probability scores

### Visual Analysis:
- **ROC Curves**: True vs False Positive Rate trade-offs for validation and test sets
- **Precision-Recall Curves**: Precision vs Recall trade-offs, better for imbalanced data
- **Learning Curves**: Performance vs training set size to assess data sufficiency
- **K-Value Optimization Plots**: Systematic analysis of different k values (1-20)

### Cross-Validation Analysis:
- **5-Fold Stratified Cross-Validation** with confidence intervals for:
  - Accuracy, Precision, Recall, F1-Score, ROC AUC scores
  - Mean performance ± 2 standard deviations
  - Individual fold performance tracking

### Error Analysis:
- **Misclassification Patterns**: Analysis by true class (Hadron vs Gamma)
- **Confidence Analysis**: Average confidence of correct vs incorrect predictions
- **Error Rates**: Detailed breakdown of classification errors

### Hyperparameter Optimization:
- **K-Value Analysis**: Systematic evaluation of k=1 to k=20
- **Multiple Metric Optimization**: Finding optimal k for accuracy, F1-score, and ROC AUC
- **Bias-Variance Trade-off**: Visualization of training vs validation performance gap

### Expected Performance:
- kNN achieves excellent performance on this dataset due to clear feature separation
- Feature scaling is crucial for optimal results
- Oversampling effectively addresses class imbalance
- Optimal k values typically range from 3-7 for this dataset
- Cross-validation confirms robust and reliable performance

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
```

---

## Educational Notes

This implementation serves as an excellent introduction to:
- **Classification algorithms**: Understanding supervised learning with comprehensive evaluation
- **Data preprocessing**: Feature scaling and handling class imbalance with oversampling
- **Model evaluation**: Multiple metrics beyond accuracy for robust assessment
- **Performance visualization**: ROC curves, PR curves, and learning curves
- **Cross-validation**: Reliable performance estimation with confidence intervals
- **Hyperparameter tuning**: Systematic k-value optimization with multiple criteria
- **Error analysis**: Understanding model failures and prediction confidence
- **Bias-variance trade-off**: Visualizing the impact of model complexity
- **Metric selection**: Choosing appropriate metrics for different problem types

The kNN algorithm combined with comprehensive evaluation provides valuable insights into:
- How different metrics can reveal different aspects of model performance
- The importance of using multiple evaluation techniques
- How to systematically optimize hyperparameters
- The relationship between training data size and performance
- The trade-offs between model complexity and generalization
