# Support Vector Machine (SVM) Implementation with Comprehensive Metrics

This directory contains a comprehensive implementation of the Support Vector Machine algorithm using the Tic-Tac-Toe Endgame Dataset for binary classification, featuring extensive evaluation metrics and analysis techniques.

## Table of Contents
1. [Algorithm Overview](#algorithm-overview)
2. [How SVM Works](#how-svm-works)
3. [Dataset Description](#dataset-description)
4. [Implementation Details](#implementation-details)
5. [Usage](#usage)
6. [Results](#results)
7. [Dependencies](#dependencies)

## Algorithm Overview

**Support Vector Machine (SVM)** is a powerful supervised learning algorithm used for both classification and regression tasks. SVM is particularly effective for high-dimensional data and is known for its ability to find optimal decision boundaries with maximum margin separation between classes.

### Key Characteristics:
- **Margin-based**: Finds the hyperplane that maximizes the margin between classes
- **Kernel-based**: Can handle non-linear relationships through kernel functions
- **Memory efficient**: Uses only support vectors for predictions
- **Versatile**: Effective for both linear and non-linear classification
- **Robust**: Less prone to overfitting, especially in high-dimensional spaces

## How SVM Works

### The Algorithm Principles:

1. **Find the optimal hyperplane** that separates classes with maximum margin
2. **Support vectors** are the data points closest to the decision boundary
3. **Kernel trick** allows SVM to work in higher-dimensional spaces
4. **Regularization** parameter C controls the trade-off between margin maximization and classification errors

### Mathematical Foundation:

For a binary classification problem, SVM finds the hyperplane:
$$f(x) = w^T x + b$$

Where the optimization objective is:
$$\min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$

Subject to:
$$y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

### Kernel Functions:
Common kernel functions used in SVM:

- **Linear Kernel**:
  $$K(x_i, x_j) = x_i^T x_j$$

- **Polynomial Kernel**:
  $$K(x_i, x_j) = (x_i^T x_j + c)^d$$

- **Radial Basis Function (RBF) Kernel** (used in this implementation):
  $$K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$$

- **Sigmoid Kernel**:
  $$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + c)$$

### Key Parameters:
- **C (Regularization)**: Controls trade-off between smooth decision boundary and classifying training points correctly
  - High C: Lower bias, higher variance (more complex model)
  - Low C: Higher bias, lower variance (simpler model)
- **gamma (for RBF kernel)**: Controls the influence of a single training example
  - High gamma: Points very close to decision boundary have influence (more complex)
  - Low gamma: Points far from decision boundary have influence (simpler)

### Advantages:
- Effective in high-dimensional spaces
- Memory efficient (uses subset of training points)
- Versatile with different kernel functions
- Works well when number of features > number of samples
- Strong theoretical foundation

### Disadvantages:
- Sensitive to feature scaling
- No probabilistic output (requires calibration)
- Choice of kernel and parameters can be challenging
- Can be slow on large datasets
- Sensitive to outliers

## Dataset Description

This implementation uses the **Tic-Tac-Toe Endgame Dataset** from the UCI Machine Learning Repository.

### Dataset Characteristics:
- **Total samples**: 958 legal tic-tac-toe endgame board configurations
- **Features**: 9 categorical variables representing each square of the tic-tac-toe board
- **Target**: Binary classification (Win for X vs Loss/Draw for X)
  - Class "positive": X wins (has three-in-a-row) → mapped to 1
  - Class "negative": X loses or draws → mapped to 0

### Feature Descriptions:
| Feature                   | Description                                  | Original Values    | Mapped Values |
|---------------------------|----------------------------------------------|--------------------|---------------|
| `top-left-square`         | Top-left position of tic-tac-toe board       | {x, o, b}          | {1, -1, 0}    |
| `top-middle-square`       | Top-center position of tic-tac-toe board     | {x, o, b}          | {1, -1, 0}    |
| `top-right-square`        | Top-right position of tic-tac-toe board      | {x, o, b}          | {1, -1, 0}    |
| `middle-left-square`      | Middle-left position of tic-tac-toe board    | {x, o, b}          | {1, -1, 0}    |
| `middle-middle-square`    | Center position of tic-tac-toe board         | {x, o, b}          | {1, -1, 0}    |
| `middle-right-square`     | Middle-right position of tic-tac-toe board   | {x, o, b}          | {1, -1, 0}    |
| `bottom-left-square`      | Bottom-left position of tic-tac-toe board    | {x, o, b}          | {1, -1, 0}    |
| `bottom-middle-square`    | Bottom-center position of tic-tac-toe board  | {x, o, b}          | {1, -1, 0}    |
| `bottom-right-square`     | Bottom-right position of tic-tac-toe board   | {x, o, b}          | {1, -1, 0}    |

### Feature Value Mapping:
- **x**: Player X has taken this square → **1**
- **b**: Blank square (not taken) → **0**
- **o**: Player O has taken this square → **-1**

### Class Distribution:
- **Positive cases** (X wins): ~65.3% of samples (626 samples)
- **Negative cases** (X loses/draws): ~34.7% of samples (332 samples)

### Dataset Context:
This dataset encodes all possible board configurations at the end of tic-tac-toe games, where "X" is assumed to have played first. The target concept is "win for X", which is true when X has one of the 8 possible ways to create a "three-in-a-row" (horizontal, vertical, or diagonal).

## Implementation Details

### Data Preprocessing:
1. **Data Loading**: Load the dataset from CSV format
2. **Column Naming**: Apply descriptive names to board positions
3. **Target Encoding**: Map classes ('positive' → 1, 'negative' → 0)
4. **Feature Encoding**: Map board states (x→1, o→-1, b→0)
5. **Data Shuffling**: Randomize order for better train/validation/test splits
6. **Feature Scaling**: StandardScaler normalization (mean = 0, std = 1)
7. **Oversampling**: RandomOverSampler on training data to handle class imbalance

### Dataset Splits:
- **Training**: 70% (with oversampling to balance classes)
- **Validation**: 15%
- **Test**: 15%

### Model Configuration:
- **Algorithm**: scikit-learn's SVC (Support Vector Classifier)
- **Kernel**: RBF (Radial Basis Function)
- **Regularization (C)**: 1.0 (default)
- **Gamma**: 'scale' (1/(n_features * X.var()))
- **Probability**: True (enables probability estimates)
- **Random State**: 42 (for reproducibility)

### Key Implementation Features:
- Comprehensive data visualization with distribution plots and correlation matrix
- Modular scaling and oversampling function
- Detailed classification reports for validation and test sets
- Individual cells for each metric calculation (15+ different metrics)
- ROC and Precision-Recall curve visualizations
- Cross-validation analysis with confidence intervals (5-fold stratified)
- Error analysis and prediction confidence evaluation
- Learning curve analysis for data sufficiency assessment
- Support vector analysis capabilities

## Usage

### Prerequisites:
Make sure you have the required dependencies installed (see [Dependencies](#dependencies) section).

### Running the Implementation:

1. **Navigate to the SVM directory:**
   ```bash
   cd src/algorithms/support_vector_machine/
   ```

2. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook support_vector_machine.ipynb
   ```

3. **Execute the cells sequentially** to:
   - Install and import required libraries
   - Load and preprocess the tic-tac-toe endgame dataset
   - Visualize feature distributions and correlations
   - Split and scale the data with oversampling
   - Train SVM model with RBF kernel
   - Calculate comprehensive performance metrics
   - Generate ROC and Precision-Recall curves
   - Perform cross-validation analysis
   - Conduct error analysis and confidence evaluation
   - Analyze learning curves for model assessment

### Code Structure:
```
support_vector_machine.ipynb
├── 01. Library Installation
├── 02. Library Imports (with comprehensive metrics)
├── 03. Data Loading and Preprocessing
│   ├── Dataset explanation and context
│   ├── Data loading and column naming
│   ├── Target and feature encoding
│   └── Data exploration
├── 04. Data Visualization
│   ├── Feature distribution plots by class
│   └── Correlation matrix heatmap
├── 05. Dataset Splitting and Scaling
│   ├── Train/validation/test splits (70/15/15)
│   ├── Feature scaling with StandardScaler
│   └── Random oversampling for class balance
├── 06. Support Vector Machine Implementation and Evaluation
│   ├── SVM model training (RBF kernel, C = 1.0, gamma = 'scale')
│   ├── Predictions on validation and test sets
│   ├── Classification reports and confusion matrices
│   ├── Individual metric calculations:
│   │   ├── Basic metrics (accuracy, precision, recall, F1)
│   │   ├── Advanced metrics (balanced accuracy, MCC, Cohen's kappa)
│   │   ├── Probability-based metrics (ROC AUC, PR AUC, log loss)
│   │   ├── ROC curve visualizations
│   │   ├── Precision-Recall curve visualizations
│   │   ├── Cross-validation analysis (5-fold stratified)
│   │   ├── Error analysis and confidence evaluation
│   │   └── Learning curve analysis
└── 07. Summary and Interpretation of Additional Metrics
```

## Results

The implementation provides a comprehensive evaluation of SVM performance using multiple metrics and analysis techniques:

### Basic Performance Metrics:
- **Accuracy**: Overall classification accuracy on validation and test sets
- **Precision**: Class-specific and weighted precision scores for win/loss prediction
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
- **Feature Distribution Plots**: Understanding class separability in feature space

### Cross-Validation Analysis:
- **5-Fold Stratified Cross-Validation** with confidence intervals for:
  - Accuracy, Precision, Recall, F1-Score, ROC AUC scores
  - Mean performance ± 2 standard deviations
  - Individual fold performance tracking

### Error Analysis:
- **Misclassification Patterns**: Analysis by true class (Win vs Loss/Draw)
- **Confidence Analysis**: Average confidence of correct vs incorrect predictions
- **Error Rates**: Detailed breakdown of classification errors

### SVM-Specific Analysis:
- **Kernel Performance**: RBF kernel effectiveness for non-linear decision boundaries
- **Support Vector Analysis**: Understanding which training points influence the model
- **Decision Boundary Quality**: Maximum margin separation visualization capabilities
- **Probability Calibration**: Quality of probability estimates from SVM

### Expected Performance:
- SVM achieves excellent performance on this dataset due to clear logical patterns
- RBF kernel effectively captures the non-linear decision boundaries in tic-tac-toe rules
- Feature scaling is crucial for optimal SVM performance
- Oversampling effectively addresses class imbalance
- Cross-validation confirms robust and reliable performance
- The model successfully learns the logical rules of tic-tac-toe winning conditions

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
- **Support Vector Machines**: Understanding margin-based classification with comprehensive evaluation
- **Kernel methods**: RBF kernel for handling non-linear decision boundaries
- **Data preprocessing**: Feature scaling and handling class imbalance with oversampling
- **Model evaluation**: Multiple metrics beyond accuracy for robust assessment
- **Performance visualization**: ROC curves, PR curves, and learning curves
- **Cross-validation**: Reliable performance estimation with confidence intervals
- **Error analysis**: Understanding model failures and prediction confidence
- **Probability calibration**: Working with SVM probability estimates

The SVM algorithm combined with comprehensive evaluation provides valuable insights into:
- How kernel functions enable non-linear classification
- The importance of feature scaling for distance-based algorithms
- Different evaluation metrics for imbalanced classification problems
- The relationship between model complexity and generalization
- Support vector interpretation and model decision-making

**Why SVM Works Well for Tic-Tac-Toe:**
- The 9-dimensional feature space provides clear separability between winning and losing positions
- RBF kernel captures the complex logical relationships in tic-tac-toe rules
- The dataset size (958 samples) is well-suited for SVM training
- Feature scaling ensures equal importance of all board positions
- The problem's discrete nature aligns well with SVM's margin-based approach
