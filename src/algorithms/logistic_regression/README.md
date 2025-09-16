# Logistic Regression Implementation

This directory contains a comprehensive implementation of Logistic Regression using the Car Evaluation Dataset for multi-class classification of car acceptability.

## Table of Contents
1. [Algorithm Overview](#algorithm-overview)
2. [How Logistic Regression Works](#how-logistic-regression-works)
3. [Dataset Description](#dataset-description)
4. [Implementation Details](#implementation-details)
5. [Usage](#usage)
6. [Results](#results)
7. [Dependencies](#dependencies)

## Algorithm Overview

**Logistic Regression** is a linear classifier that uses the logistic (sigmoid) function to model the probability of class membership. Despite its name, it's a classification algorithm, not a regression algorithm. It's particularly effective for linearly separable data and provides probabilistic outputs that are easily interpretable.

### Key Characteristics:
- **Probabilistic**: Outputs class probabilities between 0 and 1
- **Linear decision boundary**: Creates linear separation between classes
- **No distributional assumptions**: Unlike Naive Bayes, doesn't assume feature distributions
- **Regularization support**: Can handle overfitting through L1/L2 regularization
- **Multi-class capable**: Extends naturally to multi-class problems

## How Logistic Regression Works

### The Logistic Function (Sigmoid):
The core of logistic regression is the logistic function that maps any real number to a value between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z$ is the linear combination of features:
$$z = \beta_0 + \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n = \mathbf{w}^T\mathbf{x}$$

### Binary Classification:
For binary classification, the probability of class 1 is:
$$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T\mathbf{x}}}$$

And the probability of class 0 is:
$$P(y=0|\mathbf{x}) = 1 - P(y=1|\mathbf{x})$$

### Multi-class Classification (One-vs-Rest):
For multi-class problems, logistic regression typically uses the One-vs-Rest approach:
- Train one binary classifier for each class vs. all other classes
- For prediction, choose the class with the highest probability

### Alternative: Multinomial Logistic Regression:
Uses the softmax function for direct multi-class modeling:
$$P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T\mathbf{x}}}$$

### Cost Function (Log-Likelihood):
Logistic regression minimizes the negative log-likelihood:
$$J(\mathbf{w}) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\mathbf{w}(\mathbf{x}^{(i)})) + (1-y^{(i)})\log(1-h_\mathbf{w}(\mathbf{x}^{(i)}))]$$

### Optimization:
Common optimization methods include:
- **Gradient Descent**: Iterative optimization using gradients
- **Newton-Raphson**: Faster convergence using second derivatives
- **L-BFGS**: Limited-memory optimization for large datasets

### Advantages:
- **Interpretable**: Coefficients show feature importance and direction
- **Probabilistic output**: Provides confidence estimates
- **No tuning required**: Often works well with default parameters
- **Fast training and prediction**: Efficient for large datasets
- **No feature scaling required**: Though it can help convergence
- **Less prone to overfitting**: Especially with regularization

### Disadvantages:
- **Linear decision boundary**: Cannot capture complex non-linear relationships
- **Sensitive to outliers**: Extreme values can affect the model significantly
- **Requires large sample sizes**: For stable results, especially with many features
- **Feature independence assumption**: Performs better when features are not highly correlated
- **Complete separation issues**: Problems when classes are perfectly separable

## Dataset Description

This implementation uses the **Car Evaluation Dataset** from the UCI Machine Learning Repository.

### Dataset Characteristics:
- **Total samples**: 1,728 observations
- **Features**: 6 categorical variables representing car characteristics
- **Target**: Multi-class classification (Car acceptability levels)
- **Complete coverage**: All possible combinations of attribute values are included
- **No missing values**: Clean dataset with complete information

### Feature Descriptions:
| Feature       | Description       | Possible Values       | Encoding      |
|---------------|-------------------|-----------------------|---------------|
| `buying`      | Buying price      | vhigh, high, med, low | 3, 2, 1, 0    |
| `maint`       | Maintenance price | vhigh, high, med, low | 3, 2, 1, 0    |
| `doors`       | Number of doors   | 2, 3, 4, 5more        | 0, 1, 2, 3    |
| `persons`     | Person capacity   | 2, 4, more            | 0, 1, 2       |
| `lug_boot`    | Luggage boot size | small, med, big       | 0, 1, 2       |
| `safety`      | Estimated safety  | low, med, high        | 0, 1, 2       |

### Class Distribution:
| Class | Label                 | Count | Percentage    |
|-------|-----------------------|-------|---------------|
| 0     | unacc (unacceptable)  | 1,210 | 70.0%         |
| 1     | acc (acceptable)      | 384   | 22.2%         |
| 2     | good                  | 69    | 4.0%          |
| 3     | v-good (very good)    | 65    | 3.8%          |

### Hierarchical Structure:
The dataset was derived from a hierarchical decision model:
```
CAR (acceptability)
├── PRICE (overall price)
│   ├── buying (buying price)
│   └── maint (maintenance price)
└── TECH (technical characteristics)
    ├── COMFORT
    │   ├── doors (number of doors)
    │   ├── persons (person capacity)
    │   └── lug_boot (luggage boot size)
    └── safety (estimated safety)
```

### Research Context:
- **Originally created**: For demonstrating DEX expert system (1990)
- **ML applications**: Evaluating structure discovery and constructive induction methods
- **Decision modeling**: Hierarchical multi-attribute decision making
- **Benchmark dataset**: Standard dataset for multi-class classification evaluation

## Implementation Details

### Data Preprocessing:
1. **Data Loading**: Load dataset from CSV format
2. **Column Naming**: Apply descriptive names to features
3. **Categorical Type Definition**: Create ordered categorical types for proper encoding
4. **Ordinal Encoding**: Convert categorical values to numerical codes preserving order
5. **Data Shuffling**: Randomize order for better train/validation/test splits
6. **No Feature Scaling**: Not strictly necessary for logistic regression but can help convergence

### Dataset Splits:
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

### Model Configuration:
- **Algorithm**: scikit-learn's LogisticRegression
- **Multi-class strategy**: One-vs-Rest (default)
- **Solver**: Default solver (typically LBFGS for small datasets)
- **Max iterations**: 200 (increased for convergence)
- **No regularization tuning**: Uses default L2 regularization

### Key Implementation Features:
- Comprehensive categorical data visualization with pie charts
- Proper ordinal encoding preserving feature order relationships
- Distribution analysis by class using KDE plots
- Correlation matrix visualization
- Confusion matrices for detailed performance analysis
- Classification reports with precision, recall, and F1-scores

### Why Logistic Regression for this Dataset:
- **Ordinal features**: Can capture linear relationships in ordered categorical data
- **Multi-class problem**: Handles the 4-class classification naturally
- **Interpretability**: Coefficients show which car features most influence acceptability
- **Baseline model**: Provides good baseline for comparison with more complex algorithms
- **Class imbalance**: Handles imbalanced classes reasonably well

## Usage

### Prerequisites:
Make sure you have the required dependencies installed (see [Dependencies](#dependencies) section).

### Running the Implementation:

1. **Navigate to the Logistic Regression directory:**
   ```bash
   cd src/algorithms/logistic_regression/
   ```

2. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook logistic_regression.ipynb
   ```

3. **Execute the cells sequentially** to:
   - Install and import required libraries (including comprehensive metrics)
   - Load and preprocess the car evaluation dataset
   - Visualize feature distributions and correlations
   - Split the data into train/validation/test sets
   - Train the logistic regression model
   - Evaluate performance with comprehensive metrics, visual analysis, and cross-validation

### Code Structure:
```
logistic_regression.ipynb
├── 01. Library Installation and Imports
├── 02. Data Loading and Preprocessing
├── 03. Data Visualization and Exploration
├── 04. Dataset Splitting and Scaling
├── 05. Logistic Regression Implementation
├── 06. Comprehensive Model Evaluation
│   ├── Basic Classification Metrics (Accuracy, Precision, Recall, F1)
│   ├── Advanced Metrics (Balanced Accuracy, MCC, Cohen's Kappa)
│   ├── Probability-based Metrics (ROC-AUC, PR-AUC, Log Loss)
│   ├── Visual Analysis (ROC Curves, PR Curves for Multi-class)
│   ├── Cross-Validation with Confidence Intervals
│   ├── Error Analysis and Confidence Assessment
│   └── Learning Curve Analysis
└── 07. Summary and Interpretation
```

### Expected Workflow:
1. **Data Exploration**: Understand categorical feature distributions and car acceptability patterns
2. **Preprocessing**: Proper ordinal encoding of categorical variables to preserve relationships
3. **Visualization**: Analyze feature relationships and class distributions
4. **Model Training**: Fit logistic regression with proper iterations for convergence
5. **Evaluation**: Comprehensive performance analysis across multiple metrics and visualizations

## Results

The implementation provides comprehensive evaluation metrics to thoroughly assess model performance:

### Basic Performance Metrics:
- **Accuracy**: Overall classification accuracy on validation and test sets
- **Precision**: Proportion of positive predictions that are correct (per class and weighted)
- **Recall (Sensitivity)**: Proportion of actual positives correctly identified (per class and weighted)
- **F1-Score**: Harmonic mean of precision and recall (per class and weighted)
- **Balanced Accuracy**: Accuracy adjusted for class imbalance
- **Matthews Correlation Coefficient (MCC)**: Balanced measure considering all confusion matrix elements

### Advanced Evaluation:
- **Cohen's Kappa**: Inter-rater reliability accounting for chance agreement
- **ROC-AUC (One-vs-Rest)**: Area under the Receiver Operating Characteristic curve for multi-class
- **Average Precision (PR-AUC)**: Area under Precision-Recall curve (better for imbalanced data)
- **Log Loss**: Probabilistic loss function measuring prediction uncertainty

### Visual Analysis:
- **ROC Curves (Multi-class)**: True vs False Positive Rate for each class vs rest
- **Precision-Recall Curves (Multi-class)**: Precision vs Recall trade-off for each class
- **Feature Distribution Plots**: KDE plots showing feature distributions by car acceptability class
- **Learning Curves**: Training and validation scores vs training set size

### Statistical Validation:
- **Cross-Validation**: 5-fold stratified cross-validation with confidence intervals
- **Error Analysis**: Detailed analysis of misclassified samples by car acceptability class
- **Confidence Analysis**: Analysis of prediction confidence for correct vs incorrect classifications
- **Performance Stability**: Multiple metrics to ensure robust evaluation

### Expected Performance:
- Logistic regression should perform well on this dataset due to clear ordinal relationships in features
- The model will likely excel at identifying unacceptable cars (majority class - 70% of data)
- Performance on minority classes (good: 4%, v-good: 3.8%) may be limited due to severe class imbalance
- Cross-validation provides confidence intervals for performance estimates
- Feature coefficients provide insights into which car attributes most influence acceptability decisions

### Performance Characteristics:
- **Fast Training**: Very quick to train even on larger datasets
- **Interpretable Results**: Clear understanding of feature importance through coefficients
- **Probabilistic Output**: Confidence estimates for each prediction class
- **Robust Evaluation**: Multiple complementary metrics provide comprehensive assessment
- **Multi-class Handling**: Effective One-vs-Rest strategy for handling 4-class problem

## Dependencies

```python
# Core libraries
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine Learning
scikit-learn>=1.0.0

# Additional metrics and evaluation
# (included in scikit-learn):
# - sklearn.metrics for comprehensive evaluation
# - sklearn.model_selection for cross-validation
# - sklearn.preprocessing for data processing and binarization
```

### Installation:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Or install from the notebook:
```python
%pip install matplotlib numpy pandas scikit-learn seaborn
```

---

## Educational Notes

This implementation serves as an excellent introduction to:
- **Linear classification**: Understanding linear decision boundaries
- **Multi-class classification**: Handling problems with more than 2 classes
- **Categorical data processing**: Proper encoding of ordinal categorical features
- **Model interpretability**: Understanding feature coefficients and their meanings
- **Performance evaluation**: Comprehensive metrics for multi-class problems

### Comparison with Previous Algorithms:
- **vs. k-NN**: Much faster prediction, more interpretable, but linear boundaries only
- **vs. Naive Bayes**: No independence assumption, but requires linear separability
- **Decision boundaries**: Linear vs. complex non-linear boundaries
- **Interpretability**: High interpretability with coefficient analysis

### When to Use Logistic Regression:
- **Linear relationships**: When features have roughly linear relationships with log-odds
- **Interpretability required**: When you need to understand feature importance
- **Baseline model**: As a starting point before trying complex algorithms
- **Large datasets**: When you need fast training and prediction
- **Probabilistic output**: When you need confidence estimates, not just classifications

The Logistic Regression implementation provides an excellent foundation for understanding linear classification and serves as a crucial stepping stone to more advanced machine learning algorithms.
