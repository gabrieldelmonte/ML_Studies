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
   - Install and import required libraries
   - Load and preprocess the car evaluation dataset
   - Visualize feature distributions and correlations
   - Split the data into train/validation/test sets
   - Train the logistic regression model
   - Evaluate performance with classification reports and confusion matrices

### Code Structure:
```
logistic_regression.ipynb
├── 01. Library Installation
├── 02. Library Imports
├── 03. Data Loading and Preprocessing
├── 04. Data Visualization
├── 05. Dataset Splitting and Scaling
└── 06. Logistic Regression Implementation and Evaluation
```

### Expected Workflow:
1. **Data Exploration**: Understand categorical feature distributions
2. **Preprocessing**: Proper ordinal encoding of categorical variables
3. **Visualization**: Analyze relationships and correlations
4. **Model Training**: Fit logistic regression with proper iterations
5. **Evaluation**: Comprehensive performance analysis

## Results

The implementation provides comprehensive evaluation metrics:

- **Classification Reports**: Precision, recall, F1-score for all 4 classes
- **Validation Performance**: Model evaluation on unseen validation data
- **Test Performance**: Final model evaluation on test data
- **Confusion Matrices**: Visual representation of prediction accuracy per class
- **Correlation Analysis**: Understanding feature relationships

### Expected Performance:
- Logistic regression should perform well on this dataset due to the clear ordinal relationships
- The model will likely excel at identifying unacceptable cars (majority class)
- Performance on minority classes (good, v-good) may be limited due to class imbalance
- Feature coefficients will provide insights into which attributes most influence car acceptability

### Performance Characteristics:
- **Fast training**: Very quick to train even on larger datasets
- **Interpretable results**: Clear understanding of feature importance
- **Probabilistic output**: Confidence estimates for each prediction
- **Good baseline**: Solid performance for comparison with complex models

## Dependencies

```python
# Core libraries
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine Learning
scikit-learn>=1.0.0
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
