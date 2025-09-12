# k-Nearest Neighbors (kNN) Implementation

This directory contains a comprehensive implementation of the k-Nearest Neighbors algorithm using the MAGIC Gamma Telescope Dataset for binary classification.

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
- **k values tested**: 1, 3, 5 neighbors

### Key Implementation Features:
- Comprehensive data visualization with histograms by class
- Modular scaling and oversampling function
- Multiple k-value evaluation
- Detailed classification reports for both validation and test sets

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
   - Install and import required libraries
   - Load and preprocess the dataset
   - Visualize feature distributions
   - Split and scale the data
   - Train kNN models with different k values
   - Evaluate model performance

### Code Structure:
```
knn.ipynb
├── 01. Library Installation
├── 02. Library Imports
├── 03. Data Loading and Preprocessing
├── 04. Data Visualization
├── 05. Dataset Splitting and Scaling
└── 06. kNN Implementation and Evaluation
```

## Results

The implementation evaluates kNN performance with k=1, k=3, and k=5 neighbors, providing:

- **Classification Reports**: Precision, recall, F1-score for both classes
- **Validation Performance**: Model evaluation on unseen validation data
- **Test Performance**: Final model evaluation on test data
- **Class Distribution Analysis**: Visual comparison of feature distributions between gamma and hadron classes

### Expected Performance:
- kNN typically performs well on this dataset due to the clear separation between gamma and hadron classes in the feature space
- Feature scaling is crucial for good performance
- Oversampling helps address class imbalance in the training data

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
- **Classification algorithms**: Understanding supervised learning
- **Data preprocessing**: Feature scaling and handling class imbalance
- **Model evaluation**: Using validation sets and classification metrics
- **Hyperparameter tuning**: Finding optimal k values
- **Data visualization**: Understanding feature distributions

The kNN algorithm is particularly valuable for beginners because of its intuitive nature and the ability to visualize decision boundaries in low-dimensional spaces.
