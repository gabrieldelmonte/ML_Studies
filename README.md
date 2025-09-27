# ML_Studies

A comprehensive collection of machine learning algorithm implementations with extensive evaluation metrics and educational resources.

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Models](#running-the-models)
- [Key Features](#key-features)
- [Educational Value](#educational-value)
- [Contributing](#contributing)
- [License](#license)

## Overview

**ML_Studies** is an educational repository containing comprehensive implementations of fundamental machine learning algorithms. Each implementation includes detailed explanations, extensive evaluation metrics, and real-world datasets to provide a thorough understanding of how these algorithms work in practice.

This repository serves as both a learning resource for students and practitioners new to machine learning, and a reference implementation for more experienced developers looking for well-documented, production-quality code examples.

## Motivation

The creation of this repository was driven by several key objectives:

### **Educational Purpose**
- **Bridge Theory and Practice**: Provide implementations that clearly connect mathematical concepts to working code
- **Comprehensive Documentation**: Each algorithm includes detailed explanations of how it works, when to use it, and its advantages/disadvantages
- **Multiple Perspectives**: Show different approaches to evaluating and understanding model performance

### **Research and Experimentation**
- **Standardized Evaluation**: All implementations use consistent evaluation metrics for fair algorithm comparison
- **Reproducible Results**: Fixed random seeds and detailed methodology ensure reproducible experiments
- **Real-World Datasets**: Use of diverse, well-known datasets from different domains (medical, computer vision, game theory, etc.)

### **Practical Implementation**
- **Production-Ready Code**: Clean, well-structured code that follows best practices
- **Complete Workflow**: From data loading and preprocessing to model evaluation and interpretation
- **Visual Analysis**: Comprehensive visualizations including ROC curves, precision-recall curves, and learning curves

### **Knowledge Sharing**
- **Documentation Standards**: Each algorithm includes comprehensive README files with mathematical foundations
- **Best Practices**: Demonstrate proper ML workflow including data preprocessing, model validation, and performance evaluation
- **Comparative Analysis**: Enable easy comparison between different algorithms on the same datasets

## Repository Structure

```
ML_Studies/
├── README.md               # This file - main repository documentation
├── LICENSE                 # License file
├── .gitignore              # Git ignore configuration
└── src/                    # Source code and implementations
    ├── README.md           # Detailed setup and running instructions
    ├── requirements.txt    # Python dependencies
    ├── algorithms/         # Algorithm implementations
    │   └── ...
    └── datasets/           # Curated datasets for experiments
        └── ...
```

## Getting Started

### Prerequisites

- **Python 3.9+** (recommended: Python 3.11+)
- **Git** for cloning the repository
- **4GB+ RAM** (8GB+ recommended for larger datasets)

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gabrieldelmonte/ML_Studies.git
   cd ML_Studies
   ```

2. **Set up the Python environment:**
   ```bash
   cd src
   python -m venv ml_venv
   source ml_venv/bin/activate  # On Windows: ml_venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

5. **Start exploring!** Navigate to any algorithm folder and open the `.ipynb` file.

## Running the Models

### Individual Algorithm Exploration
Navigate to any algorithm directory and run the Jupyter notebook:

```bash
cd src/algorithms/knn/
jupyter notebook knn.ipynb
```

## Key Features

### **Comprehensive Evaluation Metrics** (15+ metrics per algorithm)
- **Basic Metrics**: Accuracy, Precision, Recall, F1-Score
- **Advanced Metrics**: ROC-AUC, PR-AUC, Matthews Correlation Coefficient
- **Statistical Validation**: Cross-validation with confidence intervals
- **Visual Analysis**: ROC curves, Precision-Recall curves, Learning curves

### **Extensive Visualizations**
- Feature distribution analysis
- Correlation matrices and heatmaps  
- Performance curves and confidence intervals
- Error analysis and misclassification patterns

### **Robust Data Preprocessing**
- Missing value handling
- Feature scaling and normalization
- Class imbalance treatment (oversampling/undersampling)
- Categorical data encoding

### **Educational Documentation**
- Mathematical foundations for each algorithm
- Step-by-step implementation explanations
- When to use each algorithm (advantages/disadvantages)
- Real-world applications and use cases

### **Reproducible Research**
- Fixed random seeds for consistent results
- Detailed methodology documentation
- Version-controlled dependencies
- Standardized evaluation procedures

## Educational Value

This repository is designed for:

### **Students**
- Learn ML algorithms through hands-on implementation
- Understand the complete ML pipeline from data to insights
- Compare different approaches on real datasets
- Build practical skills with industry-standard tools

### **Educators**
- Use as curriculum material for ML courses
- Demonstrate theoretical concepts with working code
- Provide students with ready-to-use experimental frameworks
- Show best practices in ML implementation

### **Practitioners**
- Reference implementations for common algorithms
- Evaluation metrics and validation techniques
- Data preprocessing and visualization patterns
- Comparative analysis methodologies

### **Researchers**
- Baseline implementations for algorithm comparison
- Standardized evaluation protocols
- Reproducible experimental setups
- Foundation for extending to novel approaches

## Contributing

Contributions are welcome! Here's how you can help:

### **Algorithm Implementations**
- Add new ML algorithms following the established patterns
- Ensure comprehensive documentation and evaluation
- Include appropriate datasets and use cases

### **Dataset Additions**
- Contribute new datasets from different domains
- Ensure proper documentation and preprocessing
- Maintain data quality and ethical considerations

### **Documentation Improvements**
- Enhance existing README files
- Add mathematical explanations
- Improve code comments and docstrings

### **Bug Fixes and Optimizations**
- Report and fix bugs
- Optimize code performance
- Improve visualization quality

### **Contribution Guidelines**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper documentation
4. Add comprehensive tests and evaluation
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **UCI Machine Learning Repository** for providing high-quality datasets
- **scikit-learn** for comprehensive ML algorithms and evaluation tools
- **TensorFlow/Keras** for deep learning implementations
- **Jupyter** for interactive development environment
- **Open Source Community** for the excellent Python ML ecosystem
