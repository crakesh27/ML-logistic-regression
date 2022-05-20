# ML Classification Algorithms
Simple python implementation of various ML binary and multiclass classification algortithms.

## Requirements
* python
* numpy
* pandas
* matplotlib
* mpl_toolkits

## 1. Logistic regression
code file: `1_logisitic_binary.py`

data file: `data3.xlsx`

This dataset contains 4 features for different instances as four columns and fifth column is for class label.

Logistic regression algorithm for the binary classification using hold-out cross-validation technique with 60 % of instances as training and the remaining 40% as testing.

## 2. Multiclass logistic regression
code file: `2_logistic_multiclass.py`

data file: `data4.xlsx`

This dataset contains 4 features for different instances as four columns and fifth column is for class label (3 classes).

Multiclass logistic regression algorithm using both “One VS All” and “One VS One” multiclass coding techniques. Hold-out cross-validation approach (60% training and 40% testing) used for the selection of training and test instances.

## 3. Multiclass logistic regression classifier using 5-fold cross-validation
code file: `3_logisitic_multiclass_k_fold.py`

data file: `data4.xlsx`

Multiclass logistic regression classifier using 5-fold cross-validation approach.

## 4. Likelihood ratio test (LRT) for the binary classification
code file: `4_LRT_binary.py`

data file: `data3.xlsx`

Likelihood ratio test (LRT) for the binary classification using hold-out cross-validation technique with 60 % of instances as training and the remaining 40% as testing.

## 5. Maximum a posteriori (MAP) decision rule for multiclass classification
code file: `5_MAP_multiclass.py`

data file: `data4.xlsx`

Maximum a posteriori (MAP) decision rule for multiclass classification task using hold-out cross-validation approach (70% training and 30% testing).

## 6. Maximum likelihood (ML) decision rule for multiclass classification
code file: `6_ML_multiclass.py`

data file: `data4.xlsx`

Maximum likelihood (ML) decision rule for multiclass classification using hold-out cross-validation approach (70% training and 30% testing).

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.