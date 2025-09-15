<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>


# Machine Learning Interview Notes

## 2. Ensemble Methods

### 2.1 Overview
- **Definition**: Combine multiple models to improve performance.
- **Key Types**:
  - **Bagging** (e.g., Random Forest)
  - **Boosting** (e.g., Gradient Boosting, XGBoost)
  - **Stacking**

---

### 2.2 Bagging
- **Definition**: Bagging (Bootstrap Aggregating) trains multiple models on different random subsets of data and aggregates their results to improve stability and accuracy.

#### How It Works:
1. **Data Sampling**: Create multiple random subsets of the dataset (with replacement).
2. **Independent Training**: Train a model (e.g., decision tree) on each subset.
3. **Aggregation**: Combine results using averaging (regression) or majority voting (classification).

---

#### Pseudo Code: Decision Tree Training
```pseudo
function train_decision_tree(data, target, depth):
    if depth == 0 or data is homogeneous:
        return create_leaf_node(target)
    
    best_split = find_best_split(data, target)
    left_data, right_data = split_data(data, best_split)
    
    left_subtree = train_decision_tree(left_data, target, depth - 1)
    right_subtree = train_decision_tree(right_data, target, depth - 1)
    
    return create_internal_node(best_split, left_subtree, right_subtree)
```

#### How a Decision Tree Finds the Best Fit

A **decision tree** finds the "best fit" by selecting a feature and a threshold that splits the dataset into two subsets, aiming to maximize the **purity** of the resulting subsets (i.e., make them as homogeneous as possible). This process is repeated recursively to build the tree.

---

##### Steps to Find the Best Split:

   1. **Choose a Splitting Criterion**:
      The algorithm evaluates potential splits using a metric that measures the "quality" of the split. Common criteria include:
      - **Gini Impurity** (used in classification)
      - **Entropy/Information Gain** (used in classification)
      - **Mean Squared Error (MSE)** (used in regression)


##### Algorithm for Finding the Best Split:
1. **For each feature** in the dataset:
   - Try all possible thresholds (e.g., midpoints between feature values).
   - Split the data into two subsets based on the threshold.
   - Calculate the splitting criterion (e.g., Gini, entropy, MSE).

2. **Choose the split** with the lowest impurity or highest information gain.

3. **Repeat recursively** for each subset to grow the tree.

---

#### Pros and Cons of Bagging
- **Pros**:
  - Reduces variance and overfitting.
  - Robust to noisy data.
  - Models can be trained in parallel.
- **Cons**:
  - Doesn’t reduce bias.
  - Requires more computational resources.

---

### 2.3 Boosting
- **Definition**: Boosting trains models sequentially, where each model corrects the errors of its predecessor, leading to improved performance on difficult instances.

#### How It Works:
1. **Initialize Weights**: Assign equal weights to all training samples.
2. **Train Weak Learner**: Train a simple model (e.g., shallow decision tree).
3. **Update Weights**: Increase weights for misclassified samples to focus on them.
4. **Combine Models**: Aggregate predictions using weighted majority or summation.

---

#### Pseudo Code: Boosting
```pseudo
function train_boosting(data, target, num_iterations):
    weights = initialize_weights(data)
    models = []
    
    for iteration in range(num_iterations):
        model = train_weak_learner(data, target, weights)
        error = calculate_error(model, data, target, weights)
        
        update_weights(weights, model, data, target)
        models.append(model)
    
    return combine_models(models)
```
---

#### weak learner:
- a simple model that performs slightly better than random guessing
- It's called "weak" because it may not perform well on its own but can contribute to an overall strong model when combined with other weak learners
- e.g.
  - **Decision stumps**: Decision trees with only one split.
  - **Shallow trees**: Decision trees with limited depth (e.g., depth = 1 or 2).
  - **Linear classifiers**: Simple models like logistic regression or a single-layer perceptron.

#### Pros and Cons of Boosting
- **Pros**:
  - Reduces bias and variance.
  - Focuses on difficult-to-predict instances.
  - Flexible and tunable (e.g., learning rate, number of estimators).
- **Cons**:
  - Computationally expensive.
  - Can overfit without proper tuning.

---

### 2.4 Comparison of Bagging and Boosting
| **Aspect**      | **Bagging**                          | **Boosting**                         |
|-----------------|--------------------------------------|--------------------------------------|
| **Training**    | Parallel                             | Sequential                          |
| **Focus**       | Reduces variance                     | Reduces bias and variance           |
| **Strength**    | Handles overfitting                  | Improves weak learners              |
| **Speed**       | Faster (parallelizable)              | Slower (sequential)                 |
| **Overfitting** | Less prone                           | More prone without tuning           |
| **Example**     | Random Forest                        | Gradient Boosting, XGBoost, AdaBoost          |

### 2.5 Decision Trees
**Decision trees** by themselves are neither bagging nor boosting. However, they are commonly used as the base learners in both bagging and boosting frameworks.

---

#### **Decision Trees in Bagging**
- **How It Works**:
  - Bagging, short for **Bootstrap Aggregating**, trains multiple decision trees independently on different bootstrapped subsets of the data.
  - Example: **Random Forest** combines decision trees trained in parallel, with randomness added in feature selection and dataset sampling.

- **Why Decision Trees?**:
  - Decision trees are prone to overfitting, and bagging helps reduce variance by averaging the predictions of many trees.
  - Each tree is a **strong learner** trained independently.

- **Example**:
  - Random Forest is a classic example where decision trees are the base learners in a bagging ensemble.

---

#### **Decision Trees in Boosting**
- **How It Works**:
  - Boosting builds an ensemble of decision trees sequentially, where each tree tries to correct the errors of the previous ones.
  - Example: **AdaBoost** uses shallow decision trees (often stumps, i.e., trees with depth = 1) as weak learners, combining them to form a strong model.

- **Why Decision Trees?**:
  - Shallow decision trees are fast to train and prone to underfitting individually, making them ideal weak learners for boosting.
  - The sequential training process reduces bias and improves overall accuracy.

- **Example**:
  - Gradient Boosting, XGBoost, and LightGBM use decision trees as the base learners in a boosting ensemble.

---

#### **Key Takeaway**
- A single **decision tree** is neither bagging nor boosting.
- In **bagging**, decision trees are trained independently on random subsets of the data.
- In **boosting**, decision trees are trained sequentially, focusing on correcting errors of previous trees.

### 2.5 Gradient Boosting Machine (GBM)

**Definition**:  
Gradient Boosting Machines (GBM) are a type of **ensemble method** that combines multiple weak learners (usually shallow decision trees) to form a strong predictive model. GBM works by training models sequentially, where each model tries to correct the errors of its predecessor.

---

#### **Why GBM is an Ensemble Method**
- GBM builds an ensemble of weak learners to reduce both **bias** and **variance**.
- It uses **boosting**, which means the models are trained sequentially, and each model focuses on the residual errors of the previous models.

#### **How GBM Works**

1. **Initial Prediction**:
   - Start with a simple model, such as predicting the mean of the target variable (regression) or uniform probabilities (classification).

2. **Sequential Training**:
   - Train a weak learner (e.g., decision tree) to predict the residuals (errors) from the previous step.

3. **Update Predictions**:
   - Combine the predictions of the weak learner with the existing ensemble using a **learning rate** to control the contribution.

4. **Repeat**:
   - Continue adding weak learners until the stopping criterion is met (e.g., a fixed number of iterations or minimal residual error).

#### **Key Components**

1. **Weak Learners**:
   - Typically, shallow decision trees are used as weak learners.

2. **Learning Rate (η)**:
   - Controls how much each new model contributes to the ensemble.
   - Lower values reduce the risk of overfitting but require more iterations.

3. **Loss Function**:
   - Measures the difference between the predicted and actual values.
   - Common choices:
     - **MSE (Mean Squared Error)** for regression.
     - **Log Loss** for classification.

#### **Why GBM is Powerful**
- Captures complex non-linear relationships.
- Reduces both bias and variance through iterative improvements.
- Works well with structured/tabular data.

#### **Popular GBM Frameworks**

1. **XGBoost**:
   - Highly optimized and scalable.
   - Supports regularization (L1 and L2) to prevent overfitting.
   - Handles missing values automatically.

2. **LightGBM**:
   - Faster and more efficient, especially with large datasets.
   - Splits leaf-wise instead of level-wise for better performance.

3. **CatBoost**:
   - Designed for categorical features.
   - Requires minimal data preprocessing.

#### **Pros and Cons**

**Pros**:
- High accuracy and performance.
- Can handle missing data and outliers.
- Flexible with different loss functions.

**Cons**:
- Computationally expensive, especially for large datasets.
- Requires careful hyperparameter tuning.
- Sensitive to noisy data and overfitting if not properly regularized.

#### **When to Use GBM**
- For structured/tabular datasets.
- When high accuracy is required.
- When other simpler models underperform.

#### Example (Scikit-learn)
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```


