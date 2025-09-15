<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Machine Learning Interview Notes

## Supervised Models

### 1. Linear Regression

**Definition**:  
Linear Regression is a supervised learning algorithm used for predicting a **continuous** target variable based on one or more input features. (i.e., only for regression tasks) It assumes a linear relationship between the input features (independent variables) and the target variable (dependent variable).

#### Equation
The model predicts the target variable \( y \) using the formula:  
$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon
$$
  
Where:  
- `β0`: the intercept (bias).  
- `β1, β2, ..., βn`: the coefficients (weights).  
- `x1, x2, ..., xn`: the input features.  
- `ε`: the error term (noise).

#### Loss Function (Mean Squared Error - MSE)
The goal is to minimize the difference between predicted and actual values by minimizing the **Mean Squared Error (MSE)**:  
$$
MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$  
Where:  
- `N`: the number of data points.  
- `yi`: the actual value.  
- `ŷi`: the predicted value.

---

#### Algorithm (Pseudo Code)
```pseudo
function train_linear_regression(X, y):
    # X: input features, y: target variable
    # Add a bias column to X
    X = add_bias_column(X)
    
    # Compute weights using Normal Equation
    weights = (X' * X)^(-1) * X' * y
    
    return weights

function predict_linear_regression(X, weights):
    return X * weights
```

#### Example (Scikit-learn)
```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

#### Pros and Cons

**Pros**:
- Simple and easy to interpret.
- Computationally efficient.
- Works well when the relationship between features and target is linear.

**Cons**:
- Assumes linearity, which may not hold in real-world data.
- Sensitive to outliers.
- Poor performance on complex, non-linear relationships.

---

### 2. Polynomial Regression
Polynomial Regression is an extension of Linear Regression where the relationship between the input features and the target variable is modeled as an \(n\)-degree polynomial. (i.e., only for regression tasks) It is used when data shows a non-linear relationship.

#### Equation
The model predicts the target variable `y` using the following formula: 
<center>
y = β0 + β1 * x + β2 * x^2 + β3 * x^3 + ... + βn * x^n
</center>

Where:  
- `β0`: Intercept (bias term).  
- `β1, β2, ..., βn`: Coefficients for polynomial terms.  
- `x, x^2, x^3, ..., x^n`: Input features transformed into polynomial terms.  

#### Loss Function (Mean Squared Error)
The goal is to minimize the Mean Squared Error (MSE) between predicted and actual values:
<center>
MSE = (1 / N) * Σ (yi - ŷi)^2
</center>

Where:  
- `yi`: Actual value.  
- `ŷi`: Predicted value.  
- `N`: Number of data points.

#### Algorithm (Pseudo Code)
```pseudo
function train_polynomial_regression(X, y, degree):
    # Step 1: Transform X into polynomial features of the given degree
    X_poly = generate_polynomial_features(X, degree)
    
    # Step 2: Use Linear Regression on the transformed features
    weights = train_linear_regression(X_poly, y)
    
    return weights

function predict_polynomial_regression(X, weights, degree):
    X_poly = generate_polynomial_features(X, degree)
    return X_poly * weights
```

#### Pros and Cons

**Pros**:
- Can model non-linear relationships effectively.
- Extends linear regression without requiring a new algorithm.

**Cons**:
- Prone to overfitting if the degree is too high.
- Requires careful selection of the polynomial degree.
- Can become computationally expensive for high-degree polynomials.

#### When to Use Polynomial Regression:
- When you notice a non-linear relationship in your data but still want to use a regression-based approach.
- For datasets where the relationship between variables is not captured well by a simple straight line.

---

### 3. Logistic Regression
**Definition**:  
Logistic Regression is a supervised learning algorithm used for **binary classification** tasks. Instead of predicting a continuous value like Linear Regression, it predicts the probability of a data point belonging to one of two classes.

#### Equation
Logistic Regression uses the **sigmoid function** to map predictions to probabilities between 0 and 1:
<center>
P(y=1|x) = 1 / (1 + exp(-z))
</center>

Where:
- `z = β0 + β1 * x1 + β2 * x2 + ... + βn * xn`
- `β0`: Intercept (bias term).  
- `β1, β2, ..., βn`: Coefficients for the input features.  
- `x1, x2, ..., xn`: Input features.

The predicted class is determined by applying a **threshold** (typically 0.5):
<center>
y_pred = 1 if P(y=1|x) >= 0.5 else 0
</center>

#### Loss Function (Log Loss / Cross-Entropy Loss)
The loss function for Logistic Regression is the **logarithmic loss**:
<center>
Log Loss = -(1/N) * Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
</center>

Where:
- `N`: Number of data points.
- `y_i`: Actual label (0 or 1).
- `p_i`: Predicted probability for class 1.


#### Algorithm (Pseudo Code)
```pseudo
function train_logistic_regression(X, y, learning_rate, iterations):
    initialize weights and bias to 0
    
    for i in range(iterations):
        # Compute linear combination
        z = X * weights + bias
        
        # Apply sigmoid function
        predictions = 1 / (1 + exp(-z))
        
        # Compute gradient
        gradient_weights = (1/N) * X.T * (predictions - y)
        gradient_bias = (1/N) * sum(predictions - y)
        
        # Update weights and bias
        weights = weights - learning_rate * gradient_weights
        bias = bias - learning_rate * gradient_bias
    
    return weights, bias

function predict_logistic_regression(X, weights, bias, threshold=0.5):
    probabilities = 1 / (1 + exp(-(X * weights + bias)))
    return [1 if p >= threshold else 0 for p in probabilities]
```

#### Pros and Cons

**Pros**:
- Simple and easy to interpret.
- Computationally efficient and fast to train.
- Outputs probabilities, useful for decision-making.
- Works well for linearly separable data.

**Cons**:
- Assumes a linear relationship between the input features and the log-odds of the target.
- Can underperform when dealing with non-linear data.
- Sensitive to outliers.

#### When to Use Logistic Regression:
- When you have a **binary classification problem**.
- When the relationship between features and the target is approximately linear in the log-odds.
- For interpretable models where you need to understand feature importance.

### 4. Decision Trees

**Definition**:  
A Decision Tree is a supervised learning algorithm used for both classification and regression tasks. It splits the data into subsets based on feature values, forming a tree structure where each node represents a feature, each branch represents a decision rule, and each leaf node represents the predicted outcome.

#### How It Works
1. Start at the root node (entire dataset).
2. Split the data based on the feature that provides the best split (measured by a criterion like Gini Impurity, Entropy, or MSE).
3. Repeat the splitting process recursively for each subset.
4. Stop when a stopping criterion is met (e.g., maximum depth or pure leaf nodes).

---

#### Key Concepts
1. **Impurity Metrics**:
   - **Gini Impurity** (used in classification):
     ```
     G = 1 - Σ(p_i^2)
     ```
     Where `p_i` is the probability of class `i` in the subset.

   - **Entropy** (used in classification):
     ```
     H = -Σ(p_i * log2(p_i))
     ```

   - **Mean Squared Error (MSE)** (used in regression):
     ```
     MSE = (1 / N) * Σ(y_i - ȳ)^2
     ```

2. **Splitting Criteria**:
   - Choose the feature and threshold that minimize the impurity of the resulting subsets.

#### Algorithm (Pseudo Code)
```pseudo
function build_decision_tree(data, depth, max_depth):
    if stopping_criteria_met or depth == max_depth:
        return leaf_node
    
    best_split = find_best_split(data)
    left, right = split_data(data, best_split)
    
    return node(
        feature=best_split.feature,
        threshold=best_split.threshold,
        left=build_decision_tree(left, depth+1, max_depth),
        right=build_decision_tree(right, depth+1, max_depth)
    )
```
#### Pros and Cons

**Pros**:
- Simple to understand and interpret.
- Can handle both numerical and categorical data.
- Non-parametric, so no assumptions about data distribution.

**Cons**:
- Prone to overfitting (high variance) without proper regularization.
- Sensitive to small changes in the data.
- May struggle with linear relationships unless properly tuned.

#### When to Use Decision Trees:
- When interpretability is crucial.
- For datasets with non-linear relationships.
- As base models for ensemble methods like Random Forests or Gradient Boosting.

#### Are Decision Trees Only for Classification?

No, **decision trees** can be used for both **classification** and **regression** tasks. Here's the difference between the two:

##### **1. Decision Trees for Classification**
- **Goal**: Assign data points to discrete categories (e.g., "Yes" or "No", "Cat" or "Dog").
- **How It Works**:
  - Splits the dataset based on impurity measures such as **Gini Impurity** or **Entropy**.
  - Each leaf node represents a class label, and the decision path through the tree assigns a data point to a specific class.
  
- **Examples**:
  - Predicting whether a loan will be approved (binary classification).
  - Classifying species of flowers (multi-class classification).

##### **2. Decision Trees for Regression**
- **Goal**: Predict continuous values (e.g., house prices, stock prices).
- **How It Works**:
  - Splits the dataset to minimize the **Mean Squared Error (MSE)** or **Mean Absolute Error (MAE)**.
  - Each leaf node represents the average value of the target variable for the data points in that leaf.
  
- **Examples**:
  - Predicting the price of a house based on its features.
  - Forecasting the temperature for the next day.

##### **Key Differences Between Classification and Regression Trees**
| Aspect                | Classification Tree                | Regression Tree                  |
|-----------------------|-------------------------------------|----------------------------------|
| **Output**            | Class labels                       | Continuous values                |
| **Split Criterion**   | Gini Impurity, Entropy             | MSE, MAE                         |
| **Leaf Node Value**   | Class label                        | Average of target values         |
| **Loss Function**     | Log Loss                           | MSE or MAE                       |

##### **Scikit-learn Example: Regression Tree**
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, len(X))

# Train a regression tree
model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(X, y)

# Predict and plot
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

plt.scatter(X, y, label="Data", color="blue")
plt.plot(X_test, y_pred, label="Regression Tree Fit", color="red")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Decision Tree Regression")
plt.show()
```

### 5. Support Vector Machines (SVM)

**Definition**:  
Support Vector Machines (SVM) are supervised learning algorithms used for both **classification** and **regression** tasks. They work by finding the hyperplane that best separates data points from different classes in a high-dimensional space.

#### **Key Concepts**

1. **Hyperplane**:
   - A decision boundary that separates different classes.
   - In a 2D space, it's a line; in a 3D space, it's a plane; in higher dimensions, it's a hyperplane.

2. **Support Vectors**:
   - The data points closest to the hyperplane that influence its position and orientation.
   - The SVM algorithm uses only these points to define the hyperplane, making it robust to outliers.

3. **Margin**:
   - The distance between the hyperplane and the nearest support vectors.
   - SVM maximizes this margin to achieve the best separation.

<center>
<img src="../../images/svm.png" alt="Overfitting v.s. Underfitting" width="400" />

##### [source: edureka.com]
</center>

#### **Types of SVM**

1. **Linear SVM**:
   - Used when data is linearly separable.
   - Finds a straight hyperplane that separates the classes.

2. **Non-Linear SVM**:
   - Used when data is not linearly separable.
   - Applies the **kernel trick** to transform data into a higher-dimensional space where a linear hyperplane can separate the classes.

#### **Kernel Functions** (for Non-Linear SVM)
1. **Linear Kernel**:
   - No transformation applied.
2. **Polynomial Kernel**:
   - Transforms data into polynomial features.
3. **RBF (Radial Basis Function) Kernel**:
   - Maps data to an infinite-dimensional space, commonly used for non-linear problems.
4. **Sigmoid Kernel**:
   - Similar to the activation function in neural networks.

#### **Mathematical Formulation**
1. **Objective Function**:
   - Maximize the margin while minimizing misclassification.
   - For a hyperplane defined by `w` and `b`, the optimization problem is:
     ```
     Minimize: (1/2) * ||w||^2
     Subject to: y_i * (w * x_i + b) >= 1 for all i
     ```

2. **Soft Margin (for non-linearly separable data)**:
   - Introduces a penalty for misclassified points using a regularization parameter `C`:
     ```
     Minimize: (1/2) * ||w||^2 + C * Σ ξ_i
     ```
     Where `ξ_i` are slack variables for misclassified points.

#### **Pros and Cons**

**Pros**:
- Effective in high-dimensional spaces.
- Works well with both linear and non-linear data using kernels.
- Robust to outliers (focuses on support vectors).

**Cons**:
- Computationally expensive, especially for large datasets.
- Requires careful tuning of hyperparameters (e.g., kernel, C, gamma).
- Can be sensitive to the choice of kernel.

#### Pseudo Code for SVM
```
**Step 1: Input Data**
- Given:
  - Feature matrix `X` (size: N x d, where N is the number of samples and d is the number of features).
  - Labels `y` (size: N, with each label `y_i` ∈ {+1, -1} for binary classification).

**Step 2: Initialize Parameters**
- Initialize:
  - Weight vector `w` (size: d, initialized to zeros).
  - Bias term `b` (scalar, initialized to zero).
  - Regularization parameter `C` (controls the tradeoff between maximizing the margin and minimizing classification errors).

**Step 3: Training the SVM**
- Repeat until convergence or a maximum number of iterations:
  1. For each sample `(x_i, y_i)` in `X`:
     - Compute the decision value:  
       `decision = y_i * (w • x_i + b)`
     - If `decision >= 1`:
       - The sample is correctly classified, update `w` and `b` minimally:
         ```
         w = w - η * (2 * λ * w)
         ```
     - If `decision < 1`:
       - The sample is misclassified, update `w` and `b`:
         ```
         w = w - η * (2 * λ * w - y_i * x_i)
         b = b + η * y_i
         ```
       Where:
       - `η`: Learning rate.
       - `λ`: Regularization strength (related to `C`).

**Step 4: Predicting with SVM**
- For a new data point `x`:
  - Compute the decision function:
    ```
    y_pred = sign(w • x + b)
    ```
  - If `y_pred > 0`, classify as `+1`.
  - If `y_pred < 0`, classify as `-1`.

**Step 5: Output**
- Return the trained weight vector `w` and bias `b`.
```

#### **Example (Scikit-learn)**
```python
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, random_state=42, class_sep=1.5)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model with RBF kernel
model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

#### Does SVM Work for Both Classification and Regression?

Yes, **Support Vector Machines (SVM)** work for both **classification** and **regression** tasks, though their implementations and objectives differ.

##### **1. SVM for Classification**

- **Purpose**: To separate data points into different classes by finding the optimal hyperplane that maximizes the margin between classes.
- **Key Characteristics**:
  - **Hyperplane**: A decision boundary that separates classes.
  - **Margin**: The distance between the hyperplane and the nearest data points (support vectors) from either class.
  - **Types**:
    - **Linear SVM**: Used for linearly separable data.
    - **Non-Linear SVM**: Uses kernel functions (e.g., RBF, polynomial) to handle non-linearly separable data.
- **Outputs**:
  - Class labels (e.g., `+1`, `-1`).
  - Probabilities (using extensions like Platt Scaling).

##### **2. SVM for Regression (Support Vector Regression, SVR)**

- **Purpose**: To predict continuous values by fitting a model within a certain margin of tolerance, called the **epsilon-tube**.
- **Key Characteristics**:
  - Instead of minimizing classification error, SVR minimizes the **distance between predicted and actual values**, allowing for a margin of tolerance (`epsilon`).
  - Data points within the `epsilon` margin are ignored during training.
  - Uses support vectors to define the regression boundary.
- **Types**:
  - **Linear SVR**: Fits a linear regression line.
  - **Non-Linear SVR**: Uses kernels to capture non-linear relationships.
- **Outputs**:
  - Continuous values (e.g., predicting house prices).

---

##### **Key Differences Between SVM for Classification and Regression**

| Aspect                  | SVM for Classification                | SVM for Regression                |
|-------------------------|----------------------------------------|------------------------------------|
| **Objective**           | Maximize margin between classes        | Fit a line/curve within a margin  |
| **Output**              | Discrete class labels (`+1`, `-1`)     | Continuous values                 |
| **Loss Function**       | Hinge loss                             | Epsilon-insensitive loss          |
| **Margin**              | Separates classes                      | Allows for deviations within `epsilon` |
| **Use Cases**           | Spam detection, image classification   | House price prediction, stock forecasting |

---

##### **Example of SVR (Scikit-learn)**

```python
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, len(X))

# Train SVR model with RBF kernel
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X, y)

# Predict and plot
y_pred = model.predict(X)
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='SVR Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Support Vector Regression')
plt.show()
```
##### When to Use SVM for Classification vs. Regression
- Use SVM for classification when your target variable consists of discrete labels.
- Use SVR when your target variable is continuous, and you want a robust regression model.


### 6. K Nearest Neighbors (KNN)


### 7. Naive Bayes


### 8. Lasso and Ridge Regression


### 9. Ensemble Methods
details in ml prep/notes/ml prep 101/chapter 2.1-Ensemble Methods.md

### 10. Neural Networks
details in ml prep/notes/ml prep 101/chapter 2.2-Neural Networks.md

