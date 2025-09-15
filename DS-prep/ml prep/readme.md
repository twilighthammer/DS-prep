<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>


#### How a Decision Tree Finds the Best Fit

A **decision tree** finds the "best fit" by selecting a feature and a threshold that splits the dataset into two subsets, aiming to maximize the **purity** of the resulting subsets (i.e., make them as homogeneous as possible). This process is repeated recursively to build the tree.

---

### Steps to Find the Best Split:

1. **Choose a Splitting Criterion**:
   The algorithm evaluates potential splits using a metric that measures the "quality" of the split. Common criteria include:
   - **Gini Impurity** (used in classification)
   - **Entropy/Information Gain** (used in classification)
   - **Mean Squared Error (MSE)** (used in regression)

---

### Gini Impurity (Classification)
Gini impurity measures how often a randomly chosen element would be incorrectly classified if labeled according to the distribution in a subset.

**Formula**:  
$$
G = 1 - \sum_{i=1}^{n} p_i^2
$$  
Where:  
- \( p_i \) is the proportion of class \( i \) in the subset.  
- \( n \) is the total number of classes.

A Gini Impurity of 0 means the subset is pure (all instances belong to one class).

---

### Entropy and Information Gain (Classification)
**Entropy** measures the disorder or randomness in a subset. Lower entropy means greater purity.

**Entropy formula**:  
$$
H = -\sum_{i=1}^{n} p_i \log_2(p_i)
$$

**Information Gain** is the reduction in entropy from a split:  
$$
\text{IG} = H(\text{parent}) - \left[ \frac{N_{\text{left}}}{N} H(\text{left}) + \frac{N_{\text{right}}}{N} H(\text{right}) \right]
$$  
Where:  
- \( H(\text{parent}) \) is the entropy of the original dataset.  
- \( H(\text{left}) \) and \( H(\text{right}) \) are the entropies of the left and right subsets after the split.  
- \( N_{\text{left}} \), \( N_{\text{right}} \), and \( N \) are the sizes of the left, right, and total subsets.

---

### Mean Squared Error (Regression)
For regression tasks, the goal is to minimize the variance within each subset after the split.

**Formula**:  
$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2
$$  
Where:  
- \( y_i \) is the actual value.  
- \( \bar{y} \) is the mean of the subset.

---

### Algorithm for Finding the Best Split:
1. **For each feature** in the dataset:
   - Try all possible thresholds (e.g., midpoints between feature values).
   - Split the data into two subsets based on the threshold.
   - Calculate the splitting criterion (e.g., Gini, entropy, MSE).

2. **Choose the split** with the lowest impurity or highest information gain.

3. **Repeat recursively** for each subset to grow the tree.

---

### Example of Gini Impurity Split:
Suppose you have a dataset with two classes: `A` and `B`.

Initial dataset:  
- `4 A` and `4 B` → Gini impurity =  
$$
1 - \left[ \left(\frac{4}{8}\right)^2 + \left(\frac{4}{8}\right)^2 \right] = 0.5
$$

Split into two subsets:  
1. Left: `3 A, 1 B` → Gini =  
$$
1 - \left[ \left(\frac{3}{4}\right)^2 + \left(\frac{1}{4}\right)^2 \right] = 0.375
$$  
2. Right: `1 A, 3 B` → Gini =  
$$
1 - \left[ \left(\frac{1}{4}\right)^2 + \left(\frac{3}{4}\right)^2 \right] = 0.375
$$

Weighted average Gini impurity of the split:  
$$
\text{Gini split} = \frac{4}{8} \times 0.375 + \frac{4}{8} \times 0.375 = 0.375
$$

This split reduces impurity compared to the original dataset (\(0.5 \to 0.375\)).
