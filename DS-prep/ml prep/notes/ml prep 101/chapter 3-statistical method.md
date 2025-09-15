## Statistical Terms

### 1. Impurity Measures for Decision Tree
Impurity measures are metrics used in decision trees to evaluate how well a split at a node separates the data into distinct classes or reduces variance. These measures help the algorithm decide which feature and threshold to use when splitting the dataset at each step.

#### Gini Impurity (Classification)
Gini impurity measures how often a randomly chosen element would be incorrectly classified if labeled according to the distribution in a subset.

**Formula**:  
$$
G = 1 - \sum_{i=1}^{n} p_i^2
$$  
Where:  
- \( p_i \) is the proportion of class \( i \) in the subset.  
- \( n \) is the total number of classes.
- Range=[0, 0.5]
  - 0: Pure node (only one class present).
  - 0.5: Maximum impurity for a binary classification.

##### Example of Gini Impurity Split: 
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

---

#### Entropy and Information Gain (Classification)
**Entropy** measures the disorder or randomness in a subset. Lower entropy means greater purity.

**Entropy formula**:  
$$
H = -\sum_{i=1}^{n} p_i \log_2(p_i)
$$

Where:
- p_i is the proportion of data points belonging to class i in the node.

- Range: [0, log2(C)] (C = number of classes)
  - 0: Pure node (only one class present).
  - Higher entropy indicates more uncertainty.

**Information Gain** is the reduction in entropy from a split:  
$$
\text{IG} = H(\text{parent}) - \left[ \frac{N_{\text{left}}}{N} H(\text{left}) + \frac{N_{\text{right}}}{N} H(\text{right}) \right]
$$  
Where:  
- H(parent) is the entropy of the original dataset.  
- H(left) and H(right) are the entropies of the left and right subsets after the split.  
- N_{left}, N_{right}, and N are the sizes of the left, right, and total subsets.

---

#### Mean Squared Error (Regression)
For regression tasks, the goal is to minimize the variance within each subset after the split.

**Formula**:  
$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2
$$  
Where:  
- y_i is the actual value.
- ȳ is the mean target value in the node.
- N is the number of data points in the node.

Lower MSE indicates a better split.



