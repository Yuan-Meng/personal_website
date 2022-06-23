---
title: Code ML Algorithms From Scratch 
date: 2022-06-22
math: true
tags:
    - ML
    - data science
    - algorithms
categories:
- tutorials
keywords:
    - ML, data science, algorithms
include_toc: true
---


Coding interviews can mean different things for "traditional" software engineers (back-end, front-end, full-stack, etc.) and engineers with a machine learning focus. Apart from LeetCode-style questions, ML engineers (as well as applied scientist, research engineer, and, occasionally, machine learning data scientists) may be asked to implement a classic ML algorithm from scratch during an interview. 

This may sound scary if you've only used libraries to train models without understanding how learning algorithms work under the hood. Moreover, there are way too many algorithms to memorize. The good news is, there are are only 4 algorithms: **Linear regression**, **logistic regression**, **k-nearest neighbors** (k-NN), and **k-means**. Let's implement each of them using the popular Scikit-Learn library or vanilla NumPy (you can PyTorch in interviews if you're super familiar with the syntax).

# Linear Regression

Linear regression uses a vector of real-valued inputs $\mathbf{x} \in \mathbb{R}^D$ (e.g., age, years of education, one-hot encoded job categories) to predict a real-value output $y \in \mathbb{R}$ (e.g., income): $\hat{y_i} = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \ldots + b$ (predicted income for person $i$). 

In the formula above, $\mathbf{w}$ are "regression coefficients" (statistics) or "weights" (ML), which quantify how much each feature impacts the outcome with all else being held constant. The bias term $b$ ("intercept" in stats) is the outcome value when all features are 0 --- or, we can understand it as a force that moves $\hat{y}$ up and down.

We can use several methods to solve for $\mathbf{w}$ and $\mathrm{b}$, such as Ordinary Least Squares (OLS) and gradient descent. Since the latter method applies to a wide range of supervised learning algorithms, not just linear regression, I'll implement it first.

## Gradient Descent 

When using gradient descent, we can initialize $\mathbf{w}$ and $b$ with random values, get terrible results at first, and then gradually learn from our mistakes. If using gradient descent, below are the building blocks of a linear regressor:

1. **Cost function**: Mean squared error (MSE), $J(\mathbf{w}, b) = \frac{\sum_{i}^{n} (y_i - \hat{y_i})^2}{n}$, can measure how "off" the model predictions are. $y_i$ is the observed value of observation $i$, $\hat{y_i}$ is the predicted value of $i$, and $n$ is the number of observations in the dataset. The objective is to find values of $\mathbf{w}$ and $b$ that minimize $J(\mathbf{w}, b)$.
2. **Optimization routine**: If you wanna brush of your memory of gradient descent, I strongly suggestion YouTube videos by [StatQuest](https://youtu.be/sDv4f4s2SB8) and [3Blue1Brown](https://youtu.be/IHZwWFHWa-w). 

    A gradient is the partial derivative of a function with respect to a parameter. For visualization purposes, say we only have one parameter and the picture below shows how the cost function changes with different parameter values. If the gradient is negative, we wanna move a bit to the right; if positive, a bit to the left. When the gradient is 0, it means we're at a minimum, global or local. 

    {{< figure src="https://www.dropbox.com/s/rn0va0kzagzhu6g/gradient.jpg?raw=1" width="400">}}

    Luckily, we don't have to code up different scenarios: By **subtracting** gradients from parameter values, we'll naturally move to the right when gradients are negative and to the left when they're positive. One thing to note is that if we move too much at a time, we may well shoot past the minima --- instead, we can adjust the step size using a learning rate parameter, which is usually a small number (e.g., 0.01, 0.001). However, if the learning rate is too small (e.g., 0.000001), training can take a long time and we may get stuck in local minima. We can use cross-validation to find an ideal learning rate.
    
    For each set of parameter values, each observation has its own gradients. In canonical gradient descent, we calculate the total gradients over all observations. Let $\mathbf{X}$ be the feature matrix of $n$ observations and $\mathbf{y}$ the outcome vector; below are gradients with regard $\mathbf{w}$ and $b$ (see the [derivation](https://www.analyticsvidhya.com/blog/2021/04/gradient-descent-in-linear-regression/#:~:text=Gradient%20Descent%20Algorithm.-,Gradient%20Descent%20Algorithm,a%20smaller%20number%20of%20iterations.&text=For%20some%20combination%20of%20m,us%20our%20best%20fit%20line.)):
    - With regard to $\mathbf{w}$: $\frac{\partial J(\mathbf{w}, b)}{\partial \mathbf{w}} = \frac{-2 \mathbf{X}^T \cdot (\mathbf{y}-\hat{\mathbf{y}})}{n}$
    - With regard to $b$: $\frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{-2 \sum_{i}^{n} y_i - \hat{y_i}}{n}$
3. **Stopping criteria**: We can stop when gradients are 0 (or sufficiently small), or if we've run the algorithm after a fixed number of epochs.

Scikit-Learn doesn't actually implement gradient descent for linear regression. `SGDRegressor` uses stochastic gradient descent (SGD) to optimize parameter values. Instead of using all observations each time, SGD randomly chooses one observation per epoch to compute gradients, which speeds up learning and introduces randomness to the learning process. For large datasets, this added randomness is useful for avoiding getting stuck at local minima. For small datasets, it's a nightmare.

```python
# import libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

# load the diabetes dataset for demonstration
X, y = datasets.load_diabetes(return_X_y=True)

# use 80% for training and 20% for test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create a new model
lr = SGDRegressor(learning_rate="optimal", max_iter=10000)

# fit model to training data
lr.fit(X_train, y_train)

# use fitted model to predict test data
y_pred = lr.predict(X_test)

```

**Spoiler**: The diabetes dataset only has 442 data points, which turns out far from enough for the SGD regressor to converge💀.


In interviews, you're most likely not allowed to use high-level libraries such as Scikit-Learn. We can write a custom `LinearRegression` class using just NumPy (for vectorized operations). The example below is adapted from [GeeksforGeeks]( https://www.geeksforgeeks.org/linear-regression-implementation-from-scratch-using-python/).


```python
class LinearRegression:
    def __init__(self, lr=0.01, epoch=10):
        # set hyperparameter values
        self.lr = lr  # learning rate
        self.epoch = epoch  # number of epochs

    def fit(self, X, y):
        # number of observations, number of features
        self.n_obs, self.n_features = X.shape

        # initialize weights and bias
        self.w = np.zeros(self.n_features)
        self.b = 0

        # read data and labels from input
        self.X = X
        self.y = y

        # use gradient descent to update weights
        for i in range(self.epoch):
            self.update_weights()

        return self

    def update_weights(self):
        # use current parameters to predict
        y_pred = self.predict(self.X)

        # compute gradients with respect weights
        grad_w = -2 * np.dot(self.X.T, (self.y - y_pred)) / self.n_obs

        # compute gradient with respect to bias
        grad_b = -2 * np.sum(self.y - y_pred) / self.n_obs

        # update parameters by substracting lr * gradients
        self.w = self.w - self.lr * grad_w
        self.b = self.b - self.lr * grad_b

        return self

    def predict(self, X):
        # use current parameters to make predictions
        return X.dot(self.w) + self.b
```

A few things to highlight:

- **Initialization** (`__init__`): In Scikit-Learn, when initializing models, we only provided hyperparameter values. If values are not provided, we use default values. Here we got learning rate (`lr`) and number of iterations (`epoch`).
- **Model fitting** (`fit`): The `.fit` method takes two arguments, the data (a $n \times m$ array, where $n$ is the number of observations and $m$ is the number of features) and the target (a $n$-vector). Modeling fitting doesn't return any values --- rather $\mathbf{w}$ and $b$ are modified *in place* using gradient descent. 

    The helper function `update_weights` is where we implement gradient descent. Note that it calls the `.predit` method each time to generate predictions using current parameter values. After calculating gradients of $\mathbf{w}$ and $b$ by translating the two formulas into code, we finally use them to update the two parameters. This process repeats for the number of epochs specified (`epoch`).
- **Model prediction** (`predict`): The `.predict` method uses the best parameter values we found in `.fit` to generate predictions for new data, $\mathbf{y} = \mathbf{X} 
\cdot \mathbf{w} + b$. 

The root-mean-square error (RMSE, which is the square root of MSE) is a common metric to evaluate regression models. Unlike classification metrics (e.g., precision, recall, F-1 score, AUC), the absolute value of RMSE doesn't tell us much about how good a model is. We need to compare RMSE of several models to find the winner. 

{{< figure src="https://www.dropbox.com/s/ncsm5un9asyynpj/rmse_lr_gd.png?raw=1" width="450">}}

## OLS

Former Quora data scientist Zi Chong Kao has an extraordinary [geometric explanation](https://kaomorphism.com/socraticregression/ols.html) of the Ordinary Least Squares (OLS) method. Like gradient descent, the goal of OLS is to minimize prediction errors. The slight difference is that instead of minimizing MSE, OLS minimizes the sum of squared residuals  (SSR), $\lVert \mathbf{y} - \mathbf{X}\mathbf{\beta} \rVert ^2$. 

Here we're using a new formula $\mathbf{\hat{y}} = \mathbf{X}\mathbf{\beta}$, which is similar to the one before, $\mathbf{\hat{y}} = \mathbf{X} \cdot \mathbf{w} + b$, except that now we added a column all 1's to the beginning of the original feature matrix $\mathbf{X}$. Long story short, this transformation simplifies the closed-form solution of the best $\beta$ (see [OLS in Matrix Form](https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf))): $(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T \mathbf{y}$. Know that $\beta_0 = b$.


Fun fact: OLS is the method used by `LinearRegression` in Scikit-Learn.

```python
# import library
from sklearn.linear_model import LinearRegression
# create a new model
lr = LinearRegression()
# fit model to training data
lr.fit(X_train, y_train)
# use fitted model to predict test data
y_pred = lr.predict(X_test)
```

Counterintuitively, I find gradient descent easier to implement than OLS, mostly due to the transformation (e.g., adding an $n$-vector of 1's to $\mathbf{X}$) required by the latter. The code below is simplified from blogposts by [IBM](https://developer.ibm.com/articles/linear-regression-from-scratch/) and [datascience+](https://datascienceplus.com/linear-regression-from-scratch-in-python/).
```python
import copy

class LinearRegression:
    def __init__(self):
        # no hyperparameters to intialize
        pass

    def fit(self, X, y):
        # read data and labels from input
        self.X = X
        self.y = y

        # create a vector of all 1's to X
        X = copy.deepcopy(X) # keep original X intact
        dummy = np.ones(shape=X.shape[0])[..., None] # vector of 1's
        X = np.concatenate((dummy, X), 1) # add to X

        # use OLS to estimate betas
        xT = X.transpose()
        inversed = np.linalg.inv(xT.dot(X))
        betas = inversed.dot(xT).dot(y)

        # bias is the first column
        self.b = betas[0]
        # weights are the rest
        self.w = betas[1:]

        return self
    
    def predict(self, X):
        return X.dot(self.w) + self.b
```

- **Initialization** (`__init__`): OLS doesn't have any hyperparamters, so we can use `pass` to skip initialization.
- **Model fitting** (`fit`): As mentioned, we need to transform $\mathbf{X}$ by adding a new column, but we don't want to mess with the original matrix. We can create a copy of the original (need the `copy` library) and transform the copy instead. The centerpiece the `.fit` method is just translating $(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T \mathbf{y}$ into code. 

    Unless we also want to transform new data $\mathbf{X}$ when making predictions for observations in it, we can extract the bias ($\beta_0$) and the weights ($\beta_j$, where $j=1, 2, \ldots, m$) from fitted $\beta$ and apply $\mathbf{X}\cdot \mathbf{w} + b$ on the untransformed matrix.
- **Model prediction** (`predict`): This part is the same as in gradient descent.

{{< figure src="https://www.dropbox.com/s/4z9itvfdv8uzl4h/rmse_lr_ols.png?raw=1" width="450">}}

The OLS model is a bit worse than the gradient descent version, but it works faster on small datasets and doesn't require us to find the best learning rate.

# Logistic Regression

Some say the name "logistic regression" is a tad misleading because it's a classification method. However, I think the name appropriately highlights the connection between logistic regression and linear regression. In logistic regression, we still use real-valued features (e.g., income, age, one-hot encoded move genres) to first predict some real-valued output. However, the direct output doesn't carry much meaning --- we want to transform it into a probability ([0, 1]) and make a decision based on the probability ($p \geq 0.5$: will buy popcorn; $p < 0.5$: won't buy popcorn). 

- The part same as in linear regression: $z = \mathbf{w}^T \mathbf{x} + b$
- Transformation using the logistic function: $p = \frac{1}{1 + \exp(-z)}$
- Make a decision: $\hat{y} = 1$ if $p \geq t$ and 0 if $p < t$ ($t$ is the decision threshold)

Rather than using MSE as the cost function, we can use cross entropy ([explained](https://youtu.be/6ArSys5qHAU) by StatQuest): $\frac{\sum_{i}^{n} y_i \log(\hat{y_i})}{n}$, where $y_i$ is the actual outcome (0 or 1) and $\hat{y_i}$ the predicted outcome. We don't want to use classification metrics (e.g., accuracy, precision, recall) as the cost function because they don't have sensitive gradients.

Unlike linear regression, OLS is no longer appropriate for logistic regression because it make predict values greater than 1 or less than 0. We can still use gradient descent to find the best $\mathbf{w}$ and $b$. I'll skip Scikit-Learn because it's just a matter of importing different functions.

First, we can use `make_classification` to generate some toy data:

```python
# create some toy data
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

# use 80% for training and 20% for test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

```

Under cross entropy, the gradients of the parameters are almost the same as when we used MSE, except that now we don't have the scalar 2:

- With regard to $\mathbf{w}$: $\frac{\partial J(\mathbf{w}, b)}{\partial \mathbf{w}} = \frac{-\mathbf{X}^T \cdot (\mathbf{y}-\hat{\mathbf{y}})}{n}$
- With regard to $b$: $\frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{-\sum_{i}^{n} y_i - \hat{y_i}}{n}$

```python
class LogisticRegression:
    def __init__(self, lr=0.01, epoch=10):
        # set hyperparameter values
        self.lr = lr  # learning rate
        self.epoch = epoch  # number of epochs

    def fit(self, X, y):
        # number of observations, number of features
        self.n_obs, self.n_features = X.shape
        # initialize weights and bias
        self.w = np.zeros(self.n_features)
        self.b = 0
        # read data and labels from input
        self.X = X
        self.y = y
        # use gradient descent to update weights
        for i in range(self.epoch):
            self.update_weights()

        return self

    def update_weights(self):
        # use current parameters to predict
        y_pred = self.predict(self.X)
        # calculate gradients with respect to weights
        grad_w = -np.dot(self.X.T, (self.y - y_pred)) / self.n_obs
        # calculate gradients with respect to bias
        grad_b = -np.sum(self.y - y_pred) / self.n_obs
        # update parameters by substracting lr * gradients
        self.w = self.w - self.lr * grad_w
        self.b = self.b - self.lr * grad_b

        return self

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        # linear part
        z = X.dot(self.w) + self.b
        # sigmoid -> transform [0, 1]
        p = self.sigmoid(z)
        # make decisions
        return np.where(p < 0.5, 0, 1)
```

The implementation is similar to that of linear exception, except that we need extra steps to turn real-valued predictions into binary (0 or 1) decisions.

Compared to regression metrics, classification metrics are way more complex. With the same decision threshold, we can derive accuracy, precision, recall, F1-score, and some more obscure ones (e.g., specificity) from the confusion matrix. To find a good threshold, we can use ROC AUC or Precision-Recall AUC. To learn these metrics in detail, I highly recommend StatQuest's [new book](https://www.amazon.com/StatQuest-Illustrated-Guide-Machine-Learning/dp/B09ZCKR4H6). The results look quite good.

{{< figure src="https://www.dropbox.com/s/nu0oz8ejg3n3udo/logistic_gd.png?raw=1" width="400">}}

```
              precision    recall  f1-score   support

           0       0.90      0.97      0.93       108
           1       0.96      0.87      0.91        92

    accuracy                           0.93       200
   macro avg       0.93      0.92      0.92       200
weighted avg       0.93      0.93      0.92       200
```



# K-NN

# K-Means
