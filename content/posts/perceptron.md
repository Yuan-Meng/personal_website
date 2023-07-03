---
title: Teaching A Peceptron to See 
date: 2023-07-02
math: true
tags:
    - DL foundations
categories:
- tutorials
keywords:
    - DL foundations
include_toc: true
---

Ten years after the ImageNet Challenge thawed the last AI winter, ChaptGPT and generative AI have become part of our everyday life and colloquial, like (almost) no one has imagined just 2 years back. As increasingly more folks aspire to foray into the field of ML/AI, I can't help but think about a lesson from my guitar teacher:

> Everyone wants to start playing the songs they love right off the bat, but without nailing seemingly "boring" building blocks such as scales, harmonies, and rhythms, the songs you love will sound like a nightmare...

Not to sound discouraging (yup, when people say "not to do X", they're doing exactly X...), but I think those who can ride the AI wave are those who take their time to build from the foundations. Let's begin with the oldest and simplest neural net: The *Perceptron,* ([Rosenblatt, 1959](https://psycnet.apa.org/record/1959-09865-001)).


# Seeing "7" vs. "L"

Which of 2 images below is a "7" and which one is an "L"?

{{< figure src="https://www.dropbox.com/s/g01sb2n1eelrum7/7_or_l.png?raw=1" width="600">}}

To us humans, answering this question is a piece of cake: The left image is "L" and the right "7". But how is it that we know (except for saying "intuitions")?

## How do humans see, in plain English?

We can divide each image into 4 [pixels](https://en.wikipedia.org/wiki/Pixel) and check whether each pixel is filled (1) or empty (0), based on which we can come up with rules to classify images:

- "L": top right is empty (0) +  lower left is filled (1)
- "7": lower left is empty (0) + top right is filled (1)

## But how can a machine see?

However, machines don't have eyes and, more importantly, it'd be cumbersome to write bespoke rules for each and every use case. As a common strategy in machine learning, we can *somehow* map an input vector (in this case, a 4-vector with pixel values) to an output vector (a 2-vector with probabilities of the 2 classes).

{{< figure src="https://www.dropbox.com/s/4j6l220txor8bkj/vec_2_output.png?raw=1" width="600">}}

- $\mathbf{x}$: a vector encoding the value of each pixel 👉 already known (the green arrow indicates the direction from the first to the last pixel) 
- $\mathbf{w}$: a vector encoding the weight of each pixel 👉 unknown

There are an infite number of rules that can map $\mathbf{x}$ and $\mathbf{w}$ to class labels. Let's use the sign of their dot product $\mathbf{w}^T \mathbf{x}$ (`np.dot(x, w)`) to determine the class label 👉 negative: 0 ("L") vs. positive: 1 ("7"); which class is positive is arbitrary.

In our case, weights of pixels 1 and 3 can be 0 --- these 2 are the same regardless of the label, so they are not really "pulling the punches" towards classification. Pixels 2 and 4 are likely negative in the "L" picture but positive in the "7" picture --- which are in the direction of the desired dot product sign in each image. 

## Wait... how do we "learn" the weights?

Ummm, aren't we back the realm of relying on human eyes to see the images and human brains to hand-engineer rules? We don't have to --- we can initialize $\mathbf{w}$ with 4 random floats, say `[0.5, 0.9, -0.3, 0.5]`, and "learn from mistakes".

- Attempt #1: Say the first training example is an "L". Using the random weights above, we get $\mathbf{w}^T \mathbf{x} = 0.5 \times 1 + 0.9 \times 0 -0.3 \times 1 + 0.5 \times 1 = 0.7$. Because the dot product is positive, we predict "7", which is *incorrect*.

    {{< figure src="https://www.dropbox.com/s/8145o7xczyogveb/class_l.png?raw=1" width="250">}}

    - Correction: Because the dot product is *too large* (should be < 0 to classify the image as "L"!), we *decrease* the weights $\mathbf{w}$. 
    - New weights: A crude way to do so is subtract $\mathbf{x}$ from $\mathbf{w}$ (let's save gradient descent for the future...) 👉 $\mathbf{w} = \mathbf{w}-\mathbf{x} \\ = [0.5 - 1, 0.9 - 0, -0.3 - 1, 0.5 - 1] \\ = [-0.5, 0.9, -1.3, -0.5]$

- Attempt #2: The second training example is a "7". Using updates weights, we get $\mathbf{w}^T \mathbf{x} = -0.5 \times 1 + 0.9 \times 1 -1.3 \times 0 - 0.5 \times 1 = -0.1$. Since the dot product is negative, we predict "L", which is *incorrect* again.

    {{< figure src="https://www.dropbox.com/s/5w9x132qak8coy2/class_7.png?raw=1" width="250">}}

    - Correction: Because the dot product is *too small* (should be > 0 to classify the image as "7"), we *increase* the weights $\mathbf{w}$. 
    - New weights: $\mathbf{w} = \mathbf{w}-\mathbf{x} \\ = [-0.5 + 1, 0.9 + 1, -1.3 + 0, - 0.5 + 1] \\ = [0.5, 1.9, -1.3, 0.5]$

- Final & successful attempt: The third training example is an "L". Using the newest weights, we get $\mathbf{w}^T \mathbf{x} = 0.5 \times 1 + 1.9 \times 0 -1.3 \times 1 + 0.5 \times 1 = -0.3$. Since the dot product is negative, we correctly predict "L"!

    {{< figure src="https://www.dropbox.com/s/8145o7xczyogveb/class_l.png?raw=1" width="250">}}

To achieve good performance, neural networks usually train for (far) more than 3 examples. Early stopping can be applied when model hasn't improved for a while.

> (The [slides](https://dl101-perceptron.netlify.app) above are adapted from the Spring 2022 offering of UC Berkeley's CogSci 131 taught by me.)

# Code Up A Perceptron

In the L vs. 7 toy example, each image only has 4 pixels. To practice what we've learned, let's solve a slightly more complex problem: Classifying images of handwritten digits (e.g., a 28 $\times$ 28 image of "1") into numerical labels (1).

{{< figure src="https://www.dropbox.com/s/t47mo4pezekikii/MnistExamplesModified.png?raw=1" width="600">}}

> (All images and code snippets used below can be found in this [repo](https://github.com/Yuan-Meng/dl101_notebooks).)



## Read image inputs

If using a convolutional neural net (check out this great intuitive [explanation](https://www.youtube.com/watch?v=KuXjwB4LzSA) by 3Blue1Brown), you may want to keep input images as 28 $\times$ 28 matrices. In this case, however, we can just flatten each image into a 28 $\times$ 28 = 784 vector. 

```python
# dimension of all images
DIM = (28, 28)

# flattened image dimension
N = DIM[0] * DIM[1]

def load_image_files(n, path="images/"):
    """loads images of given digit and returns a list of vectors"""
    # initialize empty list to collect vectors
    images = []
    # read files in the path
    for f in sorted(os.listdir(os.path.join(path, str(n)))):
        p = os.path.join(path, str(n), f)
        if os.path.isfile(p):
            i = np.loadtxt(p)
            # check image dimension
            assert i.shape == DIM
            # flatten i into a single vector
            images.append(i.flatten())
    return images
```

## Classify 0 vs. 1

As a starter, let's classify 0 vs. 1. First off, let's translate some conceptual operations mentioned in the 7 vs. L example into helper functions.


- Use dot product to classify one image: Given a *known* weight vector, we can compute $\mathbf{w}^T \mathbf{x}$ and return 0 or 1 as the classification result

```python3
def classify_image(W, image):
    """use weight matrix W to determine digit in image"""
    # if dot product > 0, return 1; otherwise 0
    y = (np.dot(image, W) > 0).astype(int)
    return y
```

- Update weights upon feedback from each true label: Subtract $\mathbf{x}$ from $\mathbf{w}$ in the case of false positive (should be 0 but predicted 1) and add $\mathbf{x}$ to $\mathbf{w}$ in case of false negative (should be 1 but predicted 0)

```python3
def update_weights(W, images, labels):
    """updates weight matrix W based on training (image,label) pairs"""

    # loop through images and labels
    for image, label in zip(images, labels):
        # predict image label
        pred_label = classify_image(W, image)
        # update only if prediction is wrong
        if pred_label != label:
            # predicted 0 but label is 1
            if label == 1:
                W += image
            # predicted 1 but label is 0
            else:
                W -= image

    # return updated weights
    return W
```

Finally, we can write the training function that takes 4 arguments: Name of 1st digit (class 0), name of 2nd digit (class 1), # of training examples per class, and # of epochs (i.e., how many times we want to repeat the full training loop). 

```python3
def train_perceptron(digit1, digit2, n_samples=25, epoch=200):
    """train perceptron on (image, label) pairs for given steps"""

    # load images of two digits to compare
    img1 = load_image_files(digit1)
    img2 = load_image_files(digit2)

    # initialize empty list to collect accuarcies
    accuracies = []
    # initilaize random weights from standard normal
    W = np.random.normal(0, 1, size=N)

    # iterate through each epoch
    for i in tqdm(range(epoch)):

        # randomly sample images for '0'
        sample1 = random.sample(img1, n_samples)
        # randomly sample images for '1'
        sample2 = random.sample(img2, n_samples)

        # train on chosen samples
        W = update_weights(W, sample1 + sample2, [0] * n_samples + [1] * n_samples)
        # evaluate performance on all images
        accuracy = compute_accuracy(W, img1 + img2, [0] * len(img1) + [1] * len(img2))
        accuracies.append(accuracy)

    # return accuracies and final weights
    return accuracies, W
```

As you may have noticed from above, after each loop, we check the model accuracy, which is defined as # of correctly classified examples / # of samples classified. 

```python3
def compute_accuracy(W, images, labels):
    """computes accuracy on a list of images"""
    # initialize count at 0
    n_correct = 0
    # loop through image list
    for i, image in enumerate(images):
        # increment count by each time we correctly classify an image
        n_correct += (labels[i] == classify_image(W, image)).astype(int)

    # accuracy is total correct / total
    return n_correct / len(images)
```

Since the Perceptron algorithm is quite simple, training is lightening fast --- we can train for 500 epochs in a matter of minutes. Training GPT may take months.

```python3
# train on 25 examples for given number of epochs
n_samples, epochs = 25, 500
accuracies, trained_W = train_perceptron(0, 1, n_samples, epochs)

# plot accuracy as a function of epoch
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(range(epochs), accuracies)
ax.set_xlabel("epoch", fontsize=20)
ax.set_ylabel("mean accuracy", fontsize=20)
ax.set_title("training accuracy", fontsize=25)
```

We almost achieved perfect (but not 100%) accuracy on 0 vs. 1 classification.

{{< figure src="https://www.dropbox.com/s/x6iymtu1r1bkb1l/accuracy.png?raw=1" width="600">}}


## Interpret learned weights 

People say neural nets are black boxes, which may well be true in the case of complex networks. In our example, learned weights may simply represent the *exclusive disjunction* of where "0" pixels and "1" pixels are in the input images.

```python3
# reshape weights back to 28 * 28
trained_W_2d = np.reshape(trained_W, DIM)

# plot weight matrix
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(trained_W_2d)
ax.axis("off")
```

{{< figure src="https://www.dropbox.com/s/xsnzjcmmp8lpvst/learned_weights.png?raw=1" width="300">}}


## Generalize to all image pairs 

Finally, if we can classify 0 vs. 1 and check model performance, we can do the same for each digit pair (e.g., 0 vs. 1, 1 vs. 2...). The code below trains one classifier for each digit pair and track each classifier's accuracy after the last epoch. 

```python3
def get_accuracy_matrix(n_num):
    """get maxtrix with classification accuracy"""

    # initialize matrix with all 1's
    acc_matrix = np.ones([n_num, n_num])

    # loop through digit pairs
    for digit1 in range(n_num - 1):
        for digit2 in range(digit1 + 1, n_num):
            # get accuracies after 100 epochs
            acc, _ = train_perceptron(digit1, digit2, 25, 100)
            # replace value in matrix with final accuracy
            acc_matrix[digit1, digit2] = acc[-1]
            acc_matrix[digit2, digit1] = acc[-1]

    # return matrix
    return acc_matrix


def plot_accuracy_matrix(acc_matrix, n_num):
    """plot classification accuracy using a heatmap"""

    # generate heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    g = ax.imshow(acc_matrix, cmap="Blues")
    ax.set_xticks(range(0, n_num))
    ax.set_yticks(range(0, n_num))
    ax.set_title("accuracy matrix", fontsize=20)

    # add color bar
    fig.colorbar(g, ax=ax)


# plot accuracy between each digit pair 
acc_matrix = get_accuracy_matrix(10)
plot_accuracy_matrix(acc_matrix, 10)
```

As we can imagine, 5 and 8 are pretty hard to tell apart as written digits, so are 7 vs. 9 and 6 vs. 9, but the rest of digit pairs have high accuracy. 

{{< figure src="https://www.dropbox.com/s/2wswgf7xg769x2x/acc_matrix.png?raw=1" width="500">}}

