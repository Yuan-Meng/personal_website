---
title: "An Intro to Embedding-Based Retrieval"
date: 2024-06-22
math: true
tags:
    - embedding
    - information retrieval
    - vector-based search
categories:
- papers
keywords:
    - embedding, information retrieval, vector-based search
include_toc: true
---

# So, What is an Embedding?

[Embedding](https://en.wikipedia.org/wiki/Embedding) is a classic idea in mathematical topology and machine learning (click ▶ for definitions). You can think of embeddings as a special type of vectors. 

<details>
  <summary><b>Mathematical topology</b></summary>
  <p>An embedding is a function $f: X \rightarrow Y$ between two topological spaces that is injective (one-to-one) and a homeomorphism onto its image, thus preserving the topological properties of $X$ within $Y$. This concept ensures that $X$ is mapped into $Y$ in a way that $X$'s structure—such as its continuity and open/closed set properties—is maintained exactly within the host structure $Y$</p>
</details>


<details>
  <summary><b>Machine learning</b></summary>
  <p>An embedding is a transformation $f: X \rightarrow \mathbb{R}^n$ that maps entities from a high-dimensional or abstract space $X$ (e.g., words, images, or graph nodes) to vectors in a lower-dimensional, continuous vector space $\mathbb{R}^n$. This mapping aims to preserve relevant properties of the original entities, such as similarity or relational structure, thereby enabling more effective computational manipulation and analysis.</p>
</details>

A vector $\mathbb{R}^n$ is an ordered list of numbers, which can represent almost everything: 

- A geographic location, described by `[latitude, longitude]`.
- A desk, characterized by `[height, area, color, other attributes]`.
- A photo, consisting of channel values for each pixel, `[[r, g, b], ...]`.

In traditional machine learning, each training example is described by a feature vector, usually consisted of hand-crafted features. For example, in spam classification, input text features might include the presence of a "$" 🤑 symbol in the email content, whether the subject line is in all CAPITAL LETTERS, and so on. 

All vectors are not embeddings. For vectors to be considered as embeddings, similar entities in the real world must also be close in the embedding space, according to some distance function (e.g., Euclidean distance, Jaccard similarity, dot product, cosine similarity, etc.) --- a property that regular vectors do not always satisfy. Consider the example from [*Machine Learning Design Patterns*](*https://www.oreilly.com/library/view/machine-learning-design/9781098115777/*): 6 one-hot vectors are used to represent the number of babies in one birth. While singles are more similar to twins than they are to quintuplets, the cosine similarity between the single vector (`[1, 0, 0, 0, 0, 0]`) and the twin vector (`[0, 1, 0, 0, 0, 0]`) is 0, the same as that between the single vector and the quintuplets vector (`[0, 0, 0, 0, 0, 1]`). After all, these one-hot vectors are orthogonal to one another. Since one-hot vectors do not capture similarities between categories, they are *not* embeddings.

We can, however, use a lower-dimensional vector to represent each class label (column 3 in the table below), such that more similar labels are closer to one another. 

| Plurality       | One-hot encoding        | Learned encoding |
|-----------------|-------------------------|------------------|
| Single (1)      | [1,0,0,0,0,0]           | [0.4, 0.6]       |
| Multiple (2+)   | [0,1,0,0,0,0]           | [0.1, 0.5]       |
| Twins (2)       | [0,0,1,0,0,0]           | [-0.1, 0.3]      |
| Triplets (3)    | [0,0,0,1,0,0]           | [-0.2, 0.5]      |
| Quadruplets (4) | [0,0,0,0,1,0]           | [-0.4, 0.3]      |
| Quintuplets (5) | [0,0,0,0,0,1]           | [-0.6, 0.5]      |


{{< figure src="https://www.dropbox.com/scl/fi/5xi8v3omgam3126dr0ahi/Screenshot-2024-04-21-at-2.58.25-PM.png?rlkey=zt24ehipliz2c1f2o8pol4zax&st=911v312w&raw=1" width="1000">}}

The million-dollar question is, how do we learn the "proper" lower-dimensional representation of a class or an entity in the embedding space? This is the exact type of problems that ["metric learning"](https://paperswithcode.com/task/metric-learning#:~:text=The%20goal%20of%20Metric%20Learning,been%20developed%20for%20Metric%20Learning.) tries to solve. Typically, we need to mine the raw training data (e.g., search or feed logs) to construct positive/negative pairs or triplets, initialize with each entity's embedding with random values, and gradually pull similar entities (e.g., a user and an item on which they converted) closer and push dissimilar entities apart (e.g., a user and an unengaged item) using some contrastive objective (e.g., contrastive loss, triplet loss, Noise Contrastive Estimation, etc.). You can read Lilian Weng's wonderful [blog post](https://lilianweng.github.io/posts/2021-05-31-contrastive/) for more details. 

With embeddings for entities we care about, we can answer many questions --- e.g.,

- **First-pass ranking**: For a  user, how do we sift through a vast inventory of products/movies/posts/people/etc. to find items they may show interests in?
- **Passage retrieval/semantic search**: Given a natural language question, how do we retrieve passages that may contain the answer?

All this boils down to the **top-$k$ retrieval problem**: Given a query point $q$, how do we find top-$k$ document points $u \in \mathcal{X}$ that are most similar to it, so that we can minimize a distance function $\delta$ calculated on entity embeddings? 

$$\mathop{\text{arg min}}\limits^{(k)}_{u \in \mathcal{X}} \delta(q, u).$$

# Top-$k$ Retrieval Problem

The startup Pinecone is a lead provider of web-scale commercial top-$k$ retrieval solutions. In this blogpost, I review key ideas from the new monograph [*Foundations of Vector Retrieval (2024)*](https://arxiv.org/abs/2401.09350) by Sebastian Bruch, a Principal Scientist at Pinecone.

## Flavors of Distance Functions

Finding top-$k$ points "closest" to the query point first requires a distance function. The figure below shows the 3 most popular choices (↓: minimize; ↑: maximize).

{{< figure src="https://www.dropbox.com/scl/fi/hx955sne496k99umhqi7f/Screenshot-2024-04-21-at-4.24.45-PM.png?rlkey=sxnuzw2ve6tye9qjf1g8jrt3v&st=0og5opxa&raw=1" width="1000">}}


- **Euclidean distance** (↓): Straight line  from each point to the query point;
- **Cosine similarity** (↑): 1 - angular distance from each point to the query point;
- **Inner product** (↑): Imagine a hyperplane that is orthogonal to the query point and passes through a given point — the inner product between this point and the query is given by the shortest distance from this hyperplane to the query.

The 3 distance functions result in 3 flavors of vector retrieval: $k$-Nearest Neighbor Search ($k$-NN) base on Euclidean, $k$-Maximum Cosine Similarity Search ($k$-MCS) based on cosine, and $k$-Maximum Inner Product Search ($k$-MIPS) based on inner product. Which one to choose depends on the pros \& cons for your specific use case.

<table style="width:100%; border: 1px solid black; border-collapse: collapse;">
    <tr style="background-color: #f2f2f2;">
        <th style="width: 15%; padding: 10px; border: 1px solid black;">Method</th>
        <th style="width: 25%; padding: 10px; border: 1px solid black;">Domain</th>
        <th style="width: 30%; padding: 10px; border: 1px solid black;">Pros</th>
        <th style="width: 30%; padding: 10px; border: 1px solid black;">Cons</th>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid black;"><strong>$k$-NN</strong></td>
        <td style="padding: 10px; border: 1px solid black;">General, Clustering</td>
        <td style="padding: 10px; border: 1px solid black;">
            - Simple and effective in metric spaces<br>
            - Good if you care about literal distances
        </td>
        <td style="padding: 10px; border: 1px solid black;">
            - Points are too far apart in high-dimensional spaces<br>
            - Sensitive to scale
        </td>
    </tr>
    <tr style="background-color: #f9f9f9;">
        <td style="padding: 10px; border: 1px solid black;"><strong>$k$-MCS</strong></td>
        <td style="padding: 10px; border: 1px solid black;">Text processing, Sentiment analysis</td>
        <td style="padding: 10px; border: 1px solid black;">
            - Ignores vector magnitude; good for normalized vectors<br>
            - Ideal for angle-based similarity (e.g., texts)
        </td>
        <td style="padding: 10px; border: 1px solid black;">
            - Not appropriate if magnitude is meaningful<br>
            - Expensive to compute for large datasets
        </td>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid black;"><strong>$k$-MIPS</strong></td>
        <td style="padding: 10px; border: 1px solid black;">Recommendation systems, Collaborative filtering</td>
        <td style="padding: 10px; border: 1px solid black;">
            - Captures both vector magnitude and direction<br>
            - Works well in high-dimensional spaces
        </td>
        <td style="padding: 10px; border: 1px solid black;">
            - Non-metric (e.g., inner product with self may not be the largest)<br>
            - Expensive to compute
        </td>
    </tr>
</table>

## Approximate Retrieval Algorithms

Regardless of the method, exhaustively comparing the query vector with every other vector to find top $k$ is computationally expensive. To avoid exhaustive search, many approximate retrieval algorithms have been proposed, trading accuracy for speed.

### Branch-and-Bound

Branch-and-bound is the earliest algorithm for top-$k$ vector retrieval. This class of algorithms proceed in two phases: Recursively **partitioning** the vector space $\mathcal{X}$ into smaller regions and marking region boundaries, storing them in a binary search tree (BST), and **searching** only regions that could contain the top $k$ solution set. 

{{< figure src="https://www.dropbox.com/scl/fi/fxy7jr9oo6mac3j8ki0is/Screenshot-2024-04-21-at-5.56.57-PM.png?rlkey=ch8lg4imvf6xwrzecnwd662v5&st=wmmto60f&raw=1" width="400">}}


- **Partitioning**: The original vector space is partitioned into a balanced binary search tree (BST), where each internal node has a decision boundary
    - To begin, partition the vector space into regions $\mathcal{R}_l$ and $\mathcal{R}_r$
    - Exhaustively search $\mathcal{R}_l$ to find the optimal point $u_l^\ast$ that minimizes the distance to the query vector $q$, $\delta(q, u_l^\ast)$ 👉 *certify* $u_l^\ast$ is indeed optimal
        - If $\delta(q, u_l^\ast) < \delta(q, \mathcal{R_r})$: Found optimal point and can discard points in $\mathcal{R}_r$
            - $\delta$-ball centered at $q$ with radius $\delta(q, u_l^\ast)$ is contained entirely in $\mathcal{R}_l$, so no point from $\mathcal{R}_r$ has have shorter distance to $q$ than $u_l^\ast$
        - If $\delta(q, u_l^\ast) \geq \delta(q, \mathcal{R_r})$: Also search $\mathcal{R}_l$ and compare the solution with $u_l^\ast$
            - Backtrack to the parent of $\mathcal{R}_l$ and compare $\delta(q, u_l^\ast)$ with the distance of $q$ with the decision boundary
- **Retrieval**: Similar to partitioning, but needs more care during traversal
    - Traverse from root to leaf; each node determines if $q$ belongs to $\mathcal{R}_l$ or $\mathcal{R}_r$
    - Once we find the leaf region that contains $q$, we find the candidate vector $u^\ast$ 👉 backtrack and certify that $u^\ast$ is indeed optimal
        - At each internal node, compare the distance between $q$ and the current candidate with the distance between $q$ and the region on the other side of the boundary 👉 prune or search for better candidates
    - Terminate when back at root 👉 all branches are either pruned or certified 


Different instantiations of branch-and-bound algorithms differ in how they split a collection or carry out certification. In general, however, brand-and-bound algorithms work poorly on high-dimensional data because the number of leaves that may be visited during certification grows exponentially with the embedding dimension $d$.

### Clustering
Why not cluster vectors first, so that at retrieval time, we first find the cluster to which the query vector $q$ belongs, and then search top $k$ within that cluster?

- **Locality Sensitive Hashing (LSH)**: Hash each vector in $\mathbb{R}^d$ into a single bucket $h: \mathbb{R}^d \rightarrow \[b\]$ 👉 during retrieval, exhaustively search the bucket where $q$ is put into, assuming $q$ is hashed into the same bucket as its nearest neighbors

# Optimization Techniques

## Vector Compression

# References

## Books/Papers
1. Bruch, S. (2024). Foundations of Vector Retrieval. [arXiv:2401.09350](https://arxiv.org/pdf/2401.09350).

## Blog Posts
2. [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) by Lilian Weng (2021)
