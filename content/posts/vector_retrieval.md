---
title: "Sift Through the Haystack: Vector Retrieval"
date: 2024-05-07
math: true
tags:
    - search
    - information retrieval
    - vector-based search
categories:
- papers
keywords:
    - search, information retrieval, vector-based search
include_toc: true
---

# Embeddings, Embeddings Everywhere

Be it a person, a product, a place, a text, an image, or a planet --- virtually all entities you can think of can be represented as a special kind of vectors, called "embeddings." [Embedding](https://en.wikipedia.org/wiki/Embedding) is a classic idea in mathematical topology and machine learning (click ▶ for definitions), recently made popular by the rise of foundation models that are exceptionally good at embedding texts, images, and videos to empower downstream use cases, such as embedding-based retrieval, text/image/video understanding, and deep learning rankers with embedding features, to name a few. 

<details>
  <summary><b>Mathematical topology</b></summary>
  <p>An embedding is a function $f: X \rightarrow Y$ between two topological spaces that is injective (one-to-one) and a homeomorphism onto its image, thus preserving the topological properties of $X$ within $Y$. This concept ensures that $X$ is mapped into $Y$ in a way that $X$'s structure—such as its continuity and open/closed set properties—is maintained exactly within the host structure $Y$</p>
</details>


<details>
  <summary><b>Machine learning</b></summary>
  <p>An embedding is a transformation $f: X \rightarrow \mathbb{R}^n$ that maps entities from a high-dimensional or abstract space $X$ (e.g., words, images, or graph nodes) to vectors in a lower-dimensional, continuous vector space $\mathbb{R}^n$. This mapping aims to preserve relevant properties of the original entities, such as similarity or relational structure, thereby enabling more effective computational manipulation and analysis.</p>
</details>

Not all vectors are embeddings --- Entities similar in the real world must be close in the embedding space. Consider the multiple birth example from [*Machine Learning Design Patterns*](*https://www.oreilly.com/library/view/machine-learning-design/9781098115777/*): We can use 6 one-hot vectors to represent 6 labels for the number of babies in one birth. Singles are more similar to twins than they are to quintuplets; yet, the cosine similarity between the single vector (`[1, 0, 0, 0, 0, 0]`) and the twin vector (`[0, 0, 1, 0, 0, 0]`) is 0, the same as that between the single vector and the quintuplet vector (`[0, 0, 0, 0, 0, 1]`). Since these one-hot vectors do not preserve the similarity between entities, they are *not* embeddings. 

| Plurality       | One-hot encoding        | Learned encoding |
|-----------------|-------------------------|------------------|
| Single (1)      | [1,0,0,0,0,0]           | [0.4, 0.6]       |
| Multiple (2+)   | [0,1,0,0,0,0]           | [0.1, 0.5]       |
| Twins (2)       | [0,0,1,0,0,0]           | [-0.1, 0.3]      |
| Triplets (3)    | [0,0,0,1,0,0]           | [-0.2, 0.5]      |
| Quadruplets (4) | [0,0,0,0,1,0]           | [-0.4, 0.3]      |
| Quintuplets (5) | [0,0,0,0,0,1]           | [-0.6, 0.5]      |


We can learn to map each label to a vector in a way that more similar labels are closer to one another, according to some distance function (e.g., Euclidean distance, Jaccard similarity, dot product, cosine similarity, and so on).

{{< figure src="https://www.dropbox.com/scl/fi/5xi8v3omgam3126dr0ahi/Screenshot-2024-04-21-at-2.58.25-PM.png?rlkey=zt24ehipliz2c1f2o8pol4zax&st=911v312w&raw=1" width="1000">}}

This type of learning is called ["metric learning"](https://paperswithcode.com/task/metric-learning#:~:text=The%20goal%20of%20Metric%20Learning,been%20developed%20for%20Metric%20Learning.). When using embeddings as features in a deep neural net, we can start with random initial values and update them via backpropagation just like we would any other parameters in the neural net. Alternatively, we can use [contrastive representation learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) to learn user and item embeddings, for example, and input learned embeddings into another neural net. The former ensures embeddings for optimal for the task at hand, whereas the latter greatly boosts training efficiency and is the choice of most companies.

Say we already have embeddings for entities we care about: Given a query point $q$, how do we find top-$k$ points $u \in \mathcal{X}$ most similar to it, to minimize a distance function $\delta$ calculated on entity embeddings? This is the top-$k$ retrieval problem: 

$$\mathop{\text{arg min}}\limits^{(k)}_{u \in \mathcal{X}} \delta(q, u).$$

The startup Pinecone is a lead provider of web-scale commercial top-$k$ retrieval solutions. In this blogpost, I review key ideas from the new monograph [*Foundations of Vector Retrieval (2024)*](https://arxiv.org/abs/2401.09350) by Sebastian Bruch, a Principal Scientist at Pinecone.


# Flavors of Vector Retrieval

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

# Approximate Retrieval

Regardless of the method, exhaustively comparing the query vector with every other vector to find top $k$ is computationally expensive. To avoid exhaustive search, many approximate retrieval algorithms have been proposed, trading accuracy for speed.

## Branch-and-Bound

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

## Clustering
Why not cluster vectors first, so that at retrieval time, we first find the cluster to which the query vector $q$ belongs, and then search top $k$ within that cluster?

- **Locality Sensitive Hashing (LSH)**: Hash each vector in $\mathbb{R}^d$ into a single bucket $h: \mathbb{R}^d \rightarrow \[b\]$ 👉 during retrieval, exhaustively search the bucket where $q$ is put into, assuming $q$ is hashed into the same bucket as its nearest neighbors

> WIP... Finish later...

# Vector Compression

# References

- Pinecone book
- Embed everything paper
- ML design pattern paper
