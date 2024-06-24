---
title: "An Introduction to Embedding-Based Retrieval"
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
  <p>An embedding is a function $f: X \rightarrow Y$ between two topological spaces that is injective (one-to-one) and a homeomorphism onto its image, thus preserving the topological properties of $X$ within $Y$. This concept ensures that $X$ is mapped into $Y$ in a way that $X$'s structure — such as its continuity and open/closed set properties — is maintained exactly within the host structure $Y$</p>
</details>

<details>
  <summary><b>Machine learning</b></summary>
  <p>An embedding is a transformation $f: X \rightarrow \mathbb{R}^d$ that maps entities from a high-dimensional or abstract space $X$ (e.g., words, images, or graph nodes) to vectors in a lower-dimensional, continuous vector space $\mathbb{R}^d$. This mapping aims to preserve relevant properties of the original entities, such as similarity or relational structure, thereby enabling more effective computational manipulation and analysis.</p>
</details>

A vector $\mathbb{R}^d$ is an ordered list of numbers, which can represent almost everything: 

- A geographic location, described by `[latitude, longitude]`.
- A desk, characterized by `[height, area, color, other attributes]`.
- A photo, consisting of channel values for each pixel, `[[r, g, b], ...]`.

In traditional machine learning, each training example is described by a feature vector, usually consisted of hand-crafted features. For example, in spam classification, input text features might include the presence of a "$" 🤑 symbol in the email content, whether the subject line is in all CAPITAL LETTERS, and so on. 

All vectors are not embeddings. For vectors to be considered as embeddings, similar entities in the real world must also be close in the embedding space, according to some distance function (e.g., Euclidean distance, Jaccard similarity, dot product, cosine similarity, etc.) --- a property that regular vectors do not always satisfy. Consider the example from [*Machine Learning Design Patterns*](*https://www.oreilly.com/library/view/machine-learning-design/9781098115777/*): 6 one-hot vectors are used to represent the number of babies in one birth. While singles are more similar to twins than they are to quintuplets, the cosine similarity between the single vector (`[1, 0, 0, 0, 0, 0]`) and the twin vector (`[0, 1, 0, 0, 0, 0]`) is 0, the same as that between the single vector and the quintuplets vector (`[0, 0, 0, 0, 0, 1]`). After all, these one-hot vectors are orthogonal to one another. Since one-hot vectors do not capture similarities between categories, they are *not* embeddings.

We can use a lower-dimensional dense vector to represent each class label (column 3 in the table below), such that more similar labels are closer to one another. 

| Plurality       | One-hot encoding        | Learned encoding |
|-----------------|-------------------------|------------------|
| Single (1)      | [1,0,0,0,0,0]           | [0.4, 0.6]       |
| Multiple (2+)   | [0,1,0,0,0,0]           | [0.1, 0.5]       |
| Twins (2)       | [0,0,1,0,0,0]           | [-0.1, 0.3]      |
| Triplets (3)    | [0,0,0,1,0,0]           | [-0.2, 0.5]      |
| Quadruplets (4) | [0,0,0,0,1,0]           | [-0.4, 0.3]      |
| Quintuplets (5) | [0,0,0,0,0,1]           | [-0.6, 0.5]      |


{{< figure src="https://www.dropbox.com/scl/fi/5xi8v3omgam3126dr0ahi/Screenshot-2024-04-21-at-2.58.25-PM.png?rlkey=zt24ehipliz2c1f2o8pol4zax&st=911v312w&raw=1" width="1000">}}

The million-dollar question is, how do we learn the "proper" lower-dimensional representation of an entity in the embedding space? This is the exact type of problems that ["metric learning"](https://paperswithcode.com/task/metric-learning#:~:text=The%20goal%20of%20Metric%20Learning,been%20developed%20for%20Metric%20Learning.) tries to solve. Typically, we need to mine the raw training data (e.g., search or feed logs) for positive/negative pairs or triplets, initialize each entity's embedding with random values, and gradually pull similar entities (e.g., a user and a clicked item) closer and push dissimilar entities apart (e.g., a user and an unengaged item) in the embedding space using some contrastive objective (e.g., contrastive loss, triplet loss, Noise Contrastive Estimation, etc.). 

In the now-classic [paper](https://arxiv.org/abs/2006.11632), the Facebook Search team outlined the challenges of building a web-scale embedding-based retrieval system. These include defining positive/negative labels, balancing hard (e.g., impressed but unclicked search results) vs. easy (non-positive results sampled from the the mini-batch) negatives, and serving at scale. A particularly interesting finding is that training exclusively on hard negatives reduced recall by 55\% compared to training exclusively on in-batch negatives, yet adding a few hard negatives (e.g., two people on the search result page have the same name, but one is the searcher's social connection and one is not --- the latter is a hard negative) improved recall. It could be that easy negatives help the model capture textual similarities, while hard negatives force it to lean on contextual features (e.g., the searcher's location and social network). For a toy implementation of the embedding model architecture, check out this [repo](https://github.com/liyinxiao/UnifiedEmbeddingModel/blob/main/main.py). For an in-depth review of metric learning theories, read Lilian Weng's wonderful [blog post](https://lilianweng.github.io/posts/2021-05-31-contrastive/). For more examples of industry applications, you can like Jaideep's [post](https://medium.com/better-ml/embedding-learning-for-retrieval-29af1c9a1e65) and a recent industry [paper](https://arxiv.org/pdf/2006.02282) in the e-commerce space. 

After learning embeddings, we can answer many key questions --- to name a few:

- **First-pass ranking**: For a  user, how do we sift through a vast inventory of products/movies/posts/people/etc. to find items they may show interests in?
- **Passage retrieval/semantic search**: Given a natural language question, how do we retrieve passages that may contain the answer?

All this boils down to the **top-$k$ retrieval problem**: Given a query point $q$, how do we find top-$k$ document points $u \in \mathcal{X}$ that are most similar to it, so that we can minimize a distance function $\delta$ calculated on entity embeddings? 

$$\mathop{\text{arg min}}\limits^{(k)}_{u \in \mathcal{X}} \delta(q, u).$$

# Top-$k$ Retrieval Problem

The startup Pinecone is a leading provider of web-scale commercial top-$k$ retrieval solutions. In this blogpost, I review key ideas from the new monograph [*Foundations of Vector Retrieval (2024)*](https://arxiv.org/abs/2401.09350) by Sebastian Bruch, a Principal Scientist at Pinecone.

## Choices of Distance Functions

Finding top-$k$ points "closest" to the query point first requires a distance function. The figure below shows the 3 most popular choices (↓: minimize; ↑: maximize).

{{< figure src="https://www.dropbox.com/scl/fi/hx955sne496k99umhqi7f/Screenshot-2024-04-21-at-4.24.45-PM.png?rlkey=sxnuzw2ve6tye9qjf1g8jrt3v&st=0og5opxa&raw=1" width="1000" caption="Distance Functions for Top-$k$ Retrieval (Bruch, 2024, Chapter 1, p. 8)">}}


- **Euclidean distance** (↓): Straight line  from each point to the query point;
- **Cosine similarity** (↑): 1 - angular distance from each point to the query point;
- **Inner product** (↑): Imagine a hyperplane orthogonal to the query point passing through a document point — the shortest distance from this hyperplane to the query point is the inner product between the query-document pair.

<details>
  <summary><b>Proper vs. improper metrics</b></summary>
  <p>A proper metric 1) is non-negative, 2) symmetrical (i.e., $\delta(u, v) = \delta(u, v)$), and 3) satisfies the triangle inequality $\delta(u, v) \leq \delta(u, w) + \delta(w, v)$. Per these criteria, the inner product is not proper, because it is not non-negative and doesn't satisfy the triangle inequality, $\langle {u,v} \rangle \neq \langle {u,w} \rangle + \langle {w,v} \rangle$. In fact, we can't even guarantee that a vector maximizes the inner product with itself. That said, in a high enough dimension where data points $\mathcal{X}$ are i.i.d. in each dimension, we'd likely encounter "coincidences" with high confidence that $\langle {u,u} \rangle$ is greater than any $\langle {u,v} \rangle$ where $v \neq u$.</p>
</details>

The 3 distance functions lead to 3 common types of vector retrieval: 

- **$k$-Nearest Neighbor Search ($k$-NN)**: Minimizes Euclidean, $\mathop{\arg \min}\limits_{u \in \mathcal{X}}\limits^{(k)} \lVert q - u \rVert_2^2$;
- **$k$-Maximum Cosine Similarity Search ($k$-MCS)**: Minimizes angular distance, $\mathop{\arg \min}\limits_{u \in \mathcal{X}}\limits^{(k)} 1 - \frac{\langle {q,u} \rangle}{\lVert q \rVert_2 \lVert u \rVert_2}$, or maximizes cosine similarity $\mathop{\arg \max}\limits_{u \in \mathcal{X}}\limits^{(k)} \frac{\langle {q,u} \rangle}{\lVert u \rVert_2}$, given $\lVert q \rVert_2 = 1$;
- **$k$-Maximum Inner Product Search ($k$-MIPS)**: Maximizes inner product, $\mathop{\arg \max}\limits_{u \in \mathcal{X}}\limits^{(k)} \langle {q,u} \rangle$.

The 3 distance functions are related to one another. This is plain to see between $k$-MCS and $k$-MIPS: The former is a normalized version of the latter, where the inner product is divide by the $L_2$ norm of $u$. As for $k$-NN, we can expand the Euclidean distance into $\mathop{\arg \min}\limits_{u \in \mathcal{X}}\limits^{(k)} \lVert q \rVert_2^2 - 2\langle {q,u} \rangle + \lVert u \rVert_2^2$, which can be rewritten as $\mathop{\arg \min}\limits_{u' \in \mathcal{X'}}\limits^{(k)} \langle {q',u'} \rangle$ by discarding the constant term $\lVert q \rVert_2^2$ and concatenating vectors $q \in \mathbb{R}^d$ and $u \in \mathbb{R}^d$ each with a 1-dimensional vector $\[-1/2\]$ into $q' \in \mathbb{R}^{d + 1}$ and $u' \in \mathbb{R}^{d + 1}$, respectively.

When to use which? As with all ML problems, it depends on your data and use cases: 

| Distance Metric    | Common In                        | Advantage                                                 | Usage                                                   |
|--------------------|----------------------------------|-----------------------------------------------------------|---------------------------------------------------------|
| Euclidean Distance | Spatial databases, clustering    | Measures absolute differences; intuitive in low-dimensional spaces | Best when scale and actual size differences are crucial |
| Cosine Similarity  | Text retrieval, document similarity | Focuses on direction rather than magnitude; effective in high dimensions | Ideal for normalized data where orientation matters    |
| Inner Product      | Neural networks, collaborative filtering | Direct measure of alignment; computationally efficient with matrix operations | Useful when projection similarity is more relevant than geometric closeness |

## Approximate Retrieval Algorithms

Regardless of the distance function, when the embedding dimension $d$ is high and the documents are vast, it's inefficient to compute $\delta(q, v)$ for every query-document pair and return top $k$ documents in ascending order of distance. Efficient search calls for approximate top-$k$ retrieval algorithms that trade some accuracy for speed. 

The idea behind approximate top-$k$ retrieval is that we accept a vector $u$ as a valid solution so long as its distance to the query point $q$ is at most $(1 + \epsilon)$ times the distance to the $k$-th optimal vector (**Caveat**: Every vector may quality as an $\epsilon$-approximate nearest neighbor if embedding dimension $d$ is high and data are *i.i.d.* in every dimension, noted by [Beyer et al., 1999](https://minds.wisconsin.edu/bitstream/handle/1793/60174/TR1377.pdf?sequence=1&ref=https://githubhelp.com)). Recall at $k$ is often used to measure the effectiveness of approximate retrieval algorithms, which ideally maximize the overlap between the exact top-$k$ set $\mathcal{S}$ and the approximate top-$k$ set $\mathcal{\tilde{S}}$, $|\mathcal{S} \cap \mathcal{\tilde{S}}| / k$.  

In this section, we review some common algorithms for approximate top-$k$ retrieval. 

### Branch-and-Bound Algorithms

Branch-and-bound is one of the earliest algorithms for top-$k$ vector retrieval. It proceeds in two phases: 1) Recursively **partitioning** the vector space $\mathcal{X}$ into smaller regions, marking region boundaries, and storing them in a binary search tree (BST), and 2) only **searching** regions that could contain vectors in the top $k$ solution set. 

{{< figure src="https://www.dropbox.com/scl/fi/fxy7jr9oo6mac3j8ki0is/Screenshot-2024-04-21-at-5.56.57-PM.png?rlkey=ch8lg4imvf6xwrzecnwd662v5&st=wmmto60f&raw=1" width="400" caption="The Branch-and-Bound Algorithms (Bruch, 2024, Chapter 4, p. 32)">}}


- **Partitioning**: The original vector space is partitioned into a balanced binary search tree (BST), where each internal node has a decision boundary
    - Partition the vector space into regions $\mathcal{R}_l$ and $\mathcal{R}_r$; the boundary is $h$
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

Different instantiations of branch-and-bound algorithms differ in how they split a collection or conduct certification. In general, brand-and-bound algorithms work poorly on high-dimensional data as the number of leaves that may be visited during certification grows exponentially with the embedding dimension $d$. Modern approximate nearest neighbor retrieval services rarely rely on branch-and-bound.

### Locality Sensitive Hashing (LSH)

Locality Sensitive Hashing (LSH) reduces the nearest neighbor search space by hashing each vector into a single bucket, $h: \mathbb{R}^d \rightarrow \[b\]$, and searching exhaustively within the bucket. The choice of the hash function $h$ is critical because this algorithm only works if $\epsilon$-approximate $k$ nearest neighbors are hashed into the same bucket. 

To reduce the reliance on one hash function, we can independently apply $L$ hash functions, each from a family of hash functions $h \in \mathcal{H}$, to map vectors into buckets (see this Pinecone [blog post](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/) for more details on hash functions). If a query is hashed into multiple buckets, then we search all these buckets to find nearest neighbors --- we sacrifice some efficiency in hopes to get more accurate results. 

{{< figure src="https://www.dropbox.com/scl/fi/ibe1tr26h3k3pszf0lf17/Screenshot-2024-06-22-at-11.36.15-PM.png?rlkey=niw2bpqcztfc6xbevjmei8jni&st=klehneyl&raw=1" width="600" caption="Locality Sensitive Hashing (LSH) Algorithm (Bruch, 2024, Chapter 5, p. 58)">}}


### Graph Algorithms (e.g., HNSW)

Graph algorithms perform random walks from one vector to another via connected edges $(u, v) \in \mathcal{E}$, hopefully getting closer to the optimal solution with every hop. 

The graph $G(\mathcal{V}, \mathcal{E})$ is constructed during pre-processing of the vector collections ---
- **Nodes $\mathcal{V}$**: Each vector $u \in \mathcal{X}$ is a node in the graph $G$ --- i.e., $|\mathcal{V}| = |\mathcal{X}|$
- **Edges $\mathcal{E}$**: Simply connecting every node by an edge results in high space + time complexity --- how we can construct a sparse graph that solves the top-$k$ vector retrieval problem is an active research topic.

Whatever graph we decide to construct, it needs to support "best-first search", a greedy algorithm for finding top-$k$ nearest neighbors:

- **Entry**: To begin, enter the graph from an arbitrary node $u$;
- **Distance comparison**: Compare the distance from the node to query $q$ with the distance from each of the node's neighbors $N(u)$ to $q$;
  - **Terminate**: If no $N(u)$ is closer to $q$, then $u$ is a top-$k$ nearest neighbor;
  - **Hop**: If a $N(u)$ is closer to $q$ than $u$, then hop to the closest neighbor; 
- **Iteration**: Repeat until the terminal condition is met.

{{< figure src="https://www.dropbox.com/scl/fi/s5j2ithac4ukhiys5hcj7/Screenshot-2024-06-23-at-9.31.14-AM.png?rlkey=vn5kkouxd4w9ilqvzrpgi8u3x&st=e71z747p&raw=1" width="600" caption="The Greedy Best-First-Search Algorithm (Bruch, 2024, Chapter 6, p. 74)">}}

Below is a toy implementation to sketch out the algorithm we just described:

```python
import heapq
import math

def euclidean_distance(point1, point2):
    # calculate Euclidean distance between two points in space
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def best_first_search(graph, coords, start_node, query_point, k):
    # priority queue: [(negative distance to query point, current node)]
    priority_queue = [(euclidean_distance(query_point, coords[start_node]), start_node)]

    # track visited nodes to avoid revisiting
    visited = set()
    visited.add(start_node)

    # collect top-k nodes without storing distances
    result = []

    while priority_queue and len(result) < k:
        # get current node to visit
        current_distance, current_node = heapq.heappop(priority_queue)

        # assume it's a top-k solution
        is_candidate = True
        # visit each of the node's neighbors
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                # compute distance to the query
                dist = euclidean_distance(query_point, coords[neighbor])
                heapq.heappush(priority_queue, (dist, neighbor))
                visited.add(neighbor)
                # current node is not a candidate if a neighbor is closer
                if dist < current_distance:
                    is_candidate = False

        # if node is closest to query, add to result list
        if is_candidate:
            result.append(current_node)

            # if we already have k results, we can stop
            if len(result) >= k:
                break

    return result

# example graph in the form of an adjacency list
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B"],
    "F": ["C"],
}
# coordinates for each node
coords = {"A": [0, 0], "B": [1, 1], "C": [2, 2], "D": [5, 5], "E": [3, 3], "F": [4, 4]}

top_k_nodes = best_first_search(graph, coords, "A", [1, 2], 3)
# output: ['B', 'C', 'E']
```

If a graph cannot get us spatially closer to the solution with each hop, then it doesn't support best-first search. A widely used graph that does support best-first search is the [Delaunay graph](https://en.wikipedia.org/wiki/Delaunay_triangulation), which can be created from the [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram). 

- **Voronoi diagram**: The space $\mathbb{R}^d$ is partitioned into unique regions $\mathcal{R} = \bigcup_{u \in \mathcal{X}} \mathcal{R}_u$, where each region $\mathcal{R}_u$ is owned by $u \in \mathcal{X}$ and consists of $u$'s nearest neighbors;
- **Delaunay graph**: An undirected graph that connects nodes $u$ and $v$ in the Voronoi diagram if their Voronoi regions have a non-empty intersection, $\mathcal{R}_u \cap \mathcal{R}_u \neq  \emptyset$.

{{< figure src="https://www.dropbox.com/scl/fi/xlvcadtk361kqxpyuxt0p/Screenshot-2024-06-23-at-9.29.58-AM.png?rlkey=w9hwdwwkdj6ot2g5dkhcpjmjj&st=jvgy6y7w&raw=1" width="600" caption="The Delaunay Graph and the Voronoi Diagram (Bruch, 2024, Chapter 6, p. 77)">}}

If we pick an entry node far from the answer, then we must traverse all Voronoi regions in between to get there. To speed up traversal, we can add long-range edges between non-Voronoi neighbors to skip over certain regions. The question is, which long-range edges should we add? In his seminal *Nature* paper, [Kleinberg (2000)](https://www.nature.com/articles/35022643) proposed a probabilistic approach based on the lattice network:

- **Lattice network**: Every node has a directed edge to every node on a $m \times m$ grid;
- **Node distance**: The distance between two nodes $u$ and $v$ are defined by their Manhattan distance, $\delta (u, v) = \lVert u - v \rVert_1$;
- **Edge probability**: Form a long-distance edge between $u$ and $v$ with probability proportional to $\delta (u, v)^{- \alpha}$, where $\alpha \geq$ is a hyperparameter that controls the bias to forming a long-range connection (higher $\alpha$ favors longer distances).

{{< figure src="https://www.dropbox.com/scl/fi/rovai3146q4nvklhda978/Screenshot-2024-06-23-at-11.30.52-AM.png?rlkey=9rubh46vnleftougo1ytos5sf&st=1d2jinnh&raw=1" width="600" caption="Forming Long-Distance Edges in the Lattice Network (Bruch, 2024, Chapter 6, p. 88)">}}

With long-distance edges, the average number of hops required to go from one node to another significantly drops --- an observation dubbed as the "small world phenomenon". The resulting Navigable Small World (NSW) graphs are the basic of the Hierarchical Navigable Small World (HNSW) algorithm that allows for remarkably fast nearest neighbor search. You can find more details on HNSW in this Pinecone [post](https://www.pinecone.io/learn/series/faiss/hnsw/).


### Clustering (e.g., FAISS)

The motivation behind clustering is similar to that behind hashing, but instead of using a hash function to map vectors into buckets, we can use a clustering function (e.g., KMeans) to map vectors into clusters, $\xi : \mathbb{R}^d \to [C]$. At retrieval time, we apply a routing function, $\tau : \mathbb{R}^d \to [C]^{l}$, to return top-$l$ clusters whose centroids are the closest to the query vector $q$, and then search for top-$k$ neighbors over the union of top-$l$ clusters. This is the main idea behind [Facebook AI Similarity Search (FAISS)](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/), perhaps the most popular approximate retrieval algorithm today.

{{< figure src="https://www.dropbox.com/scl/fi/ka38302lxoo46pnd2xswg/Screenshot-2024-06-23-at-1.36.11-PM.png?rlkey=s1tvwknbb82a2ktzl3ttlvb51&st=s5cqpozw&raw=1" width="600" caption="Clustering Algorithms for Top-$l$ and Top-$k$ Retrieval (Bruch, 2024, Chapter 7, p. 106)">}}

For clustering algorithms to work, the data we search over must follow a multi-modal distribution --- which is fortunately usually the case with real-world data.

<!-- ### Sampling Algorithms -->

# Embedding Storage Optimization

The search algorithms above aim to reduce the search space with some optimality guarantee, whereas the optimization tricks below aim to save the embedding storage.

## Quantization

When using clustering algorithms such as FAISS, we can think of each of the $C$ centroids as a "codeword" and the $2^C$ combinations they form as the "codebook". Each vector can be encoded using $\log_2 C$ bits --- this is called **Vector Quantization**. Before quantization, $O(md)$ space is required to store the embeddings ($m$: number of embeddings; $d$: embedding dimension), but only $O(Cd + m\log_2 C)$ space is needed afterward ($O(Cd)$: stores original centroids). A larger $C$ reduces the approximation error but requires more space; conversely, a smaller $C$ saves space but increases the error. 

Today, a more popular quantization method is **Product Quantization**, which divides a high-dimensional vector (e.g., 128) into $L$ orthogonal subspaces (e.g., 8 subspaces, each of dimension 16), performs Vector Quantization on each subspace, and concatenates the quantized subspaces. This approach is particularly beneficial when the embedding dimension $d$ is so high that many centroids are needed to cover the space $\mathbb{R}^d$. In contrast, we may only need a small number of centroids to cover each subspace, so that even with $L$ subspaces, the total number of centroids still remains fewer than what we would need if we were to quantize the entire vector directly.

## Sketching 

Sketching is a type of algorithms that map a higher-dimensional vector to a lower-dimensional vector, $\phi : \mathbb{R}^d \to \mathbb{R}^{d_\circ}$ ($d_\circ < d$), after which certain properties of interest (e.g., the Euclidean distance or the inner product between any pair of points) are preserved with high probability. Then, instead of searching over original vectors, we search over their sketches $\phi(u)$ for $u \in \mathcal{X}$ to solve top-$k$ retrieval problems.

That is the theory, at least. For certain type of problems, sketching can result in unacceptable errors. I recommend that you read Ethan N. Epperly's [blog post](https://huggingface.co/blog/ethanepperly/does-sketching-work) and [paper](https://arxiv.org/abs/2311.04362) for a detailed analysis of common sketching algorithms and sketching errors.  

## Feature Multiplexing

Embeddings are typically stored in a $N \times d$ lookup table, where $N$ is the size of the "vocabulary" and $d$ the embedding dimension. In NLP, $N$ is in the order of tens of thousands. In search/ads/recommendations, $N$ can be in the order of tens of billions (e.g., users, items), which can easily blow up model parameters and storage. 

Like how we can decompose words into subword tokens, we can decompose embeddings into subspaces to reduce the vocabulary size --- each embedding is stored in multiple rows, each row representing a subspace, and the original embedding can be recovered from a weighted sum of rows. This is the "hashing trick" ([Weinberg et al., 2009](https://arxiv.org/pdf/0902.2206.pdf)). 

Researchers at DeepMind ([Coleman et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/afcac2e300bc243d15c25cd4f4040f0d-Paper-Conference.pdf)) proposed a new learning framework called *Feature Multiplexing*, where all embedding features share a single embedding table and multiple features may share one representation space (e.g., semantically-similar features on the query and the document sides). Models trained with Feature Multiplexing achieved or beat SOTA performance on both open-source and Google data.

# References

## Books/Papers
1. Bruch, S. (2024). Foundations of Vector Retrieval. [arXiv:2401.09350](https://arxiv.org/pdf/2401.09350).
2. Huang, J. T., Sharma, A., Sun, S., Xia, L., Zhang, D., Pronin, P., ... & Yang, L. (2020). Embedding-based retrieval in Facebook search. KDD ([paper](https://arxiv.org/abs/2006.11632) + [code](https://github.com/liyinxiao/UnifiedEmbeddingModel)).
3. Coleman, B., Kang, W. C., Fahrbach, M., Wang, R., Hong, L., Chi, E., & Cheng, D. (2023). Unified Embedding: Battle-tested feature representations for web-scale ML systems. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/afcac2e300bc243d15c25cd4f4040f0d-Paper-Conference.pdf).
4. Zhang, H., Wang, S., Zhang, K., Tang, Z., Jiang, Y., Xiao, Y., ... & Yang, W. Y. (2020, July). Towards personalized and semantic retrieval: An end-to-end solution for e-commerce search via embedding learning. [SIGIR](https://arxiv.org/pdf/2006.02282).

## Blog Posts
5. [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) by Lilian Weng (2021)
6. [Embedding-Based Retrieval for Search & Recommendation](https://medium.com/better-ml/embedding-learning-for-retrieval-29af1c9a1e65) by Jaideep Ray (2021)
7. [Locality Sensitive Hashing (LSH): The Illustrated Guide](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/) by Pinecone
8. [Hierarchical Navigable Small Worlds (HNSW)](https://www.pinecone.io/learn/series/faiss/hnsw/) by Pinecone
9. [FAISS: The Missing Manual](https://www.pinecone.io/learn/series/faiss/) by Pinecone
10. [Does Sketching Work?](https://huggingface.co/blog/ethanepperly/does-sketching-work) by Ethan N. Epperly (2023)