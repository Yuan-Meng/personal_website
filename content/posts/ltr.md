---
title: An Evolution of Learning to Rank
date: 2024-02-17
math: true
tags:
    - search
    - information retrieval
    - learning to rank
categories:
- papers
keywords:
    - search, information retrieval, learning to rank
include_toc: true
---


# First Thing First

> Enigmas of the universe <br/> Cannot be known without a search <br/> --- Epica, [*Omega*](https://open.spotify.com/track/34Oz0bzAq7E1aUnKksPfJJ?si=9eabd1446a6a4ccc) (2021)


In *[The Rainmaker (1997)](https://en.wikipedia.org/wiki/The_Rainmaker_(1997_film))*,  the freshly graduated lawyer Rudy Baylor faced off against a giant insurance firm in his debut case, almost getting buried by mountains of case files that the corporate lawyers never expected him to sift through. If only Rudy had a search engine that *retrieves* all files mentioning suspicious denials and *ranks* them from most to least relevant, the case prep would've been a breeze.

{{< figure src="https://www.dropbox.com/scl/fi/fto9nalobh5ku9nb2yh19/Screenshot-2024-01-18-at-12.32.46-AM.png?rlkey=l7z94c0rfv940tv2xp8u2wdym&raw=1" width="600" caption="Search process overview: Indexing, matching, ranking ([*Relevant Search*](https://www.manning.com/books/relevant-search))" >}}


For a holistic view of search applications, I highly recommend the timeless *Relevant Search* ([repo](https://github.com/o19s/relevant-search-book)). Raw documents go through *analysis* to get indexed into searchable fields (e.g., case name, case year, location, etc.). At search time, a user-issued query (e.g., "leukemia") undergoes the same analysis to *retrieve* matching documents. Top documents are *ranked* in descending order of relevance before being returned. This post focuses on ranking, especially *learning to rank* (LTR).

## Problem Formulation of LTR

> "<span style="background-color: #FDB515">**Ranking is nothing but to select a permutation**</span> $\pi_i \in \Pi_i$ for the given query $q_i$ and the associated documents $D_i$ using the scores given by the ranking model $f(q_i, D_i)$." --- [*A Short Introduction to LTR*](https://www.jstage.jst.go.jp/article/transinf/E94.D/10/E94.D_10_1854/_article), Li (2011)

LTR learns how to sort a list of documents retrieved for a query in descending order of relevance. During training, it sees many instances of how documents are ordered for different queries; once trained, LTR can order new documents for new queries, using a learned function $f(q, D)$ that scores documents $D$ given a query $q$. 

{{< figure src="https://www.dropbox.com/scl/fi/dzu1yhzmvfzdfr5itaqnp/Screenshot-2024-02-17-at-2.28.19-PM.png?rlkey=o9mhs0lxni9azzxf5soxkjq6o&raw=1" width="500" >}}

A single training instance consists of the following components:
- A query $q_i$ and a list of documents $D_i = \\{d_{i, 1}, d_{i, 2}, ..., d_{i, n}\\}$;
- The relevance score of each document given the query $Y = \\{y_{i, 1}, y_{i, 2}, ..., y_{i, n}\\}$;
- A feature vector for each query-document pair, $x_{i, j} = \phi(q_i, d_{i, j})$ 

Even if you don't work at a company with search engines, you can still play with publicly available [LTR datasets](https://paperswithcode.com/datasets?task=learning-to-rank#:~:text=Learning%20to%20Rank%20Challenge%20dataset,search%20engine%2C%20spanning%2029%2C921%20queries.&text=Publicly%20available%20dataset%20of%20naturally,purpose%20of%20automatic%20claim%20verification.), such as the famous [The Yahoo! Learning to Rank Challenge](https://paperswithcode.com/dataset/learning-to-rank-challenge) dataset. More recently at NeurIPS 2022, Baidu Search released a [dataset](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=ALargeScaleSearchDatasetfor+UnbiasedLearningtoRank&btnG=#:~:text=Create%20alert-,%5BPDF%5D%20neurips.cc,-A%20large%20scale) with richer features and presentation information. Try them out!

## What is the "Right" Order?

### Evaluation Metrics

What makes LTR special is that it penalizes predictions that lead to "out-of-order" document ranking compared to the "ideal ranking", rather than getting each individual document's score right. Ideal ranking is usually determined by human judgments (classic 5-point scale: "perfect", "excellent", "good", "fair", "bad") or user engagement (e.g., conversion: 3, click: 2, impression: 1).

{{< figure src="https://www.dropbox.com/scl/fi/tk7vwv9u1a7qah9vrcbyh/Screenshot-2024-02-17-at-3.08.24-PM.png?rlkey=3uxj0u94i5zpf3i75524lcwg4&raw=1" width="600" >}}

In the example above, the search engine ranks a less relevant document above a more relevant one, so it's not perfect. A common ranking metric that takes all relevant documents and their ordering into account is Normalized Discounted Cumulative Gain at top k results (nDCG@k), which can be computed as the following:

- **DCG**: Given a ranking for query $q_i$, we can compute a Discounted Cumulative Gain for top k documents (DCG@k), $DCG_k = \sum_{j}^{k} \frac{\mathrm{rel_{i,j}}}{\log_2 (j + 1)}$, where the relevance label of the document at position $j$ is its "gain", and $\log_2 (j + 1)$ is a position discount function. In this case, $DCG_3 = \frac{5}{\log_2 (1 + 1)} + \frac{3}{\log_2 (2 + 1)} + \frac{4}{\log_2 (2 + 1)} \approx 9.02$.
- **IDCG**: We can compute DCG@k for the ideal order, or "Ideal Discounted Cumulative Gain at k (IDCG@k)". Here, $IDCG_3 = \frac{5}{\log_2 (1 + 1)} + \frac{4}{\log_2 (2 + 1)} + \frac{3}{\log_2 (2 + 1)} \approx 8.89$. 
- **nDCG**: The ratio of DCG@k and IDCG@k is nDCG@k: $nDCG_k = \frac{DCG_k}{IDCG_k}$. 

In this example, $nDCG_3 = 8.89 / 9.02 \approx 0.98$, which is just short of perfect (1). If we completely reverse the document order from the ideal order, the result is still not bad: $DCG_3 = \frac{3}{\log_2 (1 + 1)} + \frac{4}{\log_2 (2 + 1)} + \frac{5}{\log_2 (2 + 1)} \approx 8.02$ and $nDCG_3 = 8.02 / 9.02 \approx 0.89$. This is because irrelevant results (1 or 2 ratings) didn't even make it to top k. However, what if top k results are all bad? nDCG@k would still be 1 as long as the model ranks documents in the same order as relevance labels would. nDCG@k measures the goodness of ranking, but even perfect ranking cannot save search relevance if bad documents dominate top k results due to flawed retrieval/first-pass ranking.

When we don't have fine-grained relevance ratings but only 0 ("Not Relevant") / 1 ("Relevant") labels, we can use other ranking metrics that fall under 2 categories:

- **Order-unaware metrics**: Easy to compute but document order is ignored
  - **R@k** (recall @k): # of relevant docs in top k / # of relevant docs
  - **P@K** (precision @K): # of relevant docs in top k / k
- **Order-aware metrics**: What is search, if not to *order* results?
  - **MRR** (mean reciprocal rank): On average, how early the 1st relevant doc shows up 👉 $\frac{1}{n}\sum_{1}^{n}\frac{1}{\mathrm{rank}_i}$, where $q_i$ is a query in an eval query set of size $n$, and $\mathrm{rank}_i$ is the rank of the 1st relevant doc for $q_i$
  - **MAP@k** (mean average precision @k): For each query, we can compute its average precision across top k (AP@k), weighted by binary relevance @k 👉 $AP_k = \frac{\sum_{1}^k(P@k \cdot \mathrm{rel}_k)}{\sum_1^k \mathrm{rel}_k}$. AP@k across the eval set is then the MAP@k.
  {{< figure src="https://cdn.sanity.io/images/vr8gru94/production/c009584211e87b2bad916931b348085d13f34bc5-1206x990.png" width="600" caption="MAP@k, Evaluation Measures in Information Retrieval by [Pinecone](https://www.pinecone.io/learn/offline-evaluation/)">}}


### Ranking Loss

Using nDCG to measure LTR performance, the true loss function is $L(F(\mathbf{x}), \mathbf{y}) = 1.0 - nDCG$, where $F(\cdot)$ is a function mapping from a list of feature vectors $\mathbf{x}$ to a list of scores, and $\mathbf{y}$ is a list of true scores. The sorting operation in nDCG makes the true loss non-differentiable at times. Below are 3 types of surrogate losses.


1. **Pointwise**: Predict how relevant each doc is for a given query
    - **General form**: $L'(F(\mathbf{x}), \mathbf{y}) = \sum_{i=1}^n (f(x_i) - y_i)^2$; same as a regular regression model (or binary classification, or ordinal regression models)
    - **Motivation**: If we predict each document's relevance w.r.t. the query well, hopefully we can get an ideal order for free
    - **Drawbacks**: Doesn't optimize for ranking at all --- small predictions errors on individual docs may result in a completely out-of-order list; doesn't leverage relationships between docs retrieved for the same query

2. **Pairwise**: Predict which among a pair of docs is more relevant for query
    - **General form**: $L'(F(\mathbf{x}), \mathbf{y}) = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \phi (\mathrm{sign}(y_i - y_j), f(x_i) - f(x_j))$, which makes the score of a more relevant doc higher, and the score of a less relevant doc lower for each given query
    - **Motivation**: If we can predict the better doc among each pair (e.g., A > B, B > C, A > C), we can recover the best global order (A > B > C)
    - **Drawbacks**: Doesn't optimize for global ranking --- if we swap a pair (e.g., **A < B**, B > C, A > C), then the global ranking is wrong (B > A > C)
    - **History**: [Chris Burges](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) from Microsoft proposed 3 methods for pairwise LTR: RankNet, LambdaNet, and LambdaMART (used by LightGBM)

3. **Listwise**: Predict the best permutation of docs for a query ---  like the true loss
   - **General form**: $L'(F(\mathbf{x}), \mathbf{y}) = \exp(-nDCG) = \sum_{i=1}^m (E(\pi^\*_i, \mathbf{y_i}) - E(\pi_i, \mathbf{y_i}))$, where $\pi_i$ is the permutation generated by labels and $\pi^\*_i$ by model scores
   - **Motivation**: Get the global ranking right in a straightforward way
   - **Drawbacks**: Computationally expensive to enumerate all doc permutations and far more complicated to implement than pointwise or pairwise approaches
   - **History**: Later, Google researchers proposed a generalized framework [LambdaLoss](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/), which encompasses pointwise, pairwise, and listwise approaches

# Rank without Learning

How do we learn that $f(q_i, D_i)$ function? This is a loaded question because this function is not always learned. LTR was only popularized in the last 15 years or so, whereas traditional search engines rely on the textual match between a query and documents to determine document relevance. Due to their simplicity, these methods are still used by modern search engines to retrieve top N documents to re-rank.

## TF-IDF

When searching on Netflix, for example, the query "rebecca" could match a movie's title (e.g., Hitchcock's *Rebecca*), cast (e.g., actress Rebecca Ferguson), or description (e.g., protagonist named Rebecca appeared in the summary). Queries and document fields can all be represented by [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vectors --- each element in the vector represents a token in the "vocabulary" and the value is 0 if the given token is absent or the product of "TF" (term frequency: how many times the token appears in the corpus) and "IDF" (inverse document frequency: 1 / in how many documents does the token appear). The dot product between a query's and a field's TF-IDF vectors measures their closeness (higher is better), which is used to rank documents. 

Fields can have different weights --- perhaps, a title match is 3 times more important than a cast match, and 10 times more important than a description match. These weights usually come from heuristics and are tuned by observing search defects. 

## BM25

Whether "Rebecca" appears once in a document field or 10 times, the dot product between the two would be the same. A family of ranking functions, BM25, scores each document $D$ by counting how many times terms $q_1, \ldots, q_n$ in the query $Q$ appear in it (notations differ from previous sections: $q_i$ is a query term, not the whole query).


"25" denotes the 25th version and even it takes many forms. A classic instantiation is $\mathrm{score}(D, Q) = \sum_{i=1}^{n}\mathrm{IDF}(q_i)\cdot \frac{f(q_i, D)\cdot (k1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\mathrm{avgdl}})}$, where $\mathrm{IDF}(q_i)$ is the inverse document frequency of the query term $q_i$, $f(q_i, D)$ is the frequency of $q_i$ in $D$, $|D|$ is the token-level document length, and $\mathrm{avgdl}$ is the average document length across the corpus. $k_1$ and $b$ are free parameters with typical values $k_1 \in [1.2, 2.0]$ and $b = 0.75$.


# Learning to Rank

TF-IDF and BM25 are driven by "magical numbers": How do we find the best document field weights? Where do free parameters $k_1$ and $b$ in BM25 come from? Moreover, the two or similar methods treat relevance as an inherent, static property between a query and a document. However, relevance also depends on *who* is searching in what *context* --- I'm a big Hitchcock fan so *Rebecca (1940)* is likely what I want whereas you want to watch another movie starring Rebecca Ferguson coming from *Dune* and *Mission Impossible*. Machine learning is a natural choice to leverage query, document, user, and context features to systemically optimize for search relevance. 

## Traditional ML

When using pointwise loss, ranking can be framed was a regression (predict relevance scores) or a classification (predict click probabilities) problem, for which any regression or classification models work. As discussed, this overly simplified approach doesn't optimize for document ordering, so we won't go into depth here. 

RankSVM is an early linear model that optimizes for pairwise loss. Today, pairwise LTR based on Gradient Boosted Decision Trees (GBDT) is by far the most popular --- [LambdaMART](https://ffineis.github.io/blog/2021/05/01/lambdarank-lightgbm.html) and its [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html) implementation are widely used in commercial search engines and, until recently ([Qin et al., 2021](https://openreview.net/pdf?id=Ut1vF_q_vC)), outperformed all neural rankers.  

LambdaMART uses the LambdaRank gradient for pairwise loss, $\log(1 + \exp(-\sigma(s_i - s_j)))|\Delta \mathrm{nDCG}|$, which, at its core, is a position-aware version of binary log loss.

- **Pairwise ranking as binary classification**: Pairwise loss converts document ranking into $k \choose 2$ binary classification problems, where $k$ is the number of top documents to re-rank for the given query. Say document $i$ is more relevant than $j$, how likely will the model rank $i$ above $j$? This probability $\mathrm{Pr}(i > j)$ is given by the logistic function, $\frac{1}{1 + \exp(-\sigma(s_i - s_j))}$ --- if $s_i = s_j$, the model has a fifty-fifty chance of guessing right; if the diff is in the right/wrong direction, then the model has a higher/lower probability of ranking the pair in the correct order.
    - **Note**: In LightGBM, the hyperparameter $\sigma$ defaults to 1
- **Injecting position information with nDCG**: Swapping the order of $i$ and $j$ gives us 2 ranked lists --- the nDCG difference between the 2 lists is given by $|\Delta \mathrm{nDCG}|$. This term captures our intuition that swapping positions 1 and 2 has more severe consequences than, say, swapping positions 9 and 10. 
- **LambdaRank gradient**: $\lambda_{ij}$ is simply the product of $\log \mathrm{Pr}(i > j)$ and $|\Delta \mathrm{nDCG}|$.

Like other GBDT models, LambdaMART assigns higher weights to document pairs with higher loss, so that the model will focus on such pairs in later iterations. While deep learning has become the gold standard in fields such as natural language processing (NLP) and computer vision (CV), GBDT still dominates search ranking, especially on tabular data. One reason is that GBDT is robust to feature scales --- the absolute values don't matter, so long as the relative ordering is correct --- whereas neural nets require careful feature normalization and scaling to learn. Another reason is that in academia, LTR benchmarks are not big enough for DL to shine, yet in industry, search is a surface with strict latency requirements and DL models can have much higher latency than famously fast GBDT models such as LightGBM. 

## Deep Learning

It's interesting that in CV, DL replaced non-DL methods (e.g., [hierarchical Bayes](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=d44cbf0e2997f63e45b7d409a3c25918562480ed)) as the mainstream after the [ImageNet breakthrough](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) in 2012, whereas LTR saw a transition from DL ([RankNet (2005)](https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/), [LambdaRank (2007)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf)) to non-DL ([LambdaMART (2010)](https://scholar.google.com/scholar?cites=12943173042546364194&as_sdt=2005&sciodt=0,5&hl=en)). Early neural LTR typically uses a feed-forward network with fully connected layers, which doesn't play to DL's strength: Learning high-order feature interactions. Moreover, early neural LTR didn't put much focus on feature transformation, which profoundly impacts neural nets. [Google researchers (2021)](https://openreview.net/pdf?id=Ut1vF_q_vC) demonstrated that using newer architectures on transformed data, neural LTR was able to beat LightGBM.

- **Normalization**: The feature vector $\mathbf{x}$ goes through "log1p" transformation, $\mathbf{x} = \log_e(1 + |\mathbf{x}|) \odot \mathrm{sign}(\mathbf{x})$, where $\odot$ is element-wise multiplication.
- **Gaussian noise**: A random Gaussian noise is injected to every element of the feature vector $\mathbf{x}$, $\mathbf{x} = \mathbf{x} + \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$, to augment input data.
- **Multi-head self-attention**: Documents in the ranked list are encoded with multi-head self-attention to generate a contextual embedding for each doc.

On Web30K and Istella benchmarks, the neural LTR outperformed LightGBM (still lost in the Yahoo! benchmark), showing DL is not inherently inferior for ranking.

{{< figure src="https://www.dropbox.com/scl/fi/qlv1h1710sg2gjek4lnyd/Screenshot-2024-02-17-at-11.29.21-PM.png?rlkey=z1r26whv0r9qdeprhi3a08vy8&raw=1" width="650" >}}

Smashing benchmarks may not be so exciting if that's all LTR can do --- the Airbnb trilogy ([2019](https://arxiv.org/abs/1810.09591), [2020](https://arxiv.org/abs/2002.05515), [2023](https://arxiv.org/pdf/2305.18431)) demonstrated the power of DL in commercial search engines. Not only did offline nDCG and online conversions increase, but DL allows ML engineers to shift focus from Kaggle-style feature engineering to deeper questions about ranking: What's the right learning *objective* to optimize for (multi-task, multi-objective learning)? How to better *represent* user/query/docs (two-tower)? 


<span style="background-color: #FDB515">**TODO: Read and summarize Airbnb + Google papers**</span>


### Fully-Connected DNNs

### Two-Tower Architecture

### Multi-Task Learning


## LLM Re-Rankers
<span style="background-color: #FDB515">**TODO: Read and summarize Google + Cohere papers**</span>

# Learn More
## Papers
1. Burges, CJC. (2010). [*From RankNet to LambdaRank to LambdaMART: An overview.*](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=0df9c70875783a73ce1e933079f328e8cf5e9ea2) *Microsoft Research Technical Report*.
2. Haldar, M., Abdool, M., Ramanathan, P., Xu, T., Yang, S., Duan, H., ... & Legrand, T. (2019). [*Applying deep learning to Airbnb search*](https://arxiv.org/pdf/1810.09591). *SIGKDD*.
3. Haldar, M., Ramanathan, P., Sax, T., Abdool, M., Zhang, L., Mansawala, A., ... & Liao, J. (2020). [*Improving deep learning for Airbnb search*](https://arxiv.org/pdf/2002.05515). *SIGKDD*.
4. Joachims, T. (2006). [*Training linear SVMs in linear time*](https://www.cse.iitb.ac.in/~soumen/readings/papers/Joachims2006linearTime.pdf). *SIGKDD*.
5. Li, H. (2011). [*A short introduction to learning to rank.*](https://www.semanticscholar.org/paper/A-Short-Introduction-to-Learning-to-Rank-Li/d74a1419d75e8743eb7e3da2bb425340c7753342) *IEICE TRANSACTIONS on Information and Systems, 94*(10).
6. Pradeep, R., Sharifymoghaddam, S., & Lin, J. (2023). [*RankZephyr: Effective and robust zero-shot listwise reranking is a breeze!*](https://arxiv.org/pdf/2312.02724?trk=public_post_comment-text) *arXiv*.
7. Qin, Z., Yan, L., Zhuang, H., Tay, Y., Pasumarthi, R. K., Wang, X., and Najork, M. (2021). [*Are neural rankers still outperformed by gradient boosted decision trees?*](https://research.google/pubs/are-neural-rankers-still-outperformed-by-gradient-boosted-decision-trees/), *ICLR*.
8. Qin, Z., Jagerman, R., Hui, K., Zhuang, H., Wu, J., Shen, J., ... & Bendersky, M. (2023). [*Large language models are effective text rankers with pairwise ranking prompting.*](https://arxiv.org/pdf/2306.17563) *arXiv*.
9. Tan, Chun How, Austin Chan, Malay Haldar, Jie Tang, Xin Liu, Mustafa Abdool, Huiji Gao, Liwei He, and Sanjeev Katariya. [*Optimizing Airbnb search journey with multi-task learning.*](https://arxiv.org/pdf/2305.18431) *arXiv*.
10. Wang X., Li C., Golbandi, N., Bendersky, M., and Najork, M. (2018). [*The LambdaLoss framework for ranking metric optimization.*](https://dl.acm.org/doi/pdf/10.1145/3269206.3271784) *CIKM*.
11. Yan, L., Qin, Z., Zhuang, H., Wang, X., Bendersky, M., & Najork, M. (2022). [*Revisiting two-tower models for unbiased learning to rank*.](https://dl.acm.org/doi/pdf/10.1145/3477495.3531837) *SIGIR*.

## Blogposts
1. [*Introduction to Learning to Rank*](https://everdark.github.io/k9/notebooks/ml/learning_to_rank/learning_to_rank.html) by Kyle Chung (2019)
2. [*The inner workings of the lambdarank objective in LightGBM*](https://ffineis.github.io/blog/2021/05/01/lambdarank-lightgbm.html) by Frank Fineis (2021)
2. [*Deep Multi-task Learning and Real-time Personalization for Closeup Recommendations*](https://medium.com/pinterest-engineering/deep-multi-task-learning-and-real-time-personalization-for-closeup-recommendations-1030edfe445f) by Pinterest (2023)
3. [*Using LLMs for Search with Dense Retrieval and Reranking*](https://txt.cohere.com/using-llms-for-search/) by Cohere (2023)