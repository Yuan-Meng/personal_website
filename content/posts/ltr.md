---
title: An Evolution of Learning to Rank
date: 2024-01-17
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

A LTR model learns how to sort documents by relevance for a query. During training, it sees many instances of how documents are ordered (by human label or user feedback) for each query; once trained, LTR can order new documents for new queries. To create ordering, LTR learns a function $f(q, D)$ that scores a list of documents $D$ given a query $q$. A single training instance consists of the following components:
- A query $q_i$ and a list of documents $D_i = \\{d_{i, 1}, d_{i, 2}, ..., d_{i, n}\\}$;
- The relevance score of each document given the query $Y = \\{y_{i, 1}, y_{i, 2}, ..., y_{i, n}\\}$;
- A feature vector for each query-document pair, $x_{i, j} = \phi(q_i, d_{i, j})$ 

Even if you don't work at a company with search engines, you can still play with publicly available [LTR datasets](https://paperswithcode.com/datasets?task=learning-to-rank#:~:text=Learning%20to%20Rank%20Challenge%20dataset,search%20engine%2C%20spanning%2029%2C921%20queries.&text=Publicly%20available%20dataset%20of%20naturally,purpose%20of%20automatic%20claim%20verification.). Below is the schema of the famous [The Yahoo! Learning to Rank Challenge](https://paperswithcode.com/dataset/learning-to-rank-challenge) dataset. More recently at NeurIPS 2022, Baidu Search released a [dataset](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=ALargeScaleSearchDatasetfor+UnbiasedLearningtoRank&btnG=#:~:text=Create%20alert-,%5BPDF%5D%20neurips.cc,-A%20large%20scale) with richer features and presentation information. Try them out!

```python3
FeaturesDict = {
    'query_id': [], # list of strings
    'doc_id': torch.Tensor().type(torch.int64),
    'float_features': torch.Tensor().type(torch.float64),
    'label': torch.Tensor().type(torch.float64),
}
```

> <span style="background-color: #00A598">**Work in progress... Now that I've written this, I have to finish...🤑**</span>

## What is the "Right" Order?

When training LTR, "out-of-order" predictions are penalized. "Out-of-orderness", or ranking errors, can be defined in 3 ways, leading to 3 types of scoring functions. 

### Pointwise
The pointwise approach doesn't have the concept of "out-of-orderness", but focuses on getting individual query-document relevance right. If you predict a document to be relevant to a query when it is not (or the reverse), then your ranker is wrong. 

Scored this way, LTR is equivalent to any regular regression model.

### Pairwise
If you predict that document $d_i$ is more relevant to a query than $d_j$ when it is the opposite, then your ranker is wrong.

### Listwise
If you predict the list of documents $D$ should be ordered in one way when in fact it should be ordered in another, then your ranker is wrong.

# Rank without Learning

Shocking as it may sound, ranking is not always done in the machine learning way. 

## TF-IDF
## BM25

# Learning to Rank
## Classic ML
## Deep Learning
## LLM as Re-Ranking Agent

# Learn More
## Papers
1. Li, H. (2011). [*A short introduction to learning to rank. IEICE TRANSACTIONS on Information and Systems, 94*](https://www.semanticscholar.org/paper/A-Short-Introduction-to-Learning-to-Rank-Li/d74a1419d75e8743eb7e3da2bb425340c7753342)(10), 1854-1862.

## Blogposts
2. [*Introduction to Learning to Rank*](https://everdark.github.io/k9/notebooks/ml/learning_to_rank/learning_to_rank.html) by Kyle Chung (2019)
2. [*Using LLMs for Search with Dense Retrieval and Reranking*](https://txt.cohere.com/using-llms-for-search/) by Cohere (2023)