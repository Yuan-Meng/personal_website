---
title: "Negative Sampling for Learning Two-Tower Networks"
date: 2024-08-31
math: true
tags:
    - negative sampling
    - information retrieval
categories:
- papers
keywords:
    - negative sampling, information retrieval
include_toc: true
---

# Web-Scale Recommender Systems

## Two-Stage Architecture

The iconic YouTube paper ([Covington et al., 2016](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)) introduced a two-stage architecture that since became the industry standard for large-scale recommender systems: 

- **Retrieval** (a.k.a. "candidate generation"): Quickly select top k (in the hundreds or thousands) loosely relevant items from a large corpus of billions
- **Ranking** (a.k.a. "reranking"): Order final candidates (dozens) by predicted reward probability (e.g., an ads click, a listing booking, a video watch, *etc.*)

{{< figure src="https://www.dropbox.com/scl/fi/xfzcn69xt5umpiqgiblvj/Screenshot-2024-09-01-at-3.27.32-PM.png?rlkey=co9ingrwzjdawfkj4nfg7j4eg&st=ym59f9sr&raw=1" width="500">}}

## Two-Tower Network for Retrieval

Reranking models are highly customized for each product at each company, tailored to bespoke ranking objectives for the business. For retrieval, the two-tower network is the go-to choice at many companies to efficiently maximize Recall@k. 

{{< figure src="https://www.dropbox.com/scl/fi/74ymkpk7f6140fa2s5qrh/Screenshot-2024-09-01-at-11.48.42-PM.png?rlkey=wwbu4q3e5do8zj0fsqac61no7&st=m1mixkkp&raw=1" width="500">}}

As the name suggests, the two-tower network consists of two deep neural networks, each shaped like a "tower" with every successive hidden layer halving in the number of hidden units. The left tower encodes a query $x \in \mathcal{X}$ (e.g., usually includes a user in some context, and sometimes a search query or a seed item) whereas the right tower encodes an item $y \in \mathcal{Y}$. The query and the item towers both take sparse IDs and dense features as inputs and learn a function mapping from the input feature to a dense embedding, $u : \mathcal{X} \times \mathbb{R}^d \rightarrow \mathbb{R}^k$ and $v : \mathcal{Y} \times \mathbb{R}^d \rightarrow \mathbb{R}^k$, respectively. The output of the network is the dot product of the query and the item embeddings, 

$$s(x, y) = \langle u(x, \theta), v(y, \theta) \rangle.$$

How do we know if $s(x_i, y_i)$ for the pair of query $x_i$ and item $y_i$ is good or not? 

Intuitively, if item $y_i$ is positive for query $x_i$, $s(x_i, y_i)$ should be greater than if $y_i$ is negative for $x_i$. After training, the model should learn to embed positive $(x_i, y_i)$ pairs closer in the embedding space and negative pairs further apart.

Formally, we can frame retrieval as extreme multi-class classification: Given the query $x_i$, predict the probability of selecting each item $y_j$ from $M$ items $\\{y_j\\}_{j=1}^M$ ---


$$p(y_i | x_i; \theta) = \frac{e^{s (x_i, y_i)}}{\sum_{j \in \[M\]} e^{s (x_i, y_j)}},$$


where $\theta$ is the model parameters. Each pair of query $x_i$ and item $y_i$ is associated with a reward $r_i \in \mathbb{R}$ denoting binary (1: positive; 0: negative) or varying degrees of user engagement. How "wrong" $p(y_i | x_i; \theta)$ is can be given by the softmax loss:

$$\mathcal{L_T(\theta)} = -\frac{1}{T}\sum_{i \in \[T\]}r_i \cdot \log(p(y_i | x_i; \theta)) = -\frac{1}{T}\sum_{i \in \[T\]}r_i \cdot \log(\frac{e^{s (x_i, y_i)}}{\sum_{j \in \[M\]} e^{s (x_i, y_j)}}),$$

where $T$ is the training sample size. For those from an NLP background, the above framing is exactly the same as *[language modeling](https://paperswithcode.com/task/language-modelling)*, i.e., given a sequence of tokens $x_{0:t - 1}$, we want to predict the probability of the next token $x_t$ from the vocabulary. 

In language modeling, the corpus size $M$ is in the order of tens of thousands (e.g., 30,522 in [original BERT](https://arxiv.org/abs/1810.04805), 40,478 in [original GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)) --- already, exhaustive softmax computation is way too expensive; in a large-scale recommender system with billions of items (Netflix is an exception with <10k titles), it just cannot be done. 


<!-- Positive \<query, item\> pairs are "facts" in the data, such as an app a user downloaded in a context. Negative \<query, item\> pairs are counterfactual and vast: A lack of engagement doesn't always imply irrelevance (e.g., as a sci-fi diehard, I haven't finished all sci-fi shows on Netflix, but it doesn't mean I don't like the rest); moreover, for every positively engaged item, there are astronomically more unengaged ones in the corpus. In this post, I review common negative sampling methods for selecting \<query, item\> pairs to effectively learn the two-tower network. -->

<!-- [Hierarchical Softmax](https://paperswithcode.com/method/hierarchical-softmax#:~:text=Hierarchical%20Softmax%20is%20a%20is,the%20path%20to%20that%20node.) is something people talk about but rarely works in practice. ["Candidate sampling"](https://www.tensorflow.org/extras/candidate_sampling.pdf) works well in practice, where we sample a smaller candidate set from the corpus to compute softmax for each training example. In the context of two-tower network training, for each positive \<query, item\> pair, we sample a subset of negative \<query, item\> pairs from the data rather than exhausting all negative items for the given query. The devil lies in how we perform negative sampling. As an active research area in NLP and information retrieval, negative sampling has a growing body of literature (this [repo](https://github.com/RUCAIBox/Negative-Sampling-Paper) has 100+ papers on this topic). This post focuses on a handful of negative sampling techniques most common in the industry. -->

# Negative Sampling Approaches

{{< figure src="https://www.dropbox.com/scl/fi/dez9g6p5enhzq7eagllom/Screenshot-2024-09-01-at-11.16.27-PM.png?rlkey=ye85jjtq09rhythgb6cmfqo9x&st=ltmulfdi&raw=1" width="1500">}}



<!-- outline 
1. start with candidate generation (google 2020 paper)

## In-Batch Negative Sampling -->
<!-- how to construct batch: show code -->
<!-- use murphy to explain easy vs. hard -->

<!-- ## Easy Negative Sampling

## Online Hard Negative Sampling

## Mixed Negative Sampling -->


<!-- {{< figure src="https://www.dropbox.com/scl/fi/y9me1f9n190ufevsaoh0r/Screenshot-2024-09-01-at-2.47.13-PM.png?rlkey=mg3pvbxwvpoko8uuni0x4ilw0&st=6ugc1dse&raw=1" width="1500">}}

In the case of Google Play ([Yang et al., 2020](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)), we can encode user (e.g., ID, age, app browse and download history) and context (e.g., language, region, time of day, etc.) features on the query side, and app features (e.g., ID, category, app name + description, creator, etc.) on the item side. Features are concatenated before being fed into each tower. This architecture is similar to dual encoder in the NLP literature (e.g., [Yang et al., 2018](https://aclanthology.org/W18-3022/)). -->


# Case Studies

## LinkedIn: Job Retrieval

## Facebook: People Search

## Taobao: Product Retrieval


# Learn More
## Papers

1. Covington P, Adams J, Sargin E. [Deep neural networks for YouTube recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/). *RecSys '16*.
2. Yi X, Yang J, Hong L, Cheng DZ, Heldt L, Kumthekar A, Zhao Z, Wei L, Chi E. [Sampling-bias-corrected neural modeling for large corpus item recommendations](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/). *RecSys '19*.
3. Yang J, Yi X, Zhiyuan Cheng D, Hong L, Li Y, Xiaoming Wang S, Xu T, Chi EH. [Mixed negative sampling for learning two-tower neural networks in recommendations](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/). *WWW '20*.
4. Xu L, Lian J, Zhao WX, Gong M, Shou L, Jiang D, Xie X, Wen JR. [Negative sampling for contrastive representation learning: A review](https://arxiv.org/abs/2206.00212). *arXiv:2206.00212*.

## Blogposts
1. [What is Candidate Sampling](https://www.tensorflow.org/extras/candidate_sampling.pdf) by TensorFlow