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

- **Retrieval** (a.k.a. "candidate generation"): Use a lightweight model to quickly select top k (in the hundreds or thousands) candidates from a large item corpus
- **Ranking** (a.k.a. "reranking"): Use a heavier model to rerank retrieved candidates so that the final candidates (in the dozens) maximize the target action probability (e.g., an ads click, a listing booking, a video watch, *etc.*)

{{< figure src="https://www.dropbox.com/scl/fi/xfzcn69xt5umpiqgiblvj/Screenshot-2024-09-01-at-3.27.32-PM.png?rlkey=co9ingrwzjdawfkj4nfg7j4eg&st=ym59f9sr&raw=1" width="500">}}

## Two-Tower Network for Retrieval

Reranking models are highly customized for each product at each company, since we need to tailor the ranking objective to bespoke business objectives and can afford complex models on a small candidate pool. For retrieval, the two-tower network is more or less the go-to choice at many companies to efficiently maximize Recall@k. 

The high-level idea is to encode the *query* (e.g., literally a query in search; a user or a user action sequence in recommendation) and *item* using two deep neural networks that halve in the number of hidden units from one layer to the next (resembling a "tower"). During training, the two-tower network learns to pull embeddings of positive \<query, item\> pairs closer and push negative pairs apart.

{{< figure src="https://www.dropbox.com/scl/fi/n0689j6kypsw3zzzjuw4v/Screenshot-2024-09-01-at-2.48.51-PM.png?rlkey=5v59u3177uv1at1ar0p4uryg1&st=hd0tp380&raw=1" width="500">}}

Unlike traditional models such as collaborative filtering or matrix factorization, not only can the two-tower network learn from engagement, but it can also encode content features of the query and the item. In the case of Google Play ([Yang et al., 2020](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)), we can encode user (e.g., ID, age, app browse and download history) and context (e.g., language, region, time of day, etc.) features on the query side, and app features (e.g., ID, category, app name + description, creator, etc.) on the item side. Features are concatenated before being fed into each tower. This architecture is similar to dual encoder in the NLP literature (e.g., [Yang et al., 2018](https://aclanthology.org/W18-3022/)).

{{< figure src="https://www.dropbox.com/scl/fi/y9me1f9n190ufevsaoh0r/Screenshot-2024-09-01-at-2.47.13-PM.png?rlkey=mg3pvbxwvpoko8uuni0x4ilw0&st=6ugc1dse&raw=1" width="1500">}}

Positive \<query, item\> pairs are "facts" in the data, such as an app a user downloaded in a context. Negative \<query, item\> pairs are counterfactual and vast: A lack of engagement doesn't always imply irrelevance (e.g., as a sci-fi diehard, I haven't finished all sci-fi shows on Netflix, but it doesn't mean I don't like the rest); moreover, for every positively engaged item, there are astronomically more unengaged ones in the corpus. In this post, I review common negative sampling methods for selecting \<query, item\> pairs to effectively learn the two-tower network.


## Recommendation as Classification

In a recommender system, retrieval can be framed as an extreme multi-class classification problem: Given a context $c$ (e.g., a search query or a user), we want to predict the probability of a given item $i$ from the corpus $I$, 

$$p(i | c) = \frac{\exp{\epsilon (c, i)}}{\sum_{i \in I} \exp{\epsilon (c, i)}},$$

where $\epsilon (c, i)$ measures how close $i$ is to $c$. For those from an NLP background, such framing is exactly the same as *[language modeling](https://paperswithcode.com/task/language-modelling)*: Given a sequence of tokens $x_{0:t - 1}$, we want to predict the probability of the next token $x_t$ from the vocabulary $X$. 

In natural languages, vocabulary sizes are typically in the tens of thousands (e.g., 30,522 in [original BERT](https://arxiv.org/abs/1810.04805), 40,478 in [original GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)), making it already too expensive to consider the entire vocabulary when we compute token probabilities. A large-scale recommender system can have billions of items (Netflix is an exception with <10k titles), making exhaustive softmax computation outright intractable.

A family of solutions is called ["candidate sampling"](https://www.tensorflow.org/extras/candidate_sampling.pdf), where we sample a smaller set of candidates from the corpus for each training example. In the context of learning the two-tower network, for each positive \<query, item\> pair, we sample a subset of negative \<query, item\> pairs from the training data rather than exhausting all negative items for the given query. The devil lies in how we perform negative sampling. 

Negative sampling is an active research area in both NLP and information retrieval. For those who are curious, check out this [repo](https://github.com/RUCAIBox/Negative-Sampling-Paper) with over 100 papers on this topic. In this post, let's review a few negative sampling methods common in the industry.

# Negative Sampling Techniques

<!-- outline 
1. start with candidate generation (google 2020 paper)

## In-Batch Negative Sampling -->
<!-- how to construct batch: show code -->
<!-- use murphy to explain easy vs. hard -->

<!-- ## Easy Negative Sampling

## Online Hard Negative Sampling

## Mixed Negative Sampling -->


# Case Studies

## LinkedIn: Job Retrieval

## Facebook: People Search

## Taobao: Product Retrieval


# Learn More
## Papers

1. P. Covington, J. Adams, and E. Sargin. [Deep neural networks for YouTube recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/). *RecSys '16*.
2. Yang J, Yi X, Zhiyuan Cheng D, Hong L, Li Y, Xiaoming Wang S, Xu T, Chi EH. [Mixed negative sampling for learning two-tower neural networks in recommendations](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/). *WWW '20*.

## Blogposts
1. [What is Candidate Sampling](https://www.tensorflow.org/extras/candidate_sampling.pdf) by TensorFlow