---
title: "Negative Sampling for Learning Two-Tower Networks"
date: 2024-09-02
math: true
tags:
    - negative sampling
    - recommender system
categories:
- papers
keywords:
    - negative sampling, recommender system
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

{{< figure src="https://www.dropbox.com/scl/fi/xg3m8ro5h06brz50q4rcd/Screenshot-2024-09-02-at-1.23.11-PM.png?rlkey=sn15v2btnk0aameivbjtork2j&st=zhbxc8hu&raw=1" width="500">}}

As the name suggests, the two-tower network consists of two deep neural networks, each shaped like a "tower", with every successive hidden layer halving in the number of hidden units. The left tower encodes a query $x \in \mathcal{X}$ (e.g., usually contains a user in some context, and sometimes also a search query or a seed item) whereas the right tower encodes an item $y \in \mathcal{Y}$. The query and the item towers both take sparse IDs and dense features as inputs and learn a function mapping from the input feature to a dense embedding, $u : \mathcal{X} \times \mathbb{R}^d \rightarrow \mathbb{R}^k$ and $v : \mathcal{Y} \times \mathbb{R}^d \rightarrow \mathbb{R}^k$, respectively. The output of the network is the dot product of the query and the item embeddings, 

$$s(x, y) = \langle u(x, \theta), v(y, \theta) \rangle.$$

How do we know if $s(x_i, y_i)$ is any good for the pair of query $x_i$ and item $y_i$ or not? 

Intuitively, if item $y_i$ is positive for query $x_i$, $s(x_i, y_i)$ should be greater than if $y_i$ is negative for $x_i$. After training, the model should learn to embed positive $(x_i, y_i)$ pairs closer in the embedding space and negative pairs further apart.

Formally, we can frame retrieval as extreme multi-class classification: Given the query $x_i$, predict the probability of selecting each item $y_j$ from $M$ items $\\{y_j\\}_{j=1}^M$ ---

$$p(y_i | x_i; \theta) = \frac{e^{s (x_i, y_i)}}{\sum_{j \in \[M\]} e^{s (x_i, y_j)}},$$

where $\theta$ is the model parameters. Each pair of query $x_i$ and item $y_i$ is associated with a reward $r_i \in \mathbb{R}$ denoting binary (1: positive; 0: negative) or varying degrees of user engagement. A common setup in training is that for each $x_i$, only $y_i$ is positive, whereas every $y_{i \neq j}$ is negative. A good model should predict $p(y_i | x_i; \theta) > p(y_j | x_i; \theta)$ for all $j \neq i$. How "wrong" $p(y_i | x_i; \theta)$ is can be given by the softmax loss:

$$\mathcal{l(\theta)} = -r_i \cdot \log(p(y_i | x_i; \theta)) = -r_i \cdot \log(\frac{e^{s (x_i, y_i)}}{\sum_{j \in \[M\]} e^{s (x_i, y_j)}}),$$

For those from an NLP background, the above framing is exactly the same as *[language modeling](https://paperswithcode.com/task/language-modelling)*, i.e., given a sequence of tokens $x_{0:t - 1}$, predict the probability of the next token $x_t$ from the vocabulary. In fact, the two-tower network itself is inspired by the dual encoder architecture first popularized in NLP (e.g., [Neculoiu, 2016](ttps://aclanthology.org/W16-1617.pdf)).

In language modeling, the corpus size $M$ is in the order of tens of thousands (e.g., 30,522 in [original BERT](https://arxiv.org/abs/1810.04805), 40,478 in [original GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)) --- already, exhaustive softmax computation is [way too expensive](https://arxiv.org/abs/1412.2007); in a large-scale recommender system with billions of items (Netflix is an exception with < 10k titles), it just cannot be done. 

A neat solution is *negative sampling* (or ["candidate sampling"](https://www.tensorflow.org/extras/candidate_sampling.pdf)), where we use a subset of negative items from $M$ to compute the loss for each positive $(x_i, y_i)$ pair, instead of using all negatives in the full training set. There is a vast body of literature on negative sampling since it's extremely common across NLP, CV, graphs, and information retrieval. You can find 100+ papers in this amazing [repo](https://github.com/RUCAIBox/Negative-Samplaing-Paper). This post focuses on several widely adapted approaches in web-scale recommender systems.  

# Negative Sampling Approaches

Xu et al.'s (2022) [review paper](https://arxiv.org/abs/2206.00212) organizes negative sampling approaches into the following taxonomy and distills 4 principles of what makes a good negative sampler:

{{< figure src="https://www.dropbox.com/scl/fi/dez9g6p5enhzq7eagllom/Screenshot-2024-09-01-at-11.16.27-PM.png?rlkey=ye85jjtq09rhythgb6cmfqo9x&st=ltmulfdi&raw=1" width="800">}}

1. **Efficient**: The negative sampler should have low time and space complexity;
2. **Effective**: Informative negative samples allow for fast model convergence;
3. **Stable**: The negative sampler should still work when we switch datasets;
4. **Data-independent**: The negative sampler should not rely too much on side information (e.g., query/item metadata, contexts), which may not always be available in all training examples or generalize to other examples.


## Random Negative Sampling (RNS)

The simplest approach is random negative sampling (RNS) from a uniform distribution. We draw an integer $i$ from $\mathcal{U}_{\[1, M\]}$ and put the $i$-th item from the corpus $M$ into our random negative set. We repeat this process until the desired negative sample size is reached. If a sampler doesn't outperform RNS, it is practically useless. 

> <span style="background-color: #FDB515"><b>Performance review:</b></span> Good baseline, but don't use it alone
> 1. **Efficient** ❌: We have to go through the entire corpus to sample items
> 2. **Effective** ❌: May not always return informative negatives
> 3. **Stable** ✅: No additional setup to make RNS work in new corpuses
> 3. **Data-independent** ✅: RNS doesn't rely on side information

## Batch Negative Sampling (BNS)

To avoid sampling from the entire corpus, we can sample from the mini-batch: For each $(x_i, y_i)$ pair, treat all other items in the current batch as sampled negatives. 

{{< figure src="https://www.dropbox.com/scl/fi/dbuhiukd4ugmrwe39z8cm/Screenshot-2024-09-01-at-2.45.08-PM.png?rlkey=6099axb8yp3rjq6qbsodlo4ez&st=cqum67rs&raw=1" width="800">}}

Batch negative sampling (BNS) is extremely efficient since we don't have to get negatives elsewhere and works well when the batch size is reasonable (e.g., 2048). As such, BNS is extremely commonly used in the industry. Naively, we can rewrite the loss function as follows, where the corpus $M$ is replaced with the batch $B$ ---

$$\mathcal{l(\theta)} = -r_i \cdot \log(p(y_i | x_i; \theta)) = -r_i \cdot \log(\frac{e^{s (x_i, y_i)}}{\sum_{j \in \[B\]} e^{s (x_i, y_j)}}).$$

BNS suffers from selection bias as popular items appear in many mini-batches. The probability of selecting an item follows a [unigram distribution](https://aclanthology.org/2021.findings-acl.326.pdf), where each point on the x-axis represents an item and the y-axis its probability density (perhaps the most boring distribution 😂). The related [Zipfian distribution](https://en.wikipedia.org/wiki/Zipf%27s_law) is fascinating --- in many languages, the top few popular words dominate word occurrences; for example, "the" accounts for ~7% of English word occurrences. American linguist [George Kingsley Zipf](https://en.wikipedia.org/wiki/George_Kingsley_Zipf) observed that the most common word often occurs twice as often as the second most common, three times as often as the third, dubbed as the Zipf's law:

$$\text{word frequency} \propto \frac{1}{\text{word rank}}.$$

Drawing yet another NLP analogy to recommender systems: The top few popular items dominate engagement logs. As a result, they often serve as negatives for many $(x_i, y_i)$ pairs, while long-tail negative items don't get demoted enough. Consequently, models trained on batch negatives alone often return strange long-tail items irrelevant to the users (false positives) or filter out popular items (false negatives).

There are 2 popular ways to correct for sampling bias in BNG: 
1. **logQ correction**: Correct the model logit to $s^c(x_i, y_j) = s(x_i, y_i) - \log Q(i)$, where $Q(i)$ is the sampling probability of item $j$, given by $\frac{\mathrm{count}(i)}{\sum_j \mathrm{count}(i)}$ 👉 **motivation**: Avoid over-penalizing popular items, or at least to a lesser degree
2. **Mixed Negative Sampling**: Sample additional negatives and combine them with BNS 👉 **motivation**: The model gets to see more negatives from other distributions 

> <span style="background-color: #FDB515"><b>Performance review:</b></span> Super popular, but requires bias correction
> 1. **Efficient** ✅: Can get all negatives from the mini-batch
> 2. **Effective** 🤔: Over-penalizes popular items without correction
> 3. **Stable** ✅: No additional setup to make BNS work in new corpuses
> 3. **Data-independent** ✅: BNS doesn't rely on side information

## Mixed Negative Sampling (MNS)

As mentioned, Mixed Negative Sampling (MNS) mitigates the sampling bias in BNS. In the Google paper ([Yang et al., 2020](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/)) that  introduced MNS, it was done by combining RNS and BNS: For a batch $B$, randomly draw $B'$ negatives from a uniform distribution. 

{{< figure src="https://www.dropbox.com/scl/fi/d6svgsxna1aif9wcgx8l1/Screenshot-2024-09-01-at-2.45.27-PM.png?rlkey=iigixzkybw4xbotpu440aslzi&st=f897vrvf&raw=1" width="800">}}

Under MNS, the loss function now becomes the following ---

$$\mathcal{l(\theta)} = -r_i \cdot \log(p(y_i | x_i; \theta)) = -r_i \cdot \log(\frac{e^{s (x_i, y_i)}}{\sum_{j \in \[B + B'\]} e^{s (x_i, y_j)}}).$$

$B'$ is a hyperparameter we can tune to achieve the best eval and A/B performances:
- **$B'$ is too large**: Then sampling distribution is close to the uniform distribution, which deviates from the true distribution at serving time, and also, too large a batch $B + B'$ makes training inefficient;
- **$B'$ is too small**: Then sampling distribution is close to the unigram, where long-tail items don't get many chances to be learned by the model

> <span style="background-color: #FDB515"><b>Performance review:</b></span> A good trade-off between efficiency and effectiveness
> 1. **Efficient** ✅: Can cache or share negatives across batches for $B'$
> 2. **Effective** ✅: Corrects sampling bias in the popular BNS
> 3. **Stable** 🤔: Need additional indexing work in new corpuses
> 3. **Data-independent** ✅: MNS doesn't rely on side information

## Online Hard Negative Mining (HNM)

The 3 sampling approaches above (DNS, BNS, MNS) are all static samplers, since the sampling probability for each item $j$ is fixed and independent of the model's learning progress. Researchers (e.g., [Hacohen and Weinshal, 2019](https://arxiv.org/abs/1904.03626)) found that introducing hard negatives (e.g., negative items with high predicted scores) can accelerate model convergence. This approach is "curriculum learning," and using model-generated scores to select hard negatives is "online Hard Negative Mining (HNM)". Starting with hard negatives may confuse the model, but after sufficient training on easy negatives from a static sampler, adding hard negatives helps the model align its "learned hypothesis" with the "target hypothesis," leading to faster convergence.

> <span style="background-color: #FDB515"><b>Performance review:</b></span> Great addition to MNS, but don't start with it
> 1. **Efficient** ✅: Can use the model itself to rank items in mini-batch
> 2. **Effective** ✅: Accelerates convergence of a sufficiently trained model
> 3. **Stable** ❌: Sample selection is highly reliant on the model 
> 3. **Data-independent** ❌: The model *is* the sampler and may use "side info"

# Case Studies


## Social Media

### Facebook People Search ([KDD 2020](https://dl.acm.org/doi/abs/10.1145/3394486.3403305))

> <span style="background-color: #0081FB"><b>Highlight:</b></span> If Google popularized embedding-based retrieval (EBR) in recommender systems, the Facebook Search paper sets an inspiring example for applying EBR in personalized search. Apart from standard two-tower training, it discussed *hard negative mining* and *serving challenges* in detail. 

{{< figure src="https://www.dropbox.com/scl/fi/ulxyqvmwt0puycqdnr9q4/Screenshot-2024-09-02-at-5.47.39-PM.png?rlkey=up336nmsvuxyk0bhk1954quzg&st=wqnhynoj&raw=1" width="1500">}}

Search retrieval based on *semantic* embeddings has been used in CV (e.g,, visual search/discovery) and NLP (e.g., passage retrieval) for years, but searching people on Facebook is not a pure semantic problem: When you and I both search for "John Smith", we each want to see our friends, rather than the same set of profiles. 

To personalize retrieval, the authors trained a two-tower network, where the query and the item towers both encode textual (e.g., search query, profile texts), contextual (e.g., location), and social (e.g., social graph embeddings) features. 

{{< figure src="https://www.dropbox.com/scl/fi/1hwfalp8enjs46yb6ay1f/Screenshot-2024-09-02-at-6.13.10-PM.png?rlkey=0wmz220if10ug86tstg4rjwdw&st=u14u338g&raw=1" width="500">}}

They explored various positive and negative label definitions. For positives, it didn't matter whether clicks or impressions were treated as positives --- the results were the same. For negatives, the authors compared only training on hard negatives, pure BNS, and combining the two --- the hybrid sampling approach turned out the best.


- **All hard negatives** ❌: Sampling impressed but  unclicked results as negatives led to *55% absolute recall drop* 👉 **why**: impressed results are often partial matches (e.g., a "John Smith", but not my John Smith); treating them as negatives teaches the model such matches are useless, or worse, hurt relevance
- **BNS + hard negatives** ✅: BNS was a good baseline; adding hard negatives (a "John Smith", but not my John Smith) to batch negatives improved performance 👉 **why**: forcing the model to use social features to predict for hard negatives


### LinkedIn Job Search ([KDD 2024](https://arxiv.org/abs/2402.13435))

> <span style="background-color: #0e76a8"><b>Highlight:</b></span> Interesting paper in 3 regards --- 1) the same EBR service is *shared by ads and organic search*, which have different objectives (ads: job posters care the most about if applicants are qualified; organic: job seekers are driven by interests); 2) the authors applied *curriculum training*, warming up the model with easy negatives and then switching to online hard negatives; 3) they got *exact KNN* search to work (via GPU inference)!

{{< figure src="https://www.dropbox.com/scl/fi/jkmg0ici348di9cl24god/Screenshot-2024-09-02-at-7.03.23-PM.png?rlkey=0pa4bs5zq587aj7gz5szzq39y&st=lob080d4&raw=1" width="1500">}}

Unlike other social media where viral contents are celebrated, LinkedIn has to strike a delicate balance: For each job posting, they need to deliver *sufficient* candidates, but not *overwhelmingly* so --- a tight match between seekers and jobs is key. At LinkedIn, seekers $S$ and jobs $J$ both have a rich set of metadata (e.g., job title, seniority, company, etc.). From confirmed hire data, they learned important links between seeker and job metadata. This graph powered LinkedIn retrieval. 

{{< figure src="https://www.dropbox.com/scl/fi/vdubss5sl610zfnin213l/Screenshot-2024-09-02-at-7.23.37-PM.png?rlkey=c5309d5i9fh1a02do3uz0gytl&st=cvn1w3ot&raw=1" width="1500">}}

When first deployed, EBR failed to beat the baseline. The team found 2 solutions:
- **Rules**: Enforce term-based matching rules to address relevance defects
- **Exact KNN search**: Due to the sheer volume of documents and strict latency requirements, modern recommender systems and search engines have embraced Approximate Nearest Neighbor search, which sacrifices accuracy for speed (see my post for a [review](https://www.yuan-meng.com/posts/ebr/)) 👉 LinkedIn, somehow, got exact KNN to work and saw relevance improvements; this was achieved by mixing term-based retrieval (TBR) with EBR and using GPUs to speed up matrix operations behind EBR 

## E-Commerce

### JD.com Product Search ([SIGIR 2020](https://dl.acm.org/doi/abs/10.1145/3397271.3401446))

> <span style="background-color: #E31D1A"><b>Highlight:</b></span> The author crafted a multi-head design for the query tower to capture *polysemy* (one word can have different semantics --- e.g., "apple" can either be an electronics brand or a fruit) and generalized the scoring function in the two-tower network output from dot product to attention

{{< figure src="https://www.dropbox.com/scl/fi/fliq5gl9po3863xmn7kom/Screenshot-2024-09-02-at-7.44.08-PM.png?rlkey=9xkry47poqxy9zuqov49e34ru&st=l3kmiqpy&raw=1" width="1500">}}

The two-tower output scoring function $s(x, y) = \langle u(x, \theta), v(y, \theta) \rangle$ makes 2 assumptions:
- **Output quantity**: Each tower outputs a single embedding for scoring
- **Scoring method**: We apply dot product on embeddings to get the final logit

Both assumptions were modified to improve product search at JD.com ---
- **Output quantity**: The query tower outputs multiple embeddings, to capture semantics of polysemous queries 👉 the authors indeed found one head retrieved fruits and the other cellphones when the user searched for "apple" on JD.com
- **Scoring method**: Correspondingly, the authors changed the scoring function to a weighted sum of $m$ dot products between $m$ query embeddings and 1 item embedding and the loss function to attention loss

The new architecture, Deep Personalized and Semantic Retrieval (DPSR), led to a +10.03% CVR gain in long-tail queries and has been in production at JD since 2019.

### Amazon Product Search ([KDD 2019](https://www.amazon.science/publications/semantic-product-search))

> <span style="background-color: #ff9900"><b>Highlight:</b></span> Amazon's two-tower model couldn't distinguish between positive vs. negative items, until the authors separated out impressed but not purchased items 


{{< figure src="https://www.dropbox.com/scl/fi/7f8euj8cbalwzlq216ef5/Screenshot-2024-09-02-at-8.15.12-PM.png?rlkey=z09lqbphotiwg2ffhrqj5yjyw&st=2zpvqvpz&raw=1" width="1500">}}



# Learn More
## Papers

1. [*Deep Neural Networks for YouTube Recommendations*](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) (2016) by Covington, Adams, and Sargin, *RecSys*.
2. [*Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) (2019) by Yi et al., *RecSys*.
3. [*Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations*](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/) (2020) by Yang et al., *WWW*.
4. [*Negative Sampling for Contrastive Representation Learning: A Review*](https://arxiv.org/abs/2206.00212) (2022) by Xu et al., *arXiv*.
5. [*On Using Very Large Target Vocabulary for Neural Machine Translation*](https://arxiv.org/abs/1412.2007) (2014) by Jean, Cho, Memisevic, and Bengio, *arXiv*.
6. [*Learning Text Similarity with Siamese Recurrent Networks*](https://aclanthology.org/W16-1617.pdf) (2016) by Neculoiu, Versteegh, and Rotaru, *Rep4NLP@ACL*.
7. [*On The Power of Curriculum Learning in Training Deep Networks*](https://arxiv.org/abs/1904.03626) (2019) by Hacohen and Weinshall, *ICML*.
8. [*Does Negative Sampling Matter? A Review with Insights into its Theory and Applications*](https://arxiv.org/html/2402.17238v1) (2024) by Yang et al., *PAMI*.
9. [*Embedding-based Retrieval in Facebook Search*](https://dl.acm.org/doi/abs/10.1145/3394486.3403305) (2020) by Huang et al., *KDD*.
10. [*Learning to Retrieve for Job Matching*](https://arxiv.org/abs/2402.13435) (2024) by Shen et al., *KDD*.
11. [*Semantic Product Search*](https://www.amazon.science/publications/semantic-product-search) (2019) by Nigam et al., *KDD*.
12. [*Towards Personalized and Semantic Retrieval: An End-to-End Solution for E-commerce Search via Embedding Learning*](https://dl.acm.org/doi/abs/10.1145/3397271.3401446) (2020) by Zhang et al., *SIGIR*.

## Blogposts + Repos
1. [*What is Candidate Sampling*](https://www.tensorflow.org/extras/candidate_sampling.pdf), tutorial by TensorFlow
2. [*Negative-Sampling-Paper*](https://github.com/RUCAIBox/Negative-Sampling-Paper#curriculum-learning), SOTA paper collection by [RUCAIBox](https://github.com/RUCAIBox) 👉 more on recommender systems: [Awesome-RSPapers](https://github.com/RUCAIBox/Awesome-RSPapers) and [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets)
3. [*An Introduction to Embedding-Based Retrieval*](https://www.yuan-meng.com/posts/ebr/), blogpost by Yuan