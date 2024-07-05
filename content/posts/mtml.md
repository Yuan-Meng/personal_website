---
title: "The Annotated Multi-Task Ranker: An MMoE Code Example"
date: 2024-07-05
math: true
tags:
    - information retrieval
    - multi-task learning
categories:
- papers
keywords:
    - information retrieval, multi-task learning
include_toc: true
---

Natural Language Processing (NLP) has an abundance of intuitively explained tutorials with code, such as Andrej Kaparthy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html), the viral [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) and its successor [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), Umar Jamil's YouTube [series](https://www.youtube.com/@umarjamilai) dissecting SOTA models and the companion [repo](https://github.com/hkproj), among others.

When it comes to Search/Ads/Recommendations ("搜广推"), however, intuitive explanations accompanied by code are rare. Company engineering blogs tend to focus on high-level system designs, and many top conference (e.g., KDD/RecSys/SIGIR) papers don't share code. In this post, I explain the iconic Multi-gate Mixture-of-Experts (MMoE) paper ([Ma et al., 2018](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)) using implementation in the popular [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) repo, to teach myself and readers how the authors' blueprint translates into code. 

# The Paper ([Ma et al., 2018](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007))

A huge appeal of deep learning is its ability to optimize for multiple task objectives at once, such as clicks and conversions in search/ads/feed ranking. In traditional machine learning, you would have to build multiple models, one per task, making the system hard to maintain for needing separate data/training/serving pipelines, and missing out on the opportunity for transfer learning between tasks.

{{< figure src="https://www.dropbox.com/scl/fi/d1ycplzd4w2kvb8jnrlel/Screenshot-2024-07-04-at-6.53.34-PM.png?rlkey=lqw5n3xubfaovsskhk57xaxm7&st=1ayhbz4q&raw=1" width="1500">}}

In early designs, all tasks shared the same backbone that feeds into task-specific towers (e.g., Caruana, [1993](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9464d15f4f8d578f93332db4aa1c9c182fd51735), [1998](https://link.springer.com/book/10.1007/978-1-4615-5529-2)). The Shared-Bottom architecture is simple and intuitive --- the drawback is that low task correlation can hurt model performance.

As a solution to the above issue, we can replace the shared bottom with a group of bottom networks ("experts"), which explicitly learn relationships between tasks and how each task uses the shared representations. This is the Mixture-of-Experts (MoE) architecture (e.g., [Jacobs et al., 1991](http://www.cs.utoronto.ca/~hinton/absps/jjnh91.ps), [Eigen et al., 2013](https://arxiv.org/pdf/1312.4314), [Shazeer et al., 2017](https://arxiv.org/pdf/1701.06538.pdf%22%20%5Ct%20%22_blank)). 

In the original Mixture of Experts (MoE) model, a "gating network" assembles expert outputs by learning each expert's weight from input features (weights sum to 1) and returning the weighted sum of expert outputs as the output to the next layer:

$$y = \sum_{i=1}^n g(x)_i f_i(x)$$

, where $g(x)_i$ is the weight of the $i$th expert, and $f_i$ is the output from that expert. 

The Multi-gate Mixture-of-Experts (MMoE) model has as many gating networks as there are tasks. Each gate learns a specific way to leverage expert outputs for its respective task. In contrast, a One-Gate Mixture-of-Experts (OMoE) model uses a single gating network to find a best way to leverage expert outputs across all tasks.

As task correlation decreases, the MMoE architecture has a larger advantage over OMoE. Both MoE models outperform Shared-Bottom, regardless of task correlation. In today's web-scale ranking systems, MMoE is by far the most widely adopted.

{{< figure src="https://www.dropbox.com/scl/fi/2kemc5gweuh71m900xsem/Screenshot-2024-07-04-at-7.53.42-PM.png?rlkey=y9evkc53ik22wjpzwcsdycvuy&st=o0rh5pqv&raw=1" width="1500">}}

# The Data ([ByteRec](https://www.biendata.xyz/competition/icmechallenge2019/))

{{< figure src="https://www.dropbox.com/scl/fi/f1ug02fqo56tp45e1ny4l/Screenshot-2024-07-04-at-5.00.59-PM.png?rlkey=wypm9nfd7ejnbqs4rrcf6cn42&st=nfawbdlb&raw=1" width="1000">}}

# The Code ([DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch))

# Read More
1. Ranking papers
2. DeepCTR
3. Meta E7