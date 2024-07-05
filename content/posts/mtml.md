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

The Multi-gate Mixture-of-Experts (MMoE) model has as many gating networks as there are tasks. Each gate learns a specific way to leverage expert outputs for its respective task. In contrast, a One-gate Mixture-of-Experts (OMoE) model uses a single gating network to find a best way to leverage expert outputs across all tasks.

As task correlation decreases, the MMoE architecture has a larger advantage over OMoE. Both MoE models outperform Shared-Bottom, regardless of task correlation. In today's web-scale ranking systems, MMoE is by far the most widely adopted.

{{< figure src="https://www.dropbox.com/scl/fi/2kemc5gweuh71m900xsem/Screenshot-2024-07-04-at-7.53.42-PM.png?rlkey=y9evkc53ik22wjpzwcsdycvuy&st=o0rh5pqv&raw=1" width="1500">}}

# The Data ([ByteRec](https://www.biendata.xyz/competition/icmechallenge2019/))

Large-scale benchmark data played a pivotal role in the resurgence of deep learning. A prominent example is the [ImageNet dataset](https://en.wikipedia.org/wiki/ImageNet) with 14 million images from 20,000 categories, on which the CNN-based AlexNet achieved groundbreaking accuracy, outperforming non-DL models by a gigantic margin. Unlike in computer vision, ranking benchmarks are often set by companies famous for recommendation systems, such as Netflix ([Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize)) and ByteDance ([Short Video Understanding Challenge](https://www.biendata.xyz/competition/icmechallenge2019/)). 

The ByteDance data (henceforth "ByteRec") is particularly suitable for multi-task learning, since there are 2 desired user behaviors to predict --- *finish* and *share*. 


ByteRec only has a couple of features, a simplification from real search/feed logs:

{{< figure src="https://www.dropbox.com/scl/fi/5rw8nnarzv49r21wsiia9/Screenshot-2024-07-04-at-8.21.47-PM.png?rlkey=gkz9ivf2401u27m8899nm5hw5&st=yzp9uz9r&raw=1" width="1500">}}

- **Dense features**: Only `duration_time` (video watch time in seconds)
- **Sparse features**: Categorical features such as ID (`uid`, `item_id`, `author_id`, `music_id`, `channel`, `device`) and locations (`user_city`, `item_city`)
- **Targets**: Whether (`1`) or not (`0`) a user did `finish` or `share` a video

# The Code ([DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch))

For learning deep learning ranking model architectures, I find [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) (TensorFlow version: [DeepCTR](https://github.com/shenweichen/DeepCTR)) highly educational. The repo covers SOTA models spanning the last decade (e.g., Deep & Cross, DCN, DCN v2, DIN, DIEN, PNN, MMoE, etc.), even though it may not have the full functionalities needed by production-grade rankers (e.g., hash encoding for ID features). I recommend reading the [doc](https://deepctr-torch.readthedocs.io/en/latest/index.html) and the code. 

Below, I'll explain the MMoE [architecture](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/multitask/mmoe.py) and how it was used in the example [training script](https://github.com/shenweichen/DeepCTR-Torch/blob/master/examples/run_multitask_learning.py) the author provided. As with *The Annotated Transformer*, this post aims not to create an original implementation, but to provide a line-by-line explanation of an existing one. Please find my commented code in this Colab [notebook](https://colab.research.google.com/drive/1hA9K8cexY6hDLTGpYw1Dw-1kDvgIeJ_u?usp=sharing).

## Pre-Processing

{{< figure src="https://www.dropbox.com/scl/fi/f1ug02fqo56tp45e1ny4l/Screenshot-2024-07-04-at-5.00.59-PM.png?rlkey=wypm9nfd7ejnbqs4rrcf6cn42&st=nfawbdlb&raw=1" width="1000">}}

## Model Instantiation

## Model Training

# Read More
1. Ranking papers
2. DeepCTR
3. Meta E7