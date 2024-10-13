---
title: "Is Human Vision More like CNN or Vision Transformer?"
date: 2024-10-12
math: true
tags:
    - AI
    - cognition
categories:
- papers
keywords:
    - AI, cognition
include_toc: true
---

# Engineer the Mind

In Winter 2015, after coming back from grad school interviews in the States, I told my dad over hotpot that I was going to study cognitive science at Berkeley.


> \- *"So, what is cognitive science?"* he asked. <br/> \- *"It is the study of the mind, uncovering algorithms that might underlie human reasoning, perception, and language."* I tried my best to explain. <br/> \- *"Cool... How is that different from artificial intelligence?"* Dad 🤔. <br/> \- *"Hmm... AI engineers solutions that work, but CogSci reverse-engineers how humans think back from the solutions?"* 21-year-old me 🤓. <br/> \- *"If AI works, does it matter if it works like the mind? Since the mind already works, does it matter if we can reverse-engineer it?"* Dad 🧐. <br/> \- *"The weather today is quite nice..."* 21-year-old me 🥵. <br/>

Little did I know, nearly 10 years later as a machine learning engineer, I'd be repeating this conversation with recruiters, hiring managers, and curious colleagues, each asking my dad's questions. My answers, and perhaps the field's, have changed. 

## Marr: Purpose & Function

> <span style="background-color: #B3E59A"> *"Trying to understand perception by studying only neurons is like trying to understand bird flight by studying only feathers: It just cannot be done."* </span> --- David Marr (1982), *Vision*, p. 27.

When I started my PhD in 2016, it was before the [Transformer](https://paperswithcode.com/method/transformer). ResNet ([CVPR 2016](https://arxiv.org/abs/1512.03385)) had just surpassed humans in image classification, while higher-level cognition was still dominated by Bayesian models (see [Griffiths et al. 2010](https://www.ed.ac.uk/files/atoms/files/griffithstics.pdf), [Tenenbaum et al., 2011](https://wiki.santafe.edu/images/e/e1/HowToGrowAMind%282011%29Tenebaum_J.pdf), and [Lake et al., 2017](https://arxiv.org/abs/1604.00289), for reviews), like Computer Vision before AlexNet. Even Fei-Fei Li, the godmother of AI, began her career in Bayes (e.g., [CVPR 2004](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f6fffe049408c7b0343a1bdcefe0dc3d0256646d)).

Why Bayes? I hope my CogSci professors forgive me --- I think Bayesian models are neat "pseudocode" to capture a learner's inductive biases through priors and outcomes through posteriors, without worrying too much about the process in between.

This tradition of understanding the mind through its abstract function stems from the British neuroscientist [David Marr](https://en.wikipedia.org/wiki/David_Marr_(neuroscientist)). Tired of neuroscience's obsession with identifying one specialized neuron after another (e.g., Barlow, 1953, "bug detector"; Gross, Rocha-Miranda, & Bender, 1972, "hand detector"; and the Cambridge joke, the [apocryphal grandmother cell](https://en.wikipedia.org/wiki/Grandmother_cell) that supposedly activates when you see your grandma), Marr argued that to truly understand vision, we must step back and consider the *purpose* of vision and the *problems* it solves. This is the "computational" level of analysis, which laid the groundwork for modern computational cognitive science.

Marr provided a vivid example of how to understand an information-processing system through its purpose and function: *How do we understand a cash register*, which tells a customer how much to pay? Instead of examining each button on the machine, like neuroscientists did in the '50s and '60s, we can ask, *what should a cash register compute?* --- Addition. *Why addition and not, say, multiplication?* --- Because addition, unlike multiplication, meets the requirements for a successful transaction:

- **The rules of zero**: If you buy nothing, you pay nothing; if you buy nothing along with something, you should pay for that something.
- **Commutativity**: The order in which items are scanned shouldn't affect the total.
- **Associativity**: Grouping items into different piles shouldn't affect the total.
- **Inverse**: If you buy something and then return it, you should pay nothing.

Multiplication fails the rule of zero --- if you buy nothing along with something, you'd pay nothing, since $0 \times \mathrm{something} = 0$. So, any merchant aiming to make a profit wouldn't use a cash register that performs multiplication. Studying the buttons won't help us understand the cash register at this level, Marr argued, just as finding the grandmother cell doesn't bring us any closer to understanding vision.

Summarized below are levels at which we study the mind (artificial or natural). At the computational level, we define the constraints for a task (in domains such as vision, language, or reasoning) and identify a computation that satisfies these constraints. At the algorithmic level, we determine input/output representations, as well as the algorithm to perform the transformation. Finally, at the implementational level, we figure out how to physically implement these representations and algorithms, whether in the human brain or in a machine (like a CPU or a GPU).

{{< figure src="https://www.dropbox.com/scl/fi/p9d9vhkjwrcaeehjy817v/Screenshot-2024-10-12-at-3.44.14-PM.png?rlkey=npsrifdkyk27dqc4m6543hhyx&st=a7t01q4e&raw=1" width="650" >}}


## Hinton: Mechanism & Pretraining

Marr did not prescribe specific architectures for modeling vision, yet his vision for vision somehow contributed to the rejection of a generation of vision papers using neural nets, as Geoff Hinton recounted in his [conversation](https://www.youtube.com/watch?v=E14IsFbAbpI) with Fei-Fei Li.

> <span style="background-color: #9FD1FF"> *"It's hard to imagine now, but around 2010 or 2011, the top Computer Vision people were really adamantly against neural nets --- they were so against it that, for example, one of the main journals had a policy not to referee papers on neural nets at one point. Yann LeCun sent a paper to a conference where he had a neural net that was better at doing segmentation of pedestrians than the state of the art, and it was rejected. One of the reasons it was rejected was because one of the referees said this tells us nothing about vision --- they had **this view of how computer vision works**, which is, you study the nature of the problem of vision, you formulate an algorithm that will solve it, you implement that algorithm, and you publish a paper."* </span> --- Geoff Hinton (2023), <em>talk @Radical Ventures</em>, [29' 27''](https://youtu.be/E14IsFbAbpI?si=UJGqQYxA5oJ2tCGL&t=1767).

*"This view of how computer vision works"* clearly came from Marr, but I found it kind of sad that Marr's motivation to zoom out from the nitty-gritty when first understanding an intelligence was misinterpreted by some as staying at this abstract level forever, without getting back down to business once we know the direction.

And it wasn't just vision --- I remember Berkeley CogSci PhD students had to write seminar essays explaining why neural networks (dubbed as ["connectionism"](https://en.wikipedia.org/wiki/Connectionism) in CogSci) weren't as good a fit for higher-level cognition as Bayesian models. The recurring argument was that neural networks require too much data to train, and it's way harder to adjust weights in a neural net than to modify edges in a Bayes net. For instance, a human may misclassify a dolphin as a fish but can quickly correct the label to a mammal --- at that time, it was hard to imagine how a neural network could perform this one-shot belief updating that was straightforward in a Bayes net. 

Only after so many years can I admit -- I never really understood the contention between Bayesian models and neural nets. First of all, they're not even at the same level of analysis, with the former describing the solution to a task and the latter solving it. Moreover, just as it was underwhelming for Marr to see a collection of specialized neurons and call it "understanding vision," it felt similarly underwhelming to draw a Bayes net for a cognitive task and call it "understanding cognition," without implementing the nitty-gritty details to actually build one. Years later, I heard my doubts spoken aloud by Hinton, in that same talk with Fei-Fei (he might as well just say MIT's [Josh Tenenbaum](https://web.mit.edu/cocosci/josh.html)'s name out aloud 😆).

> <span style="background-color: #FFE88D"> *"For a long time in cognitive science, the general opinion was that if you give neural nets enough training data, they can do complicated things, but they need an awful lot of training data --- they need to see thousands of cats --- and people are much more statistically efficient. What they were really doing was comparing what an MIT undergraduate can learn to do on a limited amount of data with what a neural net that starts with random weights can learn to do on a limited amount of data. </br> </br> To make a fair comparison, you take a foundation model that is a neural net trained on lots and lots of data, give it a completely new task, and you ask how much data it needs to learn this completely new task --- and you discover these things are statistically efficient and compare favorably with people in how much data they need."* </span> --- Geoff Hinton (2023), <em>talk @Radical Ventures</em>, [46' 19''](https://youtu.be/E14IsFbAbpI?si=_OTpAbwpAquSqHQ-&t=2779).

When interviewing at Berkeley, I asked my student host why Bayesian models could magically explain how humans learn so much from so little, so quickly (Prof. [Alison Gopnik](http://alisongopnik.com/)'s catchphrase). I don't remember her answer. Today, I realize that prior knowledge sharpens the probability density around certain hypotheses. If we allow pretraining on a Bayes net, we should similarly allow pretraining on a neural net.

# Human vs. Computer Vision

Today's cognitive science is much more receptive to neural nets --- so much so that one might worry the best-performing model on machine learning benchmarks may just be viewed as the algorithm underlying the human mind. We need clever experimental designs and metrics to assess how well SOTA models align with human cognition. [Tuli et al.'s (2021)](https://arxiv.org/pdf/2105.07197) CogSci paper, comparing CNNs and Vision Transformers (ViT) to human vision, is an early effort in this direction. Below, I review the key ideas behind CNN + ViT and the authors' methodology for measuring model-human alignment.


## CNN

## ViT

## Model-Human Alignment

### Metrics

### Findings

# Cognitively Inspired AI

<!-- spatial intelligence -->

# References

## Papers
1. [*Are Convolutional Neural Networks or Transformers More Like Human Vision?*](https://arxiv.org/pdf/2105.07197) (2021) by Tuli, Dasgupta, Grant, and Griffiths, *CogSci*.
2. [*Vision Transformers Represent Relations Between Objects*](https://arxiv.org/abs/2406.15955) (2024) by Lepori et al., *arXiv*.
3. [*ImageNet Classification with Deep Convolutional Neural Networks*](https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (2012) by Krizhevsky, Sutskever, and Hinton, *NeurIPS*.
4. [*Deep Learning*](https://hal.science/hal-04206682/document) (2015) by LeCun, Bengio, and Hinton, *Nature*.
5. [*An Introduction to Convolutional Neural Networks*](https://arxiv.org/abs/1511.08458) (2015) by O'Shea and Nash, *arXiv*.
6. [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://openreview.net/forum?id=YicbFdNTTy) (2021) by Dosovitskiy et al., *ICLR*.
7. [*Vision*](https://mitpress.mit.edu/9780262514620/vision/) (1982) by Marr, *MIT Press*.
8. [*Probabilistic Models of Cognition: Exploring Representations and Inductive Biases*](https://www.ed.ac.uk/files/atoms/files/griffithstics.pdf) (2010) by Griffiths et al., *Trends in Cognitive Sciences.*
9. [*How to Grow a Mind: Statistics, Structure, and Abstraction*](https://wiki.santafe.edu/images/e/e1/HowToGrowAMind%282011%29Tenebaum_J.pdf) (2011) by Tenenbaum et al., *Science*.
10. [*Building Machines that Learn and Think Like People*](https://arxiv.org/abs/1604.00289) (2017) by Lake et al., *Behavioral and Brain Sciences*.
11. [*Levels of Analysis for Machine Learning*](https://arxiv.org/abs/2004.05107) (2020) by Hamrick, *arXiv*.
12. [*Yuan's Qualifying Exam Notes*](https://www.dropbox.com/scl/fo/tsuwr50z48813negx6f06/AOAFD4MnnU7kJG5mkgl7CdU?rlkey=1yiucza3jn1e3nwlzrvh6zg2q&st=rd86h6kb&dl=0) (2018), *UC Berkeley*.


## Talks
1. *[But What Is A Convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)* by 3Blue1Brown, *YouTube*.
2. *[Geoffrey Hinton and Fei-Fei Li in Conversation](https://youtu.be/E14IsFbAbpI?si=pGDRbakEIOHv9A5p)*, *YouTube*.
3. [*Aerodynamics For Cognition*](https://www.edge.org/conversation/tom_griffiths-aerodynamics-for-cognition) by Griffiths, *Edge*.
