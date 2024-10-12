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

# Study of the Mind

In Winter 2015, after coming back from grad school interviews in the States, I told my dad over hotpot that I was going to study cognitive science at Berkeley.


> \- *"So, what is cognitive science?"* he asked. <br/> \- *"It is the study of the mind, uncovering algorithms that might underlie human reasoning, perception, and language."* I tried my best to explain. <br/> \- *"Cool... How is that different from artificial intelligence?"* Dad 🤔. <br/> \- *"Hmm... AI builds whatever that works, but CogSci wants to know the exact ways in which humans think?"* 21-year-old me 🤓. <br/> \- *"But if AI works, does it matter if it works like the mind? And since the mind already works, does it matter if we can (re-)build it?"* Dad 🧐. <br/> \- *"The weather today is quite nice..."* 21-year-old me 🥵. <br/>

Little did I know, nearly 10 years later as a machine learning engineer, I'd be repeating this conversation with recruiters, hiring managers, and curious colleagues, each asking my dad's questions. My answers, and perhaps the field's, have changed. 

## Marr: Purpose & Function

> <span style="background-color: #D9CEFF"> *"Trying to understand perception by studying only neurons is like trying to understand bird flight by studying only feathers: It just cannot be done."* </span> --- David Marr (1982), *Vision*, p. 27.

Growing tired of neuroscience's obsession with identifying one specialized neuron after another (e.g., Barlow, 1953, "bug detector"; Gross, Rocha-Miranda, & Bender, 1972, "hand detector"; and the Cambridge joke, the [apocryphal grandmother cell](https://en.wikipedia.org/wiki/Grandmother_cell) that supposedly activates when you see your grandma), the young British neuroscientist [David Marr](https://en.wikipedia.org/wiki/David_Marr_(neuroscientist)) argued that to truly understand vision, we must step back and consider the *purpose* of vision and the *problems* it solves. This is the "computational" level of analysis, which sets the foundation of modern computational cognitive science.

Marr provided a vivid example of how to understand an information-processing system through its purpose and function: *How do we understand a cash register*, which tells a customer how much to pay? Instead of examining each button on the machine, like neuroscientists in the '50s and '60s did, we can ask, *what should a cash register compute?* --- Addition. *Why addition and not, say, multiplication?* --- Because addition, unlike multiplication, meets the requirements for a successful transaction:

- **The rules of zero**: If you buy nothing, you pay nothing; if you buy nothing along with something, you should pay for that something.
- **Commutativity**: The order in which items are scanned shouldn't affect the total.
- **Associativity**: Grouping items into different piles shouldn't affect the total.
- **Inverse**: If you buy something and then return it, you should pay nothing.

Multiplication fails the rule of zero --- if you buy nothing along with something, you'd pay nothing, since $0 \times \mathrm{something} = 0$. So, any merchant aiming to make a profit wouldn't use a cash register that performs multiplication. Studying the buttons won't help us understand the cash register at this level, Marr argued, just as finding the grandmother cell doesn't bring us any closer to understanding vision.

Summarized below are levels at which we study the mind (artificial or natural). At the computational level, we define the constraints for a task (in domains such as vision, language, or reasoning) and identify a computation that satisfies these constraints. At the algorithmic level, we determine input/output representations, as well as the algorithm to perform the transformation. Finally, at the implementational level, we figure out how to physically implement these representations and algorithms, whether in the human brain or in a machine (like a CPU or a GPU).

{{< figure src="https://www.dropbox.com/scl/fi/p9d9vhkjwrcaeehjy817v/Screenshot-2024-10-12-at-3.44.14-PM.png?rlkey=npsrifdkyk27dqc4m6543hhyx&st=a7t01q4e&raw=1" width="650" >}}


## Hinton: Performance & Result

# Human vs. Computer Vision

## Before Deep Learning

## CNN

## ViT

## Error Analysis 

### Metrics

### Findings

# Thoughts

# References

## Papers
1. [*Are Convolutional Neural Networks or Transformers More Like Human Vision?*](https://arxiv.org/pdf/2105.07197) (2021) by Tuli, Dasgupta, Grant, and Griffiths, *CogSci*.
2. [*Vision*](https://mitpress.mit.edu/9780262514620/vision/) (1982) by Marr, *MIT Press*.
3. [*Levels of Analysis for Machine Learning*](https://arxiv.org/abs/2004.05107) (2020) by Hamrick, *arXiv*.
4. [*How to Grow a Mind: Statistics, Structure, and Abstraction*](https://wiki.santafe.edu/images/e/e1/HowToGrowAMind%282011%29Tenebaum_J.pdf) (2011) by Tenenbaum et al., *Science*.
5. [*ImageNet Classification with Deep Convolutional Neural Networks*](https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (2015) by Krizhevsky, Sutskever, and Hinton, *NeurIPS*.
6. [*Deep Learning*](https://hal.science/hal-04206682/document) (2015) by LeCun, Bengio, and Hinton, *Nature*.
7. [*An Introduction to Convolutional Neural Networks*](https://arxiv.org/abs/1511.08458) (2015) by O'Shea and Nash, *arXiv*.
8. [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://openreview.net/forum?id=YicbFdNTTy) (2021) by Dosovitskiy et al., *ICLR*.
9. [*Yuan's Qualifying Exam Notes*](https://www.dropbox.com/scl/fo/tsuwr50z48813negx6f06/AOAFD4MnnU7kJG5mkgl7CdU?rlkey=1yiucza3jn1e3nwlzrvh6zg2q&st=rd86h6kb&dl=0) (2018), *UC Berkeley*.


## Talks
1. *[But What Is A Convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)* by 3Blue1Brown, *YouTube*.
2. *[Geoffrey Hinton and Fei-Fei Li in Conversation](https://youtu.be/E14IsFbAbpI?si=pGDRbakEIOHv9A5p)*, *YouTube*.
3. [*Aerodynamics For Cognition*](https://www.edge.org/conversation/tom_griffiths-aerodynamics-for-cognition) by Griffiths, *Edge*.
