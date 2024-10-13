---
title: "Is Human Vision More like CNN or Vision Transformer?"
date: 2024-10-13
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

In Winter 2015, after coming back from grad school interviews in the States, I told my dad over hotpot that I was going to study [cognitive science](https://en.wikipedia.org/wiki/Cognitive_science) at Berkeley.


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

{{< figure src="https://www.dropbox.com/scl/fi/p9d9vhkjwrcaeehjy817v/Screenshot-2024-10-12-at-3.44.14-PM.png?rlkey=npsrifdkyk27dqc4m6543hhyx&st=a7t01q4e&raw=1" caption="Source: David Marr's *Vision*, Chapter 1 [The Philosophy and the Approach](http://mechanism.ucsd.edu/teaching/f18/David_Marr_Vision_A_Computational_Investigation_into_the_Human_Representation_and_Processing_of_Visual_Information.chapter1.pdf)" width="650" >}}


## Hinton: Mechanism & Pretraining

Marr did not prescribe specific architectures for modeling vision, yet his vision for vision somehow contributed to the rejection of a generation of vision papers using neural nets, as Geoff Hinton recounted in his [conversation](https://www.youtube.com/watch?v=E14IsFbAbpI) with Fei-Fei Li.

> <span style="background-color: #9FD1FF"> *"It's hard to imagine now, but around 2010 or 2011, the top Computer Vision people were really adamantly against neural nets --- they were so against it that, for example, one of the main journals had a policy not to referee papers on neural nets at one point. Yann LeCun sent a paper to a conference where he had a neural net that was better at doing segmentation of pedestrians than the state of the art, and it was rejected. One of the reasons it was rejected was because one of the referees said this tells us nothing about vision --- they had **this view of how computer vision works**, which is, you study the nature of the problem of vision, you formulate an algorithm that will solve it, you implement that algorithm, and you publish a paper."* </span> --- Geoff Hinton (2023), <em>talk @Radical Ventures</em>, [29' 27''](https://youtu.be/E14IsFbAbpI?si=UJGqQYxA5oJ2tCGL&t=1767).

*"This view of how computer vision works"* clearly came from Marr, but I found it kind of sad that Marr's motivation to zoom out from the nitty-gritty when first understanding an intelligence was misinterpreted by some as staying at this abstract level forever, without getting back down to business once we know the direction.

And it wasn't just vision --- I remember Berkeley CogSci PhD students had to write seminar essays explaining why neural networks (dubbed as ["connectionism"](https://en.wikipedia.org/wiki/Connectionism) in CogSci) weren't as good a fit for higher-level cognition as Bayesian models. The recurring argument was that neural networks require too much data to train, and it's way harder to adjust weights in a neural net than to modify edges in a Bayes net. For instance, a human may misclassify a dolphin as a fish but can quickly correct the label to a mammal --- at that time, it was hard to imagine how a neural network could perform this one-shot belief updating that was straightforward in a Bayes net. 

Only after so many years can I admit -- I never really understood the contention between Bayesian models and neural nets. First of all, they're not even at the same level of analysis, with the former describing the solution to a task and the latter solving it. Moreover, just as it was underwhelming for Marr to see a collection of specialized neurons and call it "understanding vision," it felt similarly underwhelming to draw a Bayes net describing how a cognitive task should be done and call it "understanding cognition," without implementing the nitty-gritty details to build one. Years later, I heard my doubts spoken aloud by Hinton, in that same talk with Fei-Fei (he might as well just say MIT's [Josh Tenenbaum](https://web.mit.edu/cocosci/josh.html)'s name out aloud 😆).

> <span style="background-color: #FFE88D"> *"For a long time in cognitive science, the general opinion was that if you give neural nets enough training data, they can do complicated things, but they need an awful lot of training data --- they need to see thousands of cats --- and people are much more statistically efficient. What they were really doing was comparing what an MIT undergraduate can learn to do on a limited amount of data with what a neural net that starts with random weights can learn to do on a limited amount of data. </br> </br> To make a fair comparison, you take a foundation model that is a neural net trained on lots and lots of data, give it a completely new task, and you ask how much data it needs to learn this completely new task --- and you discover these things are statistically efficient and compare favorably with people in how much data they need."* </span> --- Geoff Hinton (2023), <em>talk @Radical Ventures</em>, [46' 19''](https://youtu.be/E14IsFbAbpI?si=_OTpAbwpAquSqHQ-&t=2779).

When interviewing at Berkeley, I asked my student host why Bayesian models could magically explain how humans learn so much from so little, so quickly (Prof. [Alison Gopnik](http://alisongopnik.com/)'s catchphrase). I don't remember her answer. Today, I realize that prior knowledge sharpens the probability density around certain hypotheses. If we allow pretraining on a Bayes net, we should similarly allow pretraining on a neural net.

# Human vs. Computer Vision

Today's cognitive science is much more receptive to neural nets --- so much so that one might worry the best-performing model on machine learning benchmarks may just be viewed as the algorithm underlying the human mind. We need clever experimental designs and metrics to assess how well SOTA models align with human cognition. [Tuli et al.'s (2021)](https://arxiv.org/pdf/2105.07197) CogSci paper, comparing CNNs and Vision Transformers (ViT) to human vision, is an early effort in this direction. Below, I review the key ideas behind CNN + ViT and the authors' methodology for measuring model-human alignment.


## Convolutional Neural Network (CNN)

A [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNN) is a fancier version of a feed-forward network (FNN), which extracts features through convolutional layers and pooling layers first, before feeding them to fully connected layers. But *what is a convolution*? And *what features can it extract*? These are the million-dollar questions.


### What Is a Convolution?

In math, a [convolution](https://en.wikipedia.org/wiki/Convolution) is an operation on two functions, $f$ and $g$, that creates a third function, $f * g$. That might sound a bit abstract. In his awesome [video](https://www.youtube.com/watch?v=KuXjwB4LzSA), 3Blue1Brown explains it with a classic dice example: Imagine two $N$-faced dice, each with an array of probabilities for landing on faces 1 to $N$. To find the probability of rolling a specific sum from the two dice, you use a *convolution*:


{{< figure src="https://www.dropbox.com/scl/fi/e722znugseompbmpwjtc6/Screenshot-2024-10-13-at-10.13.16-AM.png?rlkey=5xh6rr01cx8e5ca8ukernw3fw&st=xhb8vg1e&raw=1" width="500" caption="The probability of rolling a sum of 6 from 2 dice (source: [3Blue1Brown](https://www.youtube.com/watch?v=KuXjwB4LzSA)).">}}

1. Flip the second die so that its faces range from $N$ to 1, left to right;
2. Align dice with offsets 1 to $N$; sums in the overlapping region are the same;
3. Finally, to get the probability of rolling each unique sum, add the product of the probabilities from each overlapping pair of faces.

Below is a Python implementation for 1D array convolution (if this an coding interview 😅), or you could simply call `np.convolve` on the two input arrays. 

```python3
def convolve(dice1, dice2):
    # Length of the convolved array is len(dice1) + len(dice2) - 1
    n1 = len(dice1)
    n2 = len(dice2)
    result = [0] * (n1 + n2 - 1)
    
    # Perform convolution
    for i in range(n1):
        for j in range(n2):
            # Index: a unique sum
            # Value: probability of this sum
            result[i + j] += dice1[i] * dice2[j]
    
    return result

# Example 1: Two fair dice
dice1 = [1/6] * 6
dice2 = [1/6] * 6
print(convolve(dice1, dice2))
# Expected output (probabilities for sums 2 to 12):
# [0.027777777777777776, 0.05555555555555555, 0.08333333333333333, 
#  0.1111111111111111, 0.1388888888888889, 0.16666666666666669, 
#  0.1388888888888889, 0.1111111111111111, 0.08333333333333333, 
#  0.05555555555555555, 0.027777777777777776]

# Example 2: Two weighted dice
dice1 = [0.16, 0.21, 0.17, 0.16, 0.12, 0.18]
dice2 = [0.11, 0.22, 0.24, 0.10, 0.20, 0.13]
print(convolve(dice1, dice2))
# Expected output (probabilities for sums 2 to 12):
# [0.0176, 0.058300000000000005, 0.1033, 0.1214, 0.1422, 0.1644, 
#  0.1457, 0.10930000000000001, 0.06280000000000001, 
#  0.05159999999999999, 0.0234]
```

See the full results of convolving two 6-faced dice below, along with the formula.

{{< figure src="https://www.dropbox.com/scl/fi/jhrttufl0cxmsej24wtgo/Screenshot-2024-10-13-at-9.37.58-AM.png?rlkey=1lzumlvdxjfifa4d1opxbocte&st=nvc4fw39&raw=1" width="1500" caption="Probabilities of rolling possible sums from 2 dice (source: [3Blue1Brown](https://www.youtube.com/watch?v=KuXjwB4LzSA)).">}}

In a CNN, instead of convolving two 1D arrays of the same length, we convolve two 2D arrays of different dimensions --- a larger image array and a smaller $k \times k$ kernel (which, like the second 1D array, is flipped 180 degrees before applying).  


### What Features Can It Extract?

{{< figure src="https://www.dropbox.com/scl/fi/3pymdr2gleao16g5dwx1t/Screenshot-2024-10-13-at-11.27.12-AM.png?rlkey=ztglg0gj4pyybcbf8h44cvsiu&st=q83g0r6n&raw=1" width="1500" caption="An example kernel for detecting horizontal edges (source: [3Blue1Brown](https://www.youtube.com/watch?v=KuXjwB4LzSA))." >}}

*Element values in the kernel determine what features it extracts*. In the above example, element values sum up to 1, so the kernel blurs the original image by taking a moving average of neighboring pixels ("box blur"). If we allow some values in a kernel to be positive and others negative, the kernel may detect variations in pixel values and pick up on features such as vertical and horizontal edges. We can design different kernel values to detect different image features (more [examples](https://en.wikipedia.org/wiki/Kernel_(image_processing))).


In the 1D example, we considered all possible offsets between two arrays. In a CNN, however, we only compute element-wise products where the kernel is fully aligned with the original image. If the original image has dimensions $m \times n$ (ignoring the color channel for now), the output array --- or the "feature map" --- of a $k \times k$ kernel will have dimensions $(m - k + 1) \times (n - k + 1)$. This is because the kernel slides horizontally $(n - k + 1)$ times and vertically $(m - k + 1)$ times across the image.

{{< figure src="https://www.dropbox.com/scl/fi/0jlc6tkazba6ab37gvdxv/Screenshot-2024-10-13-at-11.56.49-AM.png?rlkey=xwi0e4r4186ogsp5xlkc7tfn8&st=30xd4zqs&raw=1" width="1500" caption="In a CNN, we only convolve fully aligned positions (source: [DigitalOcean](https://www.digitalocean.com/community/tutorials/writing-cnns-from-scratch-in-pytorch)).">}}

In practice, we usually add padding to keep the dimensions of each feature map at $m \times n$ instead of reducing it to $(m - k + 1) \times (n - k + 1)$. After convolution, we stack the $l$ feature maps into a tensor of size $l \times m \times n$, and then apply the ReLU activation function to each element in the tensor, setting negative values to zero.

{{< figure src="https://www.dropbox.com/scl/fi/tb263n4exx926l7vmnuuu/Screenshot-2024-10-13-at-12.25.27-PM.png?rlkey=hcf7s7bhtlq9tfwarug0n4mbu&st=sosp0r51&raw=1" width="1500" caption="Apply max pooling after convolutional layers + ReLU (source: [CS231n](https://cs231n.github.io/convolutional-networks/))." >}}

It's customary to apply max pooling after ReLU, where we use a fixed-size window to downsample each individual feature map and take the maximum value in each window. We can use a hyperparameter "stride" to control how far the window moves across the feature map --- with a stride of 2, we reduce the spatial dimensions by half. 

Using a window of size $p \times p$ and a stride of $s$, we reduce the tensor dimension to:

$$l \times \left( \left\lfloor \frac{m - p}{s} \right\rfloor + 1 \right) \times \left( \left\lfloor \frac{n - p}{s} \right\rfloor + 1 \right).$$

Finally, this dimension-reduced tensor is fed into a feed-forward network to perform the target task, such as image classification or object detection.

### Putting Together a CNN

> <span style="background-color: #D9CEFF"> *"There is no set way of formulating a CNN architecture. That being said, it would be idiotic to simply throw a few of layers together and expect it to work."* </span> --- O'Shea and Nash (2015), *[An Introduction to CNN](https://arxiv.org/abs/1511.08458)*.

{{< figure src="https://www.dropbox.com/scl/fi/6iuyho5mv6w7o5e3w6izj/Screenshot-2024-10-13-at-11.39.37-AM.png?rlkey=b2h4t1xuxo6zmb2z2fapp1bui&st=19j5vozh&raw=1" width="1500" caption="A common way to stack CNN layers (source: [O'Shea and Nash, 2015](https://arxiv.org/abs/1511.08458)).">}}

To extract complex features at increasingly levels of abstraction, we can use multiple CNN layers. A common approach is to stack two convolutional layers before each pooling layer. The code below illustrates this concept ([source](https://www.digitalocean.com/community/tutorials/writing-cnns-from-scratch-in-pytorch)).

```python
class ConvNeuralNet(nn.Module):
#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
```

## Vision Transformer (ViT)

Some say the success of CNNs in Computer Vision is no coincidence (e.g., [Yamins et al., 2014](https://www.pnas.org/doi/full/10.1073/pnas.1403112111)) --- the primate [primary visual cortex (V1)](https://en.wikipedia.org/wiki/Visual_cortex) is similar to a CNN in that it also uses local receptive fields ("kernels") with pooling to extract features from visual inputs, with increasing levels of abstraction from one layer to the next.

Transformers, on the other hand, originated from Natural Language Processing (e.g., [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) and do not bear biological similarities to the visual cortex like CNNs do, nor do they enjoy inductive biases such as translation equivariance (e.g., rotating or moving a pattern doesn't affect recognition) and locality (e.g., nearby pixels are more similar to one another than remote ones), which are inherent to CNNs. Despite lacking these inductive biases, a standard Transformer trained on large datasets (14M-300M images) performs favorably to CNNs on many benchmarks.

An image doesn't have discrete tokens like language does. To leverage the standard Transformer encoder (see this [post](https://nlp.seas.harvard.edu/annotated-transformer/), for an NLP refresher), the Vision Transformer (ViT) authors split each image into fixed-size patches and treat each patch as a token. They then apply a linear projection to embed each flattened patch. To aid classification, a learnable `[CLS]` token is prepended to the sequence of patch embeddings. Positional embeddings are added to patch embeddings to retain positional information before feeding them into the multi-headed attention and MLP blocks.

{{< figure src="https://www.dropbox.com/scl/fi/qh6a9fgumutekl5q3gav3/Screenshot-2024-10-13-at-1.08.08-PM.png?rlkey=3f98ekhp1d17pw8n14so6nynh&st=9nq5sexy&raw=1" width="1500" caption="Architecture of the Vision Transformer (source: [ Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))." >}}

It's fascinating that, with sufficient training, ViT learns to produce similar embeddings for patches in the same row or column, even though it lacks the inductive bias of CNNs that nearby patches should be similar. This observation harkens back to Hinton's [comment](http://localhost:1313/posts/human_vision/#hinton-mechanism--pretraining) that pretrained foundation models generalize as well as human learners, despite humans are endowed with even more inductive biases than CNNs.

{{< figure src="https://www.dropbox.com/scl/fi/p7wjcxeripl0yc1f7ubu9/Screenshot-2024-10-13-at-2.15.59-PM.png?rlkey=auenfi97bk1bhtyfr69x7zw9f&st=jsxjtpf7&raw=1" width="300" caption="Attention from output token to input (source: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))." >}}

## Model-Human Alignment

As far as engineers are concerned, whichever model performs better on the task at hand should be used. However, overall accuracy doesn't tell us which model behaves more like humans and, therefore, may be *closer to the nature of human vision*.

Why does the question in this blog post's title even matter? Today, as an engineer, I'm not so sure anymore. If I were to channel my CogSci professors, they might say that understanding the algorithms behind human vision is key to improving human-machine alignment and interaction. In any case, it's still a great exercise to think how we can measure and compare the alignment between models and human performance.

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
6. [*Performance-optimized Hierarchical Models Predict Neural Responses in Higher Visual Cortex*](https://www.pnas.org/doi/full/10.1073/pnas.1403112111) (2014) by Yamins et al., *PNAS*.
7. [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://openreview.net/forum?id=YicbFdNTTy) (2021) by Dosovitskiy et al., *ICLR*.
8. [*Vision*](https://mitpress.mit.edu/9780262514620/vision/) (1982) by Marr, *MIT Press*.
9. [*Probabilistic Models of Cognition: Exploring Representations and Inductive Biases*](https://www.ed.ac.uk/files/atoms/files/griffithstics.pdf) (2010) by Griffiths et al., *Trends in Cognitive Sciences.*
10. [*How to Grow a Mind: Statistics, Structure, and Abstraction*](https://wiki.santafe.edu/images/e/e1/HowToGrowAMind%282011%29Tenebaum_J.pdf) (2011) by Tenenbaum et al., *Science*.
11. [*Building Machines that Learn and Think Like People*](https://arxiv.org/abs/1604.00289) (2017) by Lake et al., *Behavioral and Brain Sciences*.
12. [*Levels of Analysis for Machine Learning*](https://arxiv.org/abs/2004.05107) (2020) by Hamrick, *arXiv*.
13. [*Yuan's Qualifying Exam Notes*](https://www.dropbox.com/scl/fo/tsuwr50z48813negx6f06/AOAFD4MnnU7kJG5mkgl7CdU?rlkey=1yiucza3jn1e3nwlzrvh6zg2q&st=rd86h6kb&dl=0) (2018), *UC Berkeley*.


## Talks
14. *[But What Is A Convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)* by 3Blue1Brown, *YouTube*.
15. *[Geoffrey Hinton and Fei-Fei Li in Conversation](https://youtu.be/E14IsFbAbpI?si=pGDRbakEIOHv9A5p)*, *YouTube*.
16. [*Aerodynamics For Cognition*](https://www.edge.org/conversation/tom_griffiths-aerodynamics-for-cognition) by Griffiths, *Edge*.
17. [*The Future of AI is Here*](https://www.youtube.com/watch?v=vIXfYFB7aBI&t=1991s) by Fei-Fei Li, *YouTube*.
