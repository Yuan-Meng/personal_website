---
title: Attention as Soft Dictionary Lookup
date: 2024-03-09
math: true
tags:
    - machine learning
    - natural language processing
categories:
- papers
keywords:
    - machine learning, natural language processing
include_toc: true
---

# The Dictionary Metaphor 🔑📚

By now, the scaled-dot product attention formula might have burned into our brains 🧠, $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$, yet still this brilliant metaphor from Kevin Murphy's PML book gives it a refreshed interpretation --- 

> "We can think of attention as a soft dictionary look up, in which we compare the query $q$ to each key $k_i$, and then retrieve the corresponding value $v_i$." --- Chapter 15, p. 513


In a real dictionary, we query values by key and only grab the value whose key matches the query (`dict[query]`). The **attention mechanism, however, allows us to grab values from multiple keys and return their weighted sum for the given query**.

{{< figure src="https://www.dropbox.com/scl/fi/f41wr0rkn85y5tgi3ss7h/Screenshot-2024-03-09-at-5.13.45-PM.png?rlkey=fevuo6cxy5gigzp682244co1q&raw=1" width="1000" caption="Image Source: Probabilistic Machine Learning: An Introduction, Chapter 15" >}}

The more compatible a key is to the query, the higher the "attention weight" we assign to the key's value. The $i$th key's attention weight is $\frac{\exp(\alpha(q, k_i))}{\sum_{j=1}^{m}\exp(a(q, k_j))}$ --- attention weights of all keys sum to 1. The numerator of the attention weight, $\alpha(q, k_i)$, is the "attention score". For masked tokens, we can set attention scores to a large negative number (e.g., $-10^6$) so that their attention weights will turn out close to 0. 

The most popular attention score function is the dot product between $q$ and $k_i$, scaled by the square root of the k dimension, $\sqrt{d_k}$. The scaled dot-product attention is used by the original transformer ([Vaswani et al., 2017](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)) and many of its descendants. You can replace $\alpha(q, k_i)$ with other similarity functions. 

{{< figure src="https://www.dropbox.com/scl/fi/5cfqswt24wey8xavtochd/Screenshot-2024-03-09-at-6.26.24-PM.png?rlkey=ei4kj7a3n98hpn5no0cuvemuk&raw=1" width="300" >}}

# Hugging Face Implementation 🦜🤗

I heard that in increasingly more MLE interviews are candidates asked to code components of the transformer architecture from scratch. I find the implementation in the Hugging Face [book](https://www.amazon.com/Natural-Language-Processing-Transformers-Revised-dp-1098136799/dp/1098136799/ref=dp_ob_title_bk) the easiest to follow and perhaps closest to the day-to-day coding style of NLP practitioners. Let's begin by reviewing key concepts such as **self-attention** and **multi-headed attention** and then code up a basic `TransformerForSequenceClassification` model relying on the transformer encoder.


## Self-Attention

The transformer encoder uses the **self-attention** between each input token and all other input tokens to create *contextual token embeddings*. Before encoding, homophones ("flies" in "time flies like an arrow" and "fruit flies like a banana") have the same initial embedding. In the first sentence, however, the token "flies" attends most strongly to "time" and "arrow", so its contextual embedding will be close to that of these two, whereas in the second sentence, "flies" attends most strongly to "fruit" and "banana", so its embedding will be close to theirs. 

After encoding, tokens attain new embeddings from associated tokens --- as the developmental psychologist Jean Piaget put it, *"Through others we become ourselves."*

{{< figure src="https://www.dropbox.com/scl/fi/kw2f34jsz04se0r4pvc7q/Screenshot-2024-03-09-at-6.35.57-PM.png?rlkey=zdtgbli16j19jegvxc7tkwhci&raw=1" width="400" caption="The `bertviz` package can visualize the attention weight between tokens." >}}


## Multi-Headed Attention

The original transformer paper pioneered the multi-head attention (MHA), where each "head" could capture a different notion of similarity (e.g., semantic, syntactic). The query $Q$, key $K$, and value $V$ matrices are split along the embedding dimension, $d_{model}$, and each split with the embedding size $d_{model} / h$ is fed to each head. Outputs from each head are concatenated to form a single output tensor --- $\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W^O$ --- before being passed to the linear layer.

> "By dividing the `hidden_dim`, each head indeed sees the entire sequence (`seq_len`) but only a "slice" or portion of each token's embedding dimension (`head_dim`). This design enables the model to parallelly attend to information from different representation subspaces at different positions, enriching the model's ability to capture diverse relationships within the data." --- *Natural Language Processing with Transformers*

{{< figure src="https://www.dropbox.com/scl/fi/qi9srxe7snob76jsiy8rn/Screenshot-2024-03-09-at-6.26.27-PM.png?rlkey=ek30opu6d42lc3vf316jy26er&raw=1" width="300" >}}

Instead of starting with random embeddings for each token, we can use the tokenizer of pre-trained model, say `bert-base-uncased`, to encode the input text.

```python
from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

# load tokenizer from model checkpoint
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# load config associated with given model
config = AutoConfig.from_pretrained(model_ckpt)

# input text
text = "time flies like an arrow"

# tokenize input text
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

# nn.Embedding is a lookup table to find embeddings of each input_id
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

# look up embeddings by id
input_embs = token_emb(inputs.input_ids)
```

For simplicity, we can use the same linear projection of the input embedding (`input_embs`) for $Q$ (`query`), $K$ (`key`), $V$ (`value`). In practice, we usually use 3 different linear projections. The `scaled_dot_product_attention` function below takes `query`, `key`, and `value` as inputs and returns output embeddings of the tokens.

```python
# init Q, K, V with input_embs
query = key = value = input_embs

def scaled_dot_product_attention(query, key, value):
    # get key dim: hidden_dim
    dim_k = key.size(-1)  # last dim

    # attention weights
    weights = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)

    # attention scores
    scores = F.softmax(weights, dim=-1)

    # output embeddings
    return torch.bmm(scores, value)
```

The code below implements MHA with 12 heads. We can instantiate a `MultiHeadAttention` object with the pre-trained model config and call it on `input_embs` to get the attention outputs that represent the input sequence.

Note that we don't need to specifically use methods such as `encode` (or however you name it) to get the output --- when calling a model object on some input data (`hidden_state`), it invokes the `forward` function and returns the output `x`.

```python
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        # init 3 independent linear layers
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
        )
        return attn_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # embedding size is 768 in the case of "bert-base-uncased"
        embed_dim = config.hidden_size

        # conventionally, hidden_size is divisible by num_heads
        num_heads = config.num_attention_heads

        # if we have 12 heads, each head get 768 // 12 = 54 hidden_dim
        head_dim = embed_dim // num_heads

        # create a list of attention heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )

        # final linear layer
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        # concat output from each head on the last dim
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        # pass through final linear layer
        x = self.output_linear(x)
        return x

# use model config from the beginning
multihead_attn = MultiHeadAttention(config)

# attention outputs concatenated from 12 heads
attn_output = multihead_attn(input_embs)
```

## Token Embeddings

While we're at it, let's finish coding up the rest of the transformer decoder. The attention outputs are passed through a feedforward network (FFN).

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
```

Before FNN, we apply layer normalization to ensure each input has 0 mean and unity (1) variance. Moreover, to preserve information throughout the layers and alleviate the vanishing gradient problem, we can apply skip connections, which pass a tensor to the next layer without processing (`x`) and add it to the processed tensor. 

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # apply layer norm on input
        hidden_state = self.layer_norm_1(x)
        # apply attention with skip connection
        x = x + self.attention(hidden_state)
        # apply feedforward with skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        # return processed
        return x
```

## Positional Encoding

Token embeddings from `TransformerEncoderLayer` agnostic to positional information, which can be injected via positional encoding. Each position (absolute or relative) in the input sequence is represented by a unique embedding, which is learned or fixed (such as sinusoidal waves below):

- Even-indexed positions: $PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$
- Odd-indexed positions: $PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$

Check out the Machine Learning Mastery [tutorial](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/) and Lilian Weng's [blogpost](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/#positional-encoding). The code below uses positional encoding that comes with the pre-trained model.


```python
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # look up token and position embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        # define layernorm and dropout layers
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.drop_out = nn.Dropout()

    def forward(self, input_ids):
        # length of the input sequence
        seq_length = input_ids.size(1)
        # position id: [0 to seq_length - 1]
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # look up embeddings by id
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # add up token and position embeddings
        embeddings = token_embeddings + position_embeddings
        # pass through layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.drop_out(embeddings)
        return embeddings
```

The code below is the final encoder of our vanilla transformer. 


```python
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        # repeat 12 times
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x
```

## Sequence Classification

A common use case of the transformer is sequence classification, which maps input embeddings (token + positional) to probabilities of class labels.

```python
class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        # select [CLS] token
        x = self.encoder(x)[:, 0, :]
        # apply dropout on embedding
        x = self.dropout(x)
        # pass through classification layer
        x = self.classifier(x)
        return x

# specify number of labels
config.num_labels = 3
encoder_classifier = TransformerForSequenceClassification(config)
encoder_classifier(inputs.input_ids).size()
```

# Resources
1. [*Probabilistic Machine Learning: An Introduction* (2023)](https://probml.github.io/pml-book/book1.html) by Kevin P. Murphy
2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). *[Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). Advances in Neural Information Processing Systems*, 30.
3. [*Natural Language Processing with Transformers* (2022)](https://www.amazon.com/Natural-Language-Processing-Transformers-Revised-dp-1098136799/dp/1098136799/ref=dp_ob_title_bk) by Hugging Face
4. *A Gentle Introduction to Positional Encoding in Transformer Models ([Part I](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/), [Part II](https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/))*
4. [*The Transformer Family Version 2.0*](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) by Lilian Weng
