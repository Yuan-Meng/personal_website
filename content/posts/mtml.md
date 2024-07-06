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

For learning deep learning ranking model architectures, I find [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) (TensorFlow version: [DeepCTR](https://github.com/shenweichen/DeepCTR)) highly educational. The repo covers SOTA models spanning the last decade (e.g., Deep & Cross, DCN, DCN v2, DIN, DIEN, PNN, MMoE, etc.), even though it may not have the full functionalities needed by production-grade rankers (e.g., hash encoding for ID features). There is a [doc](https://deepctr-torch.readthedocs.io/en/latest/index.html) accompanying the code. 

Below, I'll explain the MMoE [architecture](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/multitask/mmoe.py) and how it was used in the example [training script](https://github.com/shenweichen/DeepCTR-Torch/blob/master/examples/run_multitask_learning.py) the author provided. As with *The Annotated Transformer*, this post aims not to create an original implementation, but to provide a line-by-line explanation of an existing one. Please find my commented code in this Colab [notebook](https://colab.research.google.com/drive/1hA9K8cexY6hDLTGpYw1Dw-1kDvgIeJ_u?usp=sharing).


## Load Data

For testing, we can use a toy sample with 200 rows rather than the full data. At work, I also like testing models on small datasets, so I can fail and debug fast. 

```python
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import DNN, PredictionLayer
from deepctr_torch.inputs import (
    SparseFeat,
    DenseFeat,
    get_feature_names,
    combined_dnn_input,
)

# sample data with 200 rows
url = "https://raw.githubusercontent.com/shenweichen/DeepCTR-Torch/master/examples/byterec_sample.txt"

data = pd.read_csv(
    url,
    sep="\t",
    names=[
        "uid",
        "user_city",
        "item_id",
        "author_id",
        "item_city",
        "channel",
        "finish",
        "like",
        "music_id",
        "device",
        "time",
        "duration_time",
    ],
)
```

## Transform Features

While deep learning models are known for automated feature engineering, we still need to encode sparse categorical features and scale dense numerical features to a limited range (e.g., [0, 1]) before feeding data to the input layer. 

### Specify Feature + Target Columns

For easy references, we first specify different types of feature + target columns:

```python
sparse_features = [
    "uid",  # watcher's user id
    "user_city",  # watcher's city
    "item_id",  # video's id
    "author_id",  # author's user id
    "item_city",  # video's city
    "channel",  # author's channel
    "music_id",  # soundtrack id if the video contains music
    "device",  # user's device
]

dense_features = ["duration_time"]
target = ["finish", "like"]
```

### Encode Sparse Features

For each sparse feature, we can instantiate a `LabelEncoder` to assign an integer from `0` to `(n - 1)` to each of the `n` unique feature values. 
```python
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
```

This method faces the out-of-vocabulary (OOV) problem: At inference time, if a feature has a value not seen during training such as a new author ID, the model might assign a low score if unknown authors typically have low engagement in historical data, leading the model to downrank new author content in the future. One solution is [hash encoding](https://ai.plainenglish.io/harnessing-the-power-of-hash-encoding-for-categorical-data-in-data-science-d5fd3cff6673): different OOV feature values are randomly assigned to different buckets, temporarily "borrowing" the model's learning for that bucket to avoid systematic biases. Hash encoding is not implemented in DeepCTR-Torch.

### Scale Dense Features

For each dense feature, we can use a `MinMaxScaler` to cap its range to `[0, 1]`. 

```python
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])
```

If exact values don't matter, we can discretize dense features into buckets (e.g., age $\rightarrow$ age groups) and process them as categorical (e.g., one-hot encoding).

### Create Training Data

Different deep learning libraries expect input data to be in different formats. DeepCTR-Torch uses [named tuples](https://stackoverflow.com/questions/2970608/what-are-named-tuples-in-python) such as `SparseFeat` and `DenseFeat`  to store feature metadata such as names, data types, dimensions, *etc.*. See definitions [here](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/inputs.py). 

```python
# columns with sparse features
sparse_feature_columns = [
    SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
    for feat in sparse_features
]

# columns with dense features
dense_feature_columns = [
    DenseFeat(
        feat,
        1,
    )
    for feat in dense_features
]

# columns with all features
fixlen_feature_columns = sparse_feature_columns + dense_feature_columns
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
```

After the train-test split, we store the two datasets into two dictionaries, where the keys are feature names and the values are lists of feature values.

```python
# split data by rows
split_boundary = int(data.shape[0] * 0.8)
train, test = data[:split_boundary], data[split_boundary:]

# get feature names
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# prepare input dicts: {feature name : feature list}
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}
```

## Model Training

### Set Up Environment

We use GPU for training whenever available; otherwise, we use CPU instead:

```python
device = "cpu"
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print("cuda ready...")
    device = "cuda:0"
```

### Instantiate the Model

I will explain the inner working of the `MMOE` model class in the [next section](https://www.yuan-meng.com/posts/mtml/#the-anatomy-of-the-mmoe-class). 

To instantiate a new model, we need to provide a feature column list for the "deep" part of the network (our version of MMoE doesn't have a "wide" part), a list of task types (`binary` or `regression`), L2 regularization strength, and the names of the tasks (e.g., `["finish", "share"]`), and so on. In the DeepCTR-Torch API, we need to run `model.compile()` to specify the optimizer, the loss function for each task, and metrics to monitor during training before we can train the model.
 
```python
# instantiate a MMoE model
model = MMOE(
    dnn_feature_columns,
    task_types=["binary", "binary"],
    l2_reg_embedding=1e-5,
    task_names=target,
    device=device,
)

# specify optimizer, loss functions for each task, and metrics
model.compile(
    "adagrad",
    loss=["binary_crossentropy", "binary_crossentropy"],
    metrics=["binary_crossentropy"],
)
```

### Train the Model

The model training and inference API adopts the common `fit` and `predict` methods. `model.fit` takes a dictionary as features (`{feature name: feature list}`) and an `n`-dimensional array as targets (`n` being the number of tasks), along with training setup details (e.g., batch size, number of epochs, output verbosity, etc.). `model.predict` takes a similar dictionary as features and the inference batch size, and it outputs predictions for each task (dimension: `[batch_size, num_tasks]`).

```python
# fit model to training data
history = model.fit(
    train_model_input, train[target].values, batch_size=32, epochs=10, verbose=2
)
# generate predictions for test data
pred_ans = model.predict(test_model_input, 256) # inference batch size: 256
print("")
for i, target_name in enumerate(target):
    log_loss_value = round(log_loss(test[target[i]].values, pred_ans[:, i]), 4)
    auc_value = round(roc_auc_score(test[target[i]].values, pred_ans[:, i]), 4)

    print(f"{target_name} test LogLoss: {log_loss_value}")
    print(f"{target_name} test AUC: {auc_value}")
```

## The Anatomy of the `MMOE` Class

The "meat" of this post is the class definition of the `MMOE` model. You can find the source code [here](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/multitask/mmoe.py). The snippet below includes comments added by me. The model design is a faithful translation of the MMoE architecture (doodles also by me):

{{< figure src="https://www.dropbox.com/scl/fi/644b2o0qn6unmjcvzebk1/Screenshot-2024-07-05-at-1.19.37-PM.png?rlkey=lhpc6i3mebw2p28sdfh5asz74&st=smyl06ao&raw=1" width="1000">}}

- **Input**: For each training instance, embeddings are flattened and concatenated with dense features into a single vector, which feeds into gates and experts.
- **Expert outputs**: Each expert gets the same input and generates an vector output.
- **Gating outputs**: Each gate gets the same input and generates a scalar weight for each expert; for each gate, the weights of all experts sum up to 1.
- **MMoE outputs**: For each task, we use the corresponding gate network to output a weighted sum of expert outputs, which is the input to the task-specific tower. 
- **Task outputs**: Each tower gets a task-specific input and generates an output. 


Now, let's digest the model code bit by bit. You can read it in its entirety first:

```python
class MMOE(BaseModel):
    """Instantiates the multi-gate mixture-of-experts architecture.

    :param dnn_feature_columns: an iterable containing all the features used by deep part of the model.
    :param num_experts: integer, number of experts.
    :param expert_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of expert dnn.
    :param gate_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of gate dnn.
    :param tower_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific dnn.
    :param l2_reg_linear: float, l2 regularizer strength applied to linear part.
    :param l2_reg_embedding: float, l2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float, l2 regularizer strength applied to dnn.
    :param init_std: float, to use as the initialize std of embedding vector.
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given dnn coordinate.
    :param dnn_activation: activation function to use in dnn.
    :param dnn_use_bn: bool, whether use batchnormalization before activation or not in dnn.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression'].
    :param task_names: list of str, indicating the predict target of each tasks.
    :param device: str, ``"cpu"`` or ``"cuda:0"``.
    :param gpus: list of int or torch.device for multiple gpus. if none, run on `device`. `gpus[0]` should be the same gpu with `device`.

    :return: a pytorch model instance.
    """

    def __init__(
        self,
        dnn_feature_columns,  # a list feature used by the deep part
        num_experts=3,  # number of experts
        expert_dnn_hidden_units=(256, 128),  # each expert dnn has 2 layers
        gate_dnn_hidden_units=(64,),  # each gate dnn has 1 layer
        tower_dnn_hidden_units=(64,),  # each tower dnn has 1 layer
        l2_reg_linear=0.00001,  # l2 regularizer strength for linear part
        l2_reg_embedding=0.00001, # l2 regularizer strength for emb part
        l2_reg_dnn=0, # l2 regularizer strength for DNN part
        init_std=0.0001,
        seed=1024,
        dnn_dropout=0,
        dnn_activation="relu",
        dnn_use_bn=False,  # whether to use batch norm
        task_types=("binary", "binary"),
        task_names=("ctr", "ctcvr"),
        device="cpu",
        gpus=None,
    ):
        super(MMOE, self).__init__(
            linear_feature_columns=[],
            dnn_feature_columns=dnn_feature_columns,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            device=device,
            gpus=gpus,
        )
        self.num_tasks = len(task_names)  # infer task count from task names

        # performs input validations
        if self.num_tasks <= 1:
            raise ValueError(
                "num_tasks must be greater than 1"
            )  # multi-task model must have multiple tasks
        if num_experts <= 1:
            raise ValueError(
                "num_experts must be greater than 1"
            )  # multi-expert model must have multiple experts
        if len(dnn_feature_columns) == 0:
            raise ValueError(
                "dnn_feature_columns is null!"
            )  # the deep part must have features
        if len(task_types) != self.num_tasks:
            raise ValueError(
                "num_tasks must be equal to the length of task_types"
            )  # make sure we specify a type for each task
        for task_type in task_types:
            if task_type not in ["binary", "regression"]:
                raise ValueError(
                    f"task must be binary or regression, {task_type} is illegal"
                )  # make sure task type is valid

        self.num_experts = num_experts
        self.task_names = task_names
        self.input_dim = self.compute_input_dim(dnn_feature_columns)  
        self.expert_dnn_hidden_units = expert_dnn_hidden_units
        self.gate_dnn_hidden_units = gate_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units

        # expert dnn: each element is an expert network
        self.expert_dnn = nn.ModuleList(
            [
                DNN(
                    self.input_dim,
                    expert_dnn_hidden_units,
                    activation=dnn_activation,
                    l2_reg=l2_reg_dnn,
                    dropout_rate=dnn_dropout,
                    use_bn=dnn_use_bn,
                    init_std=init_std,
                    device=device,
                )
                for _ in range(self.num_experts)
            ]
        )

        # gate dnn: each element is a gate for a task
        if len(gate_dnn_hidden_units) > 0:
            self.gate_dnn = nn.ModuleList(
                [
                    DNN(
                        self.input_dim,
                        gate_dnn_hidden_units,
                        activation=dnn_activation,
                        l2_reg=l2_reg_dnn,
                        dropout_rate=dnn_dropout,
                        use_bn=dnn_use_bn,
                        init_std=init_std,
                        device=device,
                    )
                    for _ in range(self.num_tasks)
                ]
            )
            # select weights to regularize
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.gate_dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
        # a list of linear layers, one for each task
        self.gate_dnn_final_layer = nn.ModuleList(
            [
                nn.Linear(
                    gate_dnn_hidden_units[-1]
                    if len(gate_dnn_hidden_units) > 0
                    else self.input_dim,
                    self.num_experts,
                    bias=False,
                )
                for _ in range(self.num_tasks)
            ]
        )

        # tower dnn: each element is a tower for each task
        if len(tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [
                    DNN(
                        expert_dnn_hidden_units[-1],
                        tower_dnn_hidden_units,
                        activation=dnn_activation,
                        l2_reg=l2_reg_dnn,
                        dropout_rate=dnn_dropout,
                        use_bn=dnn_use_bn,
                        init_std=init_std,
                        device=device,
                    )
                    for _ in range(self.num_tasks)
                ]
            )
            # select weights to regularize
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    self.tower_dnn.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
        # a list of linear layers, one for each task
        self.tower_dnn_final_layer = nn.ModuleList(
            [
                nn.Linear(
                    tower_dnn_hidden_units[-1]
                    if len(tower_dnn_hidden_units) > 0
                    else expert_dnn_hidden_units[-1],
                    1,
                    bias=False,
                )
                for _ in range(self.num_tasks)
            ]
        )
        # each task type has an output
        self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])

        # add final parameters to be regularized
        regularization_modules = [
            self.expert_dnn,
            self.gate_dnn_final_layer,
            self.tower_dnn_final_layer,
        ]
        for module in regularization_modules:
            self.add_regularization_weight(
                filter(
                    lambda x: "weight" in x[0] and "bn" not in x[0],
                    module.named_parameters(),
                ),
                l2=l2_reg_dnn,
            )
        self.to(device)

    def forward(self, X):
        # list of embedding and dense feature values
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        # concat features into a single vector for each instance
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # expert dnn: collect output from each expert
        expert_outs = []
        for i in range(self.num_experts):
            expert_out = self.expert_dnn[i](dnn_input)
            expert_outs.append(expert_out)
        expert_outs = torch.stack(expert_outs, 1)  # (bs, num_experts, dim)

        # gate dnn: each gate has a way to combine expert outputs
        mmoe_outs = []
        for i in range(self.num_tasks):
            if (
                len(self.gate_dnn_hidden_units) > 0
            ):  # input => gate dnn => final gate layer
                gate_dnn_out = self.gate_dnn[i](dnn_input)
                gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)
            else:  # input => final gate layer
                gate_dnn_out = self.gate_dnn_final_layer[i](dnn_input)
            # performs matrix multiplication between post-softmax gate dnn output and expert outputs
            gate_mul_expert = torch.matmul(
                gate_dnn_out.softmax(1).unsqueeze(1), expert_outs
            )  # (bs, 1, dim)
            mmoe_outs.append(gate_mul_expert.squeeze())

        # tower dnn: each tower generates output for a specific task
        task_outs = []
        for i in range(self.num_tasks):
            if (
                len(self.tower_dnn_hidden_units) > 0
            ):  # input => tower dnn => final tower layer
                tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:  # input => final tower layer
                tower_dnn_logit = self.tower_dnn_final_layer[i](mmoe_outs[i])
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)  # output dimension: (bs, num_tasks)
        return task_outs
```

When coding up deep learning models, I think of the constructor (`def __init__`) as the LEGO blocks: You specify what the pieces are and the properties of each piece, so that later you can use them at your disposal. The `forward` pass is the building process --- you put the pieces together in a specific way, to allow the data (`X`) to flow through the network architecture and return the output. For each component, we'll look at the LEGO pieces and how they are used in the `forward` method.

### Model Inputs

`MMOE` inherits from `BaseModel` ([code](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/basemodel.py)), which contains components and methods shared by various model architectures. It uses the `input_from_feature_columns` method from `BaseModel` and the `combined_dnn_input` function from the `inputs` [module](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/inputs.py) to process the raw input into a format that can be consumed by the DNN model.


```python
# list of embedding and dense feature values
sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
    X, self.dnn_feature_columns, self.embedding_dict
)
# concat feature lists into input for the deep part
dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
```

The `input_from_feature_columns` method first identifies which features are sparse and which are dense. For sparse features, it performs a lookup. For each `embedding_name` (e.g., `uid`), it finds the embedding matrix, and then for a given feature value (a particular `uid`), it retrieves the corresponding embedding vector. The output is a list of embedding vectors (`sparse_embedding_list`: `[emb1, emb2, emb3, ...]`). For dense features, it returns a list of scalars (`dense_value_list`: `[val1, val2, val3, ...]`). These two lists are returned separately.


```python
def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    if not support_dense and len(dense_feature_columns) > 0:
        raise ValueError(
            "DenseFeat is not supported in dnn_feature_columns")

    sparse_embedding_list = [embedding_dict[feat.embedding_name](
        X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
        feat in sparse_feature_columns]

    sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                  varlen_sparse_feature_columns)
    varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                           varlen_sparse_feature_columns, self.device)

    dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                        dense_feature_columns]

    return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list
```


Then the `combined_dnn_input` function takes the two feature lists as inputs and outputs a matrix (`[batch_size, input_dim]`), where each row is a feature vector with all embeddings and dense features concatenated together:

- `sparse_dnn_input`: Concatenate sparse embeddings in `sparse_embedding_list` along the last dimension; flatten the result starting from the 2nd dimension.
- `dense_dnn_input`: Concatenate  dense values in `dense_value_list` along the last dimension; flatten the result starting from the 2nd dimension. 
- `dnn_input`: Concatenate `sparse_dnn_input` (if available) and `dense_dnn_input` (if available) along the feature dimension; one of them must be available.

```python
def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError
```

### Expert Networks

In this implementation, each expert network is a two-layer fully connected DNN. The first layer has 256 neurons, and the second has 128, as specified in `expert_dnn_hidden_units=(256, 128)`; they are connected by ReLU activation. `nn.ModuleList` stores and manages a list of expert networks as a single module.

```python
self.expert_dnn = nn.ModuleList(
    [
        DNN(
            self.input_dim,
            expert_dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
            init_std=init_std,
            device=device,
        )
        for _ in range(self.num_experts)
    ]
)
```

Each expert network processes `dnn_input` independently. Outputs from all experts are returned in a matrix of dimension `[batch_size, num_experts, output_dim]`.

```python
expert_outs = []
for i in range(self.num_experts):
    expert_out = self.expert_dnn[i](dnn_input)
    expert_outs.append(expert_out)
expert_outs = torch.stack(expert_outs, 1)
```

### Gating Networks

The gating network has a simpler structure than the expert network --- each gate is a one-layer DNN with 64 neurons, as specified in `gate_dnn_hidden_units=(64,)`. 

```python3
self.gate_dnn = nn.ModuleList(
    [
        DNN(
            self.input_dim,
            gate_dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
            init_std=init_std,
            device=device,
        )
        for _ in range(self.num_tasks)
    ]
)
```

The output from each gating network is passed to a linear layer to generate the weight of each expert that will be used in to combine expert outputs in each task.

```python3
self.gate_dnn_final_layer = nn.ModuleList(
    [
        nn.Linear(
            gate_dnn_hidden_units[-1]
            if len(gate_dnn_hidden_units) > 0
            else self.input_dim,
            self.num_experts,
            bias=False,
        )
        for _ in range(self.num_tasks)
    ]
)
```

Each gate takes `dnn_input` and outputs expert-specific weights, which are then multiplied with expert outputs to get the input for the corresponding tower.  

```python3
mmoe_outs = []
for i in range(self.num_tasks):
    if (
        len(self.gate_dnn_hidden_units) > 0
    ):  # input => gate dnn => final gate layer
        gate_dnn_out = self.gate_dnn[i](dnn_input)
        gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)
    else:  # input => final gate layer
        gate_dnn_out = self.gate_dnn_final_layer[i](dnn_input)
    # performs matrix multiplication between post-softmax gate dnn output and expert outputs
    gate_mul_expert = torch.matmul(
        gate_dnn_out.softmax(1).unsqueeze(1), expert_outs
    )  # (bs, 1, dim)
    mmoe_outs.append(gate_mul_expert.squeeze())
```

### Tower Networks

Similar to the gating networks above, each tower network is also a one-layer DNN with 64 neurons, as specified in `gate_dnn_hidden_units=(64,)`.

```python
self.tower_dnn = nn.ModuleList(
    [
        DNN(
            expert_dnn_hidden_units[-1],
            tower_dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
            init_std=init_std,
            device=device,
        )
        for _ in range(self.num_tasks)
    ]
)
```

The output from each tower is passed to a linear layer to generate the final prediction for a given task. Results of all tasks are collected in a `nn.ModuleList`. 

```python
# a list of linear layers, one for each task
self.tower_dnn_final_layer = nn.ModuleList(
    [
        nn.Linear(
            tower_dnn_hidden_units[-1]
            if len(tower_dnn_hidden_units) > 0
            else expert_dnn_hidden_units[-1],
            1,
            bias=False,
        )
        for _ in range(self.num_tasks)
    ]
)
# each task type has an output
self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])
```

During training ([`fit`](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/basemodel.py) in `BaseModel`), the total loss from all tasks and regularization is used to compute gradients and update weights --- `total_loss.backward()`. 

# Read More

Personally, I think deep learning ranking expertise is harder to come by than deep learning NLP expertise. Companies of any size can support fine-tuning "classic" NLP models such as BERT or BART, but many tech companies are still using Gradient Boosted Decision Trees (GBDT) in their ranking stack and struggling with the transition to deep learning. I'll be collecting sources as I learn more. Below are some resources I find particularly useful at this moment:

1. [Ranking papers](https://github.com/liyinxiao/Ranking_Papers): A repository curated by a Meta engineer, collecting industry papers published by Meta, Airbnb, Amazon, and many more.
2. Gaurav Chakravorty's [repo](https://github.com/gauravchak): Toy models created by a Meta E7 for educational purposes.