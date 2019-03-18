# MTSL- Multi-Task Sequence Labeling Toolkit
-----------------------------------------------------------------
Code by [**Thai-Hoang Pham**](http://www.hoangpt.com/) at Ohio State University.

## 1. Introduction
**MTSL** is a Python implementation of the multi-task sequence labeling models described in a paper [Multi-Task 
Learning with Contextualized Word Representations for Extented Named Entity 
Recognition](https://arxiv.org/abs/1902.10118). This toolkit is used for learning one main sequence labeling task with one 
auxiliary sequence labeling task and neural language model. It can work with uncontextualized word embeddings 
(GloVe) or contextualized word embeddings (ELMo). There are three main multi-task sequence labeling models in 
this toolkit including embedding-shared model, RNN-shared model, and hierarchical-shared model. Figure 1 shows the
architectures of these multi-task models.

<figure>
    <img src="https://github.com/pth1993/MTSL/blob/master/img/models.png" width="800">
    <figcaption>Figure 1: Multi-Task Sequence Labeling Models.</figcaption>
</figure>
<p>

Our system achieves an F1-score of 83.35% which is a state-of-the-art  result for fine-grained named entity 
recognition (FG-NER) task. The following table shows the performance of **MTSL** when learning FG-NER task with other 
sequence labeling tasks.

### Results in F1 scores for FG-NER

| Model                                           | FG-NER | +Chunk    | +NER (CoNLL) | +POS  | +NER (Ontonotes) |
|-------------------------------------------------|--------|-----------|--------------|-------|------------------|
| Base Model (GloVe)                              | 81.51  | -         | -            | -     | -                |
| RNN-Shared Model (GloVe)                        | -      | 80.53     | 81.38        | 80.55 | 81.13            |
| Embedding-Shared Model (GloVe)                  | -      | 81.49     | 81.21        | 81.59 | 81.24            |
| Hierarchical-Shared Model (GloVe)               | -      | 81.65     | **82.14**    | 81.27 | 81.67            |
| Base Model (ELMo)                               | 82.74  | -         | -            | -     | -                |
| RNN-Shared Model (ELMo)                         | -      | 82.60     | 82.09        | 81.77 | 82.12            |
| Embedding-Shared Model (ELMo)                   | -      | 82.75     | 82.45        | 82.34 | 81.94            |
| Hierarchical-Shared Model (ELMo)                | -      | **83.04** | 82.72        | 82.76 | 82.96            |
| Base Model (GloVe) + LM                         | 81.77  | -         | -            | -     | -                |
| RNN-Shared Model (GloVe) + Shared-LM            | -      | 80.83     | 81.34        | 80.69 | 81.45            |
| Embedding-Shared Model (GloVe) + Shared-LM      | -      | 81.54     | 81.95        | 81.86 | 81.34            |
| Hierarchical-Shared Model (GloVe) + Shared-LM   | -      | 81.69     | **81.96**    | 81.42 | 81.78            |
| Base Model (ELMo) + LM                          | 82.91  | -         | -            | -     | -                |
| RNN-Shared Model (ELMo) + Shared-LM             | -      | 82.68     | 82.64        | 81.61 | 82.36            |
| Embedding-Shared Model (ELMo) + Shared-LM       | -      | 82.61     | 82.32        | 82.46 | 82.45            |
| Hierarchical-Shared Model (ELMo) + Shared-LM    | -      | 82.87     | 82.82        | 82.85 | 82.99            |
| Hierarchical-Shared Model (GloVe) + Unshared-LM | -      | 81.77     | 81.80        | 81.72 | 81.88            |
| Hierarchical-Shared Model (ELMo) + Unshared-LM  | -      | **83.35** | 83.14        | 83.06 | 82.82            |

## 2. Installation

This toolkit requires Python 3.6 and depends on Numpy, Scipy, Pytorch, and AllenNLP packages. You must have them 
installed before using **MTSL**.

The simple way to install them is using pip:

```sh
	$ pip install -U numpy scipy pytorch allennlp
```

**Note**: You need to create **embedding** folder inside **data** folder and put embedding files into this folder before
using **MTSL** toolkit. For Glove embeddings: download embedding file from 
[here](https://nlp.stanford.edu/projects/glove/). For ELMo embeddings: download weight and option files from 
[here](https://allennlp.org/elmo).

## 3. Usage

### 3.1. Data

The input data's format of **MSTL** follows CoNLL format. In particular, it consists of two columns: one column for word 
and then another for label. The table below describes an example sentence in chunking corpus (CoNLL-2000).

| Word             | Label |
|------------------|-------|
| His              | B-NP  |
| firm             | I-NP  |
| ,                | O     |
| along            | B-PP  |
| with             | B-PP  |
| some             | B-NP  |
| others           | I-NP  |
| ,                | O     |
| issued           | B-VP  |
| new              | B-NP  |
| buy              | I-NP  |
| recommendations  | I-NP  |
| on               | B-PP  |
| insurer          | B-NP  |
| stocks           | I-NP  |
| yesterday        | B-NP  |
| .                | O     |

**Note**: Only chunking corpus is provided in this toolkit.

### 3.2. Command-line Usage

You can use MTSL software by shell commands:

For single model:

```sh
	$ bash run_main_base_model.sh
```

For embedding-shared model:

```sh
	$ bash run_main_embedding_shared_model.sh
```

For RNN-shared model:

```sh
	$ bash run_main_RNN_shared_model.sh
```

For hierarchical-shared model:

```sh
	$ bash run_main_hierarchical_shared_model.sh
```

Arguments in these scripts:

* ``--rnn_mode``:       Architecture of RNN module (choose among RNN, LSTM, GRU)
* ``--num_epochs``:       Number of training epochs
* ``--batch_size``:       Number of sentences in each batch
* ``--hidden_size``:       Number of hidden units in RNN layer
* ``--num_layers``:       Number of layers of RNN module
* ``--num_filters``:       Number of filters in CNN layer
* ``--window``:       Window size for CNN layer
* ``--char_dim``:       Dimension of Character embeddings
* ``--learning_rate``:       Learning rate for SGD optimizer
* ``--decay_rate``:       Decay rate of learning rate
* ``--momentum``:       Momentum for SGD optimizer
* ``--gamma``:       Weight for regularization
* ``--p_rnn``:       Dropout rate for RNN layer
* ``--p_in``:       Dropout rate for embedding layer
* ``--p_out``:       Dropout rate for output layer
* ``--bigram``:       Bi-gram parameter for CRF layer
* ``--schedule``:       Schedule for learning rate decay
* ``--embedding_path``:       Path for GloVe embedding dict
* ``--option_path``:       Path for ELMo option file
* ``--weight_path``:       Path for ELMo weight file
* ``--word2index_path``:       Path for Word2Index
* ``--out_path``:       Path for output
* ``--use_crf``:       Use CRF layer for prediction (If False: use feed forward with softmax layers instead)
* ``--use_lm``:       Learn with neural language model
* ``--use_elmo``:       Use ELMo embeddings (If False: use GloVe embeddings instead) 
* ``--lm_loss``:       Scale of language model loss compared to sequence labeling loss
* ``--lm_mode``:       Use separate neural language model for each sequence labeling task or use one neural language 
model for both main and auxiliary sequence labeling tasks (choose between shared and unshared)
* ``--label_type``:       Name of labels
* ``--bucket_auxiliary``:       Buckets for training auxiliary corpus
* ``--bucket_main``:       Buckets for training main corpus
* ``--train``:       Path for train files
* ``--dev``:       Path for dev files
* ``--test``:       Path for test files

## 4. References

[Thai-Hoang Pham, Khai Mai, Nguyen Minh Trung, Nguyen Tuan Duc, Danushka Bolegala, Ryohei Sasano, Satoshi Sekine, 
"Multi-Task Learning with Contextualized Word Representations for Extented Named Entity 
Recognition"](https://arxiv.org/abs/1902.10118)

## 5. Contact

[**Thai-Hoang Pham**](http://www.hoangpt.com/) < pham.375@osu.edu >

Ohio State University