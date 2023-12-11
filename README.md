

## Requirement

* Python 3
* PyTorch 0.4+
* Numpy 1.14.3+

## Usage

### Training

    python train.py conf/train.json


The training info will be outputted in standard output and log.logger\_file.


## Input Data Format

    JSON example:
    {
        "doc_label": ["mRNA", "lncRNA"],
        "doc_token": ["AAAT", "AATC", "ATCG", "TCGG"],
    }
    python dataset_preprocessing/data_preprocessor.py 
   This script convert the data into desired json format using 10-fold cross validation 
## Training & Evaluation

    python train.py conf/train.json
    
## Detail configurations and explanations

## Feature

* **min\_token\_count**
* **min\_char\_count**
* **max\_token\_dict\_size**
* **max\_char\_dict\_size**
* **max\_token\_len** No of tokens in a max length sequence
* **max\_char\_len**
* **token\_pretrained\_file**: token pre-trained embedding.

## Train

* **batch\_size**
* **eval\_train\_data**: whether evaluate training data when training.
* **start\_epoch**: start number of epochs.
* **num\_epochs**: number of epochs.
* **num\_epochs\_static\_embedding**: number of epochs that input embedding does not update.
* **decay\_rate**: Rate of decay for learning rate.
* **loss\_type**: Candidates: , "SigmodFocalCrossEntropy", "BCEWithLogitsLoss".
* **hidden\_layer\_dropout**: dropout of hidden layer.
* **visible\_device\_list**:[README.md](README.md) GPU list to use.


## Embedding

* **dimension**: dimension of embedding.
* **initializer**: Candidates: "uniform", "normal", "xavier\_uniform", "xavier\_normal", "kaiming\_uniform", "kaiming\_normal", "orthogonal".
* **fan\_mode**: Candidates: "FAN\_IN", "FAN\_OUT".
* **uniform\_bound**: If embedding_initializer is uniform, this param will be used as bound. e.g. [-embedding\_uniform\_bound,embedding\_uniform\_bound].
* **random\_stddev**: If embedding_initializer is random, this param will be used as stddev.
* **dropout**: dropout of embedding layer.


## Optimizer

* **optimizer\_type**: Candidates: "Adam", "Adadelta"
* **learning\_rate**: learning rate.
* **adadelta\_decay\_rate**: useful when optimizer\_type is Adadelta.
* **adadelta\_epsilon**: useful when optimizer\_type is Adadelta.


## Eval

* **threshold**: float trunc threshold for predict probabilities.
* **batch\_size**: batch size of evaluation.
## Log

* **logger\_file**: log file path.
* **log\_level**: Candidates: "debug", "info", "warn", "error".


### TextRNN

* **hidden\_dimension （隐藏层维度）**: dimension of hidden layer.
* **rnn\_type （RNN类型（简单RNN,长短时记忆网络LSTM,门控循环单元GRU））**: Candidates: "RNN", "LSTM", "GRU".
* **num\_layers （层数）**: number of layers.
* **doc\_embedding\_type （文本嵌入类型）**: Candidates: "AVG", "Attention", "LastHidden".
* **attention\_dimension （注意力维度）**: dimension of self-attention.
* **bidirectional （是否使用双向RNN）**: Boolean, use Bi-RNNs.



### PositionalCNN

* **kernel_size**: filter size for CNN.



