

## Requirement

* Python 3
* PyTorch 0.4+
* Numpy 1.19.5+

## Usage

### Training

    python train.py conf/train.json

All parameters are stored in conf/train.json.
The training info will be outputted in standard output and log.logger_file(**log_test_piRNA_hierar**).


## Input Data Format
    The data example:
    id,name,seq,mRNA,lncRNA
    1,piR-mmu-518742,TGAGTTCAAGGCCAGCATGGTCTACATAGA,1,0
    2,piR-mmu-311911,AAGGTCAGTTTCAGCTATACAAGGATTTCA,1,0
    3,piR-mmu-188768,TTACAGATAAGAGATTTTTAATTTGGTGGA,1,0
    ...
  "id" represents the number.
  "name" represents the piRNA name.
  "seq" represents piRNA sequence.
  An mRNA value of 0 indicates that the piRNA does not target the mRNA, and a value of 1 indicates that the piRNA targets the mRNA.
  A lncRNA value of 0 indicates that piRNA does not target lncRNA, and a value of 1 indicates that piRNA targets lncRNA.

    JSON example:
    {
        "doc_label": ["mRNA"], 
        "doc_token": ["TAAA", "AAAA", "AAAC", "AACA", "ACAA", "CAAG", "AAGT", "AGTA", "GTAC", "TACT", "ACTT", "CTTC", "TTCT", "TCTA", "CTAA", "TAAG", "AAGT", "AGTA", "GTAC", "TACT", "ACTA", "CTAG", "TAGA", "AGAC", "GACA", "ACAG", "CAGC", "AGCA"]

    }
    python dataset_preprocessing/data_preprocessor.py

   This script convert the data into desired json format using 10-fold cross validation 

### Training & Evaluation

    python train.py conf/train.json
    
### Detail configurations and explanations

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
* **loss\_type**: Candidates: "SigmodFocalCrossEntropy", "BCEWithLogitsLoss".
* **hidden\_layer\_dropout**: dropout of hidden layer.
* **visible\_device\_list**:GPU/CPU list to use.


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
* **model\_dir**: the directory where the trained model checkpoint or weights are stored.


## Log

* **logger\_file**: "log_test_piRNA_hierar".
* **log\_level**: Candidates: "debug", "info", "warn", "error".


### TextRNN

* **hidden\_dimension**: dimension of hidden layer.
* **rnn\_type**: Candidates: "RNN", "LSTM", "GRU".
* **num\_layers**: number of layers.
* **doc\_embedding\_type**: Candidates: "AVG", "Attention", "LastHidden".
* **attention\_dimension**: dimension of self-attention.
* **bidirectional**: "False" represents one-way, "true" represents bidirectional.


### Transformer
* **dim\_model**: the representation dimension of the model.
* **hidden**: dimension of hidden layer.
* **num\_head**: the number of heads in the attention mechanism,used to focus on different positions in different representation subspaces.
* **dropout**: dropout rate
* **num\_encoder**: the number of encoder

### Textcnn
* **kernel\_size**: the size of the convolutional kernel.
* **num\_filters**: the number of convolutional filters.
* **filter\_sizes**: the different sizes of convolutional kernels.









