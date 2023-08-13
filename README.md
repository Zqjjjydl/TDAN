# Topic Driven Adaptive Network for Cross-Domain Sentiment Classification  

## Introduction

Code for [Topic Driven Adaptive Network for Cross-Domain Sentiment Classification](https://www.sciencedirect.com/science/article/pii/S0306457322003314).

## Requirements

* python 3.8.8
* pytorch 1.8.1
* gensim 4.0.1
* nltk 3.6.2
* numpy 1.20.2



## Environment

- OS: Ubuntu 20.04.2 LTS
- GPU: NVIDIA TITAN 1080Ti \* 2
- CUDA: 10.2



## File organization

* `preProcess.py`: the code of prepare data
* `parameter.py`: the parameter for training
* `train.py`: run all cross-domain tasks
* `model.py`: the code of TDAN
* `raw_data/`: unprocessed data
* `processedData/`: processed data
* `result/`: training result
* `model/`: model file
* `wordvec`: word embeddings

## Running

### prepare the data

Download [Google Word2Vec](https://code.google.com/archive/p/word2vec/). Extract the file and put it under the `./wordvec` folder.

Run `preProcess` to prepare the data. This step involves training of LDA so it might takes a moment. LDA is by default trained with four threads and you can adjust the thread number according to your cpu.

```
python preProcess.py
```

### Run all cross-domain tasks

```
./run.sh
```

