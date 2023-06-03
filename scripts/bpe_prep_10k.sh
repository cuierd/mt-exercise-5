#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..
data=$base/data
vocab=$base/vocab
mkdir -p $vocab

models=$base/models
mkdir -p $models

src=it
trg=en

inpsrc=$data/train.sample.tokenized.$src-$trg.$src
inptrg=$data/train.sample.tokenized.$src-$trg.$trg

num_threads=4
vocab_size=10000

# train a joint BPE model
subword-nmt learn-joint-bpe-and-vocab \
 --input $inpsrc $inptrg \
 --write-vocabulary $vocab/vocab.10k.$src $vocab/vocab.10k.$trg \
 -s $vocab_size --total-symbols -o $models/model.10k.bpe
