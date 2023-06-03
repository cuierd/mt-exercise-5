#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..
data=$base/data
vocab=$base/vocab
bpe10000=$data/bpe_10000

models=$base/models
mkdir -p $bpe10000

src=it
trg=en

# tokenized files for train, dev, test
inpsrc=$data/train.sample.tokenized.$src-$trg.$src
inptrg=$data/train.sample.tokenized.$src-$trg.$trg

dev_inpsrc=$data/dev.tokenized.$src-$trg.$src
dev_inptrg=$data/dev.tokenized.$src-$trg.$trg

test_inpsrc=$data/test.tokenized.$src-$trg.$src
test_inptrg=$data/test.tokenized.$src-$trg.$trg

num_threads=4
vocab_size=2000

# apply BPE model
## apply on train source file (tokenized)
subword-nmt apply-bpe -c $models/model.10k.bpe \
  --vocabulary $vocab/vocab.10k.$src \
  --vocabulary-threshold 10 \
  < $inpsrc > $bpe10000/train.$src

## apply on train target file (tokenized)
subword-nmt apply-bpe -c $models/model.10k.bpe \
  --vocabulary $vocab/vocab.10k.$trg \
  --vocabulary-threshold 10 \
  < $inptrg > $bpe10000/train.$trg

## apply on dev source file (tokenized)
subword-nmt apply-bpe -c $models/model.10k.bpe \
  --vocabulary $vocab/vocab.10k.$src \
  --vocabulary-threshold 10 \
  < $dev_inpsrc > $bpe10000/dev.$src

## apply on dev target file (tokenized)
subword-nmt apply-bpe -c $models/model.10k.bpe \
  --vocabulary $vocab/vocab.10k.$trg \
  --vocabulary-threshold 10 \
  < $dev_inptrg > $bpe10000/dev.$trg

## apply on test source file (tokenized)
subword-nmt apply-bpe -c $models/model.10k.bpe \
  --vocabulary $vocab/vocab.10k.$src \
  --vocabulary-threshold 10 \
  < $test_inpsrc > $bpe10000/test.$src

## apply on test target file (tokenized)
subword-nmt apply-bpe -c $models/model.10k.bpe \
  --vocabulary $vocab/vocab.10k.$trg \
  --vocabulary-threshold 10 \
  < $test_inptrg > $bpe10000/test.$trg
