#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..
data=$base/data
vocab=$base/vocab
bpe2000=$data/bpe_2000

models=$base/models
mkdir -p $bpe2000

src=it
trg=en

# files for train, dev, test
inpsrc=$data/train.sample.tokenized.$src-$trg.$src
inptrg=$data/train.sample.tokenized.$src-$trg.$trg

dev_inpsrc=$data/dev.tokenized.$src-$trg.$src
dev_inptrg=$data/dev.tokenized.$src-$trg.$trg

test_inpsrc=$data/test.tokenized.$src-$trg.$src
test_inptrg=$data/test.tokenized.$src-$trg.$trg

num_threads=4
vocab_size=2000

# apply BPE model
## apply on train source file
subword-nmt apply-bpe -c $models/model.bpe \
  --vocabulary $vocab/vocab.$src \
  --vocabulary-threshold 10 \
  < $inpsrc > $bpe2000/train.$src

## apply on train target file
subword-nmt apply-bpe -c $models/model.bpe \
  --vocabulary $vocab/vocab.$trg \
  --vocabulary-threshold 10 \
  < $inptrg > $bpe2000/train.$trg

## apply on dev source file
subword-nmt apply-bpe -c $models/model.bpe \
  --vocabulary $vocab/vocab.$src \
  --vocabulary-threshold 10 \
  < $dev_inpsrc > $bpe2000/dev.$src

## apply on dev target file
subword-nmt apply-bpe -c $models/model.bpe \
  --vocabulary $vocab/vocab.$trg \
  --vocabulary-threshold 10 \
  < $dev_inptrg > $bpe2000/dev.$trg

## apply on test source file
subword-nmt apply-bpe -c $models/model.bpe \
  --vocabulary $vocab/vocab.$src \
  --vocabulary-threshold 10 \
  < $test_inpsrc > $bpe2000/test.$src

## apply on test target file
subword-nmt apply-bpe -c $models/model.bpe \
  --vocabulary $vocab/vocab.$trg \
  --vocabulary-threshold 10 \
  < $test_inptrg > $bpe2000/test.$trg
