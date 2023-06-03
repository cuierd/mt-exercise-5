#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..
data=$base/data
vocab=$base/vocab

src=it
trg=en

num_threads=4

# train a joint BPE model
OMP_NUM_THREADS=$num_threads python $base/tools/joeynmt/scripts/build_vocab.py \
$base/configs/transformer_iten_bpe_2000.yaml --joint
