#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data
configs=$base/configs

translations=$base/translations

mkdir -p $translations

src=it
trg=en

MOSES=$base/tools/moses-scripts/scripts

num_threads=4
device=0

# measure time

SECONDS=0

model_name=transformer_iten_word

echo "###############################################################################"
echo "model_name $model_name"

translations_sub=$translations/$model_name

mkdir -p $translations_sub

CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt translate $configs/$model_name.yaml < $data/test.tokenized.$src-$trg.$src | sed '1d;/^\s*$/d' > $translations_sub/test.tokenized.$model_name.$trg

# undo BPE

# cat $translations_sub/test.bpe.$model_name.$trg | sed 's/\@\@ //g' > $translations_sub/test.tokenized.$model_name.$trg

# undo tokenization

cat $translations_sub/test.tokenized.$model_name.$trg | $MOSES/tokenizer/detokenizer.perl -l $trg > $translations_sub/test.$model_name.$trg

# compute case-sensitive BLEU 

cat $translations_sub/test.$model_name.$trg | sacrebleu $base/data/test.$src-$trg.$trg


echo "time taken:"
echo "$SECONDS seconds"
