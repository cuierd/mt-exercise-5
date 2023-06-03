#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data
configs=$base/configs

MOSES=$base/tools/moses-scripts/scripts

src=it
trg=en

num_threads=4

# measure time
SECONDS=0


# include training data sub-sample preprocessing step here
#OMP_NUM_THREADS=$num_threads python $scripts/random_sub_sample_parallel_data.py \
# --sample_size 100000 \
# --src_input $data/train.$src-$trg.$src \
# --trg_input $data/train.$src-$trg.$trg \
# --src_output $data/train.sample.$src-$trg.$src \
# --trg_output $data/train.sample.$src-$trg.$trg \


# tokenization
# train sample data
 cat $data/train.sample.$src-$trg.$trg | $MOSES/tokenizer/tokenizer.perl -l $trg > $data/train.sample.tokenized.$src-$trg.$trg
 cat $data/train.sample.$src-$trg.$src | $MOSES/tokenizer/tokenizer.perl -l $src > $data/train.sample.tokenized.$src-$trg.$src

# dev data
 cat $data/dev.$src-$trg.$trg | $MOSES/tokenizer/tokenizer.perl -l $trg > $data/dev.tokenized.$src-$trg.$trg
 cat $data/dev.$src-$trg.$src | $MOSES/tokenizer/tokenizer.perl -l $src > $data/dev.tokenized.$src-$trg.$src

# test data
 cat $data/test.$src-$trg.$trg | $MOSES/tokenizer/tokenizer.perl -l $trg > $data/test.tokenized.$src-$trg.$trg
 cat $data/test.$src-$trg.$src | $MOSES/tokenizer/tokenizer.perl -l $src > $data/test.tokenized.$src-$trg.$src

echo "time taken:"
echo "$SECONDS seconds"
