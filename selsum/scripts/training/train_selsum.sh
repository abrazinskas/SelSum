#!/bin/bash

# general settings
PROJECT_NAME=selsum
BART_DIR=artifacts/bart
CHECKPOINT=$BART_DIR/bart.base.pt
DATA_DIR=../data/form/
LOG_INTERVAL=50
LOG_FILE=logs-$PROJECT_NAME.txt
SAVE_DIR=checkpoints/$PROJECT_NAME

# general hyper-params
LR=3e-05
WARMUP=5000
EPOCHS=14
NSENTS=5

# posterior hyper-params
Q_HIDDEN_DIM=250
Q_ENCODER_NLAYERS=2
Q_ENCODER_DROPOUT=0.1
NDOCS=10  # the number of reviews to select from each collection
SEL_SAMPLE_NUM=3
SEL_STEP_NUM=1
BASELINE_SAMPLE_NUM=3

python selsum/train_selsum.py --data=$DATA_DIR --bpe=gpt2 --user-dir=. --bart-dir=$BART_DIR  \
--save-dir=$SAVE_DIR --restore-file=$CHECKPOINT --task=selsum_task \
--layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed \
--memory-efficient-fp16 --arch=selsum --criterion=nelbo --dropout=0.15 \
--attention-dropout=0.1 --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9, 0.999)" \
--adam-eps=1e-08 --clip-norm=0.1 --lr=$LR --lr-scheduler=fixed \
--max-update=0 --warmup-updates=$WARMUP --max-epoch=$EPOCHS --num-workers=0 \
--reset-optimizer --required-batch-size-multiple=1 \
--reset-dataloader --reset-meters --reset-lr-scheduler \
--skip-invalid-size-inputs-valid-test --find-unused-parameters \
--sep-symb=" </s>" --ndocs=$NDOCS --max-sentences=$NSENTS \
--log-interval=$LOG_INTERVAL --shuffle --log-format=json --ddp-backend=no_c10d \
--sel-sample-num=$SEL_SAMPLE_NUM --sel-step-num=$SEL_STEP_NUM --bline-sample-num=$BASELINE_SAMPLE_NUM \
--q-encoder-hidden-dim=$Q_HIDDEN_DIM --q-encoder-nlayers=$Q_ENCODER_NLAYERS \
--q-encoder-dropout=$Q_ENCODER_DROPOUT | tee $LOG_FILE
