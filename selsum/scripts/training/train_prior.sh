#!/bin/bash

# general settings
PROJECT_NAME=prior
BART_DIR=artifacts/bart
TAG_DIR=artifacts/output/q_sel/
CHECKPOINT=artifacts/checkpoints/selsum.pt
DATA_DIR=../data/form
LOG_INTERVAL=50
LOG_FILE=$PROJECT_NAME.txt
SAVE_DIR=checkpoints/$PROJECT_NAME

# general hyper-parameters
LR=1e-05
MAX_TOKENS=150
LR_SCHEDULER=fixed
WARMUP=5000
EPOCHS=8

# prior-specific hyper-parameters
WS_HIDDEN_DIM=100
PROJ_HIDDEN_DIM=100
PROJ_NLAYERS=2
PROJ_NCLASSES=3
PROJ_DROPOUT=0.1

python selsum/train.py --data=$DATA_DIR --tag-path=$TAG_DIR --bpe=gpt2 --user-dir=. --bart-dir=$BART_DIR  \
--save-dir=$SAVE_DIR --restore-file=$CHECKPOINT --task=doc_tagging_task \
--reset-dataloader  --reset-meters --reset-lr-scheduler --reset-optimizer \
--memory-efficient-fp16 --arch=prior --criterion=multi_tagging \
--weight-decay=0.01 --optimizer=adam --adam-betas="(0.9, 0.999)" \
--adam-eps=1e-08 --clip-norm=0.1 --lr=$LR --lr-scheduler=$LR_SCHEDULER \
--warmup-updates=$WARMUP --max-epoch=$EPOCHS --max-tokens=$MAX_TOKENS \
--num-workers=0 --skip-invalid-size-inputs-valid-test --find-unused-parameters \
--log-interval=$LOG_INTERVAL --shuffle --log-format=json --min-docs=20 \
--ws-hidden-dim=$WS_HIDDEN_DIM --proj-hidden-dim=$PROJ_HIDDEN_DIM --proj-nlayers=$PROJ_NLAYERS \
--proj-nclasses=$PROJ_NCLASSES --proj-dropout=$PROJ_DROPOUT | tee $LOG_FILE
