## Dataset Pre-processing

Below we provide instructions on how to apply preprocessing functions if a custom dataset build is needed. 
For example, one might want to increase the maximum number of reviews or discard longer reviews. 

If you will like to pre-process the **raw dataset**, follow the steps described below. If you would like to use the **FAIRSEQ formatted dataset**, then refer to §4.
Note: Please execute these scripts from the root directory of the project. 

### 1. ASIN filtering

First of all, one needs to filter ASINs by removing too short/long reviews as well as unpopular products.
 
```
python preprocessing/scripts/filter_asins.py --input-folder-path=input_folder_path --output-folder-path=output_folder_path --min-rev-len=10 --max-rev-len=120 --min-revs=10 --max-revs=100
```

### 2. FAIRSEQ formatting

This function will create FAIRSEQ input files which after BPE encoding and binarization can be used as input to FAIRSEQ models.

```
python preprocessing/scripts/fairseq_format.py --input-folder-path=input_folder_path --output-folder-path=output_folder_path
```

For later evaluation of each summary type separately, split them using the command below.

```
python preprocessing/scripts/format_summs.py --input-file-path=your_path/valid.target --output-folder-path=your_path/eval
```

### 3. Features computation

Features for the approximate posterior can be computed using the script below. 
The script additionally relies on the fine-grained aspect lexicon, which can be found at `artifacts/misc/aspect_lexicon.txt`.

```
python preprocessing/scripts/compute_features.py --data-folder-path=data_folder_path --lex-file-path=artifacts/misc/aspect_lexicon.txt
```

### 4. FAIRSEQ BPE encoding and binarization

First, set global variables as shown below.

```
PROJECT_WORK_DIR=path_to_the_project
TASK=path_to_fairseq_data_folder

ENCODER_PATH=$PROJECT_WORK_DIR/artifacts/bart/encoder.json
VOCAB_BPE=$PROJECT_WORK_DIR/artifacts/bart/vocab.bpe
DICT_PATH=$PROJECT_WORK_DIR/artifacts/bart/dict.txt
```

From fairseq folder run the BPE encoding of sequences: 

```
for SPLIT in train valid test
do
  for LANG in source target
    do
       python -m examples.roberta.multiprocessing_bpe_encoder \
       --encoder-json $ENCODER_PATH \
       --vocab-bpe $VOCAB_BPE \
       --inputs "$TASK/$SPLIT.$LANG" \
       --outputs "$TASK/$SPLIT.bpe.$LANG" \
       --workers 60 \
       --keep-empty;
  done
done
```

Finally, binarize sequences:

```
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/valid.bpe" \
  --testpref "${TASK}/test.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict $DICT_PATH \
  --tgtdict $DICT_PATH;
```
