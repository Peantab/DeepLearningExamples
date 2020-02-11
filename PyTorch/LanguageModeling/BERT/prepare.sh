#!/bin/bash

module load plgrid/tools/python/3.6.5
module load plgrid/apps/cuda/10.0

export LD_LIBRARY_PATH=/net/people/plgapohl/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/net/people/plgapohl/cudnn/cuda/lib64:$LIBRARY_PATH
export CPATH=/net/people/plgapohl/cudnn/cuda/include:$CPATH

ALBERT_PATH=/net/people/plgapohl/scratch/albert/
EXPERIMENT=bert_base_pl_2/
DATA_DIR=${ALBERT_PATH}wiki-pl/
CONVERTED_DATA_DIR=${ALBERT_PATH}wiki-pl-converted/
LOG_DIR=${ALBERT_PATH}${EXPERIMENT}

mkdir -p $CONVERTED_DATA_DIR
cp prepare.sh $CONVERTED_DATA_DIR

unset PYTHONPATH
source ~/python-albert-pytorch/bin/activate
python create_pretraining_data.py \
            --vocab_file ${DATA_DIR}vocab-ugc.txt \
            --input_file ${DATA_DIR}wikipedia.train.tokens \
            --output_file ${CONVERTED_DATA_DIR}wikipedia \
            --dupe_factor 3
