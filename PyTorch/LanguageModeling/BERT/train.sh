#!/bin/bash

module load plgrid/tools/python/3.6.5
module load plgrid/apps/cuda/10.0

export LD_LIBRARY_PATH=/net/people/plgapohl/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/net/people/plgapohl/cudnn/cuda/lib64:$LIBRARY_PATH
export CPATH=/net/people/plgapohl/cudnn/cuda/include:$CPATH

ALBERT_PATH=/net/people/plgapohl/scratch/albert/
EXPERIMENT=bert_base_pl_2/
DATA_DIR=${ALBERT_PATH}wiki-pl-converted/
MODEL_DIR=${ALBERT_PATH}${EXPERIMENT}
LOG_DIR=${ALBERT_PATH}${EXPERIMENT}

mkdir -p $MODEL_DIR
cp bert_config.json $MODEL_DIR
cp train.sh $MODEL_DIR

unset PYTHONPATH
source ~/python-albert-pytorch/bin/activate

python -m torch.distributed.launch --nproc_per_node=8 run_pretraining.py \
            --input_dir ${DATA_DIR} \
            --config_file ./bert_config.json \
            --max_seq_length 128 \
            --train_batch_size 256 \
            --learning_rate 0.000625 \
            --max_steps 1000000 --warmup_proportion 0.003 \
            --fp16 --checkpoint_activations \
            --num_steps_per_checkpoint 2000 \
            --phase1_end_step 800000 \
            --output_dir ${MODEL_DIR} \
            --log_freq 10 \
            --resume_from_checkpoint \
            --resume_step 30000 \
            --do_train
