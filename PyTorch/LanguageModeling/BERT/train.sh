#!/bin/bash

module load plgrid/tools/python/3.6.5
module load plgrid/apps/cuda/10.1

export LD_LIBRARY_PATH=/net/people/plgapohl/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/net/people/plgapohl/cudnn/cuda/lib64:$LIBRARY_PATH
export CPATH=/net/people/plgapohl/cudnn/cuda/include:$CPATH

#export OMP_NUM_THREADS=5 # Sets the maximum number of threads in the parallel region. Should be set to -n value for srun

ALBERT_PATH=/net/people/plgapohl/scratch/albert/
MY_EXPERIMENTS_PATH=/net/people/plgpeantab/experiments/
EXPERIMENT=bert_base_pl_2/
DATA_DIR=${ALBERT_PATH}wiki-pl-bert-nvidia/
MODEL_DIR=${MY_EXPERIMENTS_PATH}${EXPERIMENT}
LOG_DIR=${MY_EXPERIMENTS_PATH}${EXPERIMENT}

mkdir -p $MODEL_DIR
cp bert_config.json $MODEL_DIR
cp train.sh $MODEL_DIR

unset PYTHONPATH
source ~/python-albert-pytorch/bin/activate

# bezposrednio przed do_train byly jeszcze
#            --resume_from_checkpoint \
#            --resume_step 30000 \
# oryginalny train batch size: 256, max_seq_length 128
# był torch.distributed.launch --nproc_per_node=8

python -m torch.distributed.launch --nproc_per_node=2 run_pretraining.py \
            --input_dir ${DATA_DIR} \
            --config_file ./bert_config.json \
            --max_seq_length 128 \
            --train_batch_size 84 \
            --learning_rate 0.000625 \
            --max_steps 1000000 --warmup_proportion 0.003 \
            --fp16 --checkpoint_activations \
            --num_steps_per_checkpoint 2000 \
            --phase1_end_step 800000 \
            --output_dir ${MODEL_DIR} \
            --log_freq 10 \
            --do_train
