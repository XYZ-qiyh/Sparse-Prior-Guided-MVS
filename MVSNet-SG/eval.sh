#!/usr/bin/env bash
DTU_TESTING="/mnt/B/MVS_GT/dtu/"
CKPT_FILE="./ckpt/model_000015.ckpt"

export CUDA_VISIBLE_DEVICES=2
python eval.py --dataset=dtu_yao_eval --batch_size=1 --testpath=$DTU_TESTING --testlist lists/dtu/test.txt  \
               --loadckpt $CKPT_FILE --outdir "./results/DTU//MVSNet-SG" --view_thres 3 \
               --use_guided
               #--use_replace
               #--use_guided $@
