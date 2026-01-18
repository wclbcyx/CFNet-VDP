#!/usr/bin/env bash
set -x
DATAPATH="/mnt/sda/wuqizheng/underwater/pythonProject/UDOS10k"
CUDA_VISIBLE_DEVICES=0 nohup python ./main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/uw_offset_train.txt --testlist ./filenames/uw_offset_val.txt \
    --epochs 20 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 4 --maxdisp 256 \
    --model cfnet --logdir ./checkpoints/train_v2 \
    --test_batch_size 1 \
    # --loadckpt ./checkpoints/train_new6/checkpoint_000019.ckpt \