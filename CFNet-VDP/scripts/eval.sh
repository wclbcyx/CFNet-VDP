# DATAPATH="/public/home/user1/shz/data/USOD10k"
# CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dataset kitti \
#     --datapath $DATAPATH --trainlist ../filenames/uw_offset_train.txt --testlist ../filenames/uw_offset_val.txt \
#     --epochs 20 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 4 --maxdisp 256 \
#     --model cfnet --logdir ./checkpoints/uwwodis/uniform_sample_d256  --mode eval --loadckpt ./checkpoints/test-71conv-rvolume-wgausenoise/checkpoint_000019.ckpt \
#     --test_batch_size 4

DATAPATH="/mnt/sda/wuqizheng/underwater/pythonProject/SQUID"
CUDA_VISIBLE_DEVICES=1 python ./main.py --dataset SQUID \
    --datapath $DATAPATH --trainlist ./filenames/SQUID_test1.txt --testlist ./filenames/SQUID_test1.txt \
    --epochs 20 --lr 0.001 --lrepochs "12,16,18,20:2" --batch_size 4 --maxdisp 256 \
    --model cfnet --logdir ./checkpoints/uwwodis/uniform_sample_d256  --mode eval --loadckpt ./checkpoints/train_v2/checkpoint_000019.ckpt \
    --test_batch_size 1