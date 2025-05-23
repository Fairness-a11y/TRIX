model_dir=../model-cifar10/trades
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_dir=$model_dir \
    --dataset cifar10 \
    --model resnet \
    --seed 3 \
    --overwrite \
    --rob_fairness_algorithm none \
    --loss=trades
