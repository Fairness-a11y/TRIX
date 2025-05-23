
model_dir=../model-cifar10/trix
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_dir=$model_dir \
    --dataset cifar10 \
    --model resnet \
    --seed 3 \
    --dafa_warmup 70 \
    --dafa_lambda 1.0 \
    --overwrite \
    --rob_fairness_algorithm dafa \
    --loss=mixed
