model_dir=../model-cifar10/trix/checkpoint-epoch150.pt
CUDA_VISIBLE_DEVICES=0 python evaluation.py \
    --model_dir=$model_dir \
    --dataset cifar10 \
    --model resnet
