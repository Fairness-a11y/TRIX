# TRIX

This is code implementation of TRIX

Prerequisites

    python 3.8.18
    pytorch 1.6.0
    torchvision 0.7.0
    numpy 1.24.3

### Training

```
bash scripts/trades_trix.sh     # for our method

bash scripts/trades.sh          # for baseline

Arguments:
  --model_dir                         Directory of model for saving checkpoint
  --dataset                           The dataset to use for training
  --model                             Name of the model architecture
  --loss                              Which loss to use, choices=(trades, pgd, mixed)
  --rob_fairness_algorithm            robust fairness algorithms, choices=(dafa, none)
  --dafa_warmup                       warmup epochs for dafa
  --dafa_lambda                       the value of hyperparmater lambda of dafa
```

### Evaluation
```
bash scripts/evaluation.sh

change the path of the evaluation checkpoint


