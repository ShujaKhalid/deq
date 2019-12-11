#!/bin/sh

#srun --gres=gpu:4 -c 24 -l -w guppy10 bash run_wt103_deq_transformer.sh train --name 10reg --reg 0.1 &
#srun --gres=gpu:4 -c 24 -l -w guppy15 bash run_wt103_deq_transformer.sh train --name 20reg --reg 0.2 &
#srun --gres=gpu:3 -c 24 -l -w guppy14 bash run_wt103_deq_transformer.sh train --name 30reg --reg 0.3 &
#srun --gres=gpu:4 -c 24 -l -w guppy16 bash run_wt103_deq_transformer.sh train --name 40reg --reg 0.4 &
#srun --gres=gpu:4 -c 24 -l -w guppy32 bash run_wt103_deq_transformer.sh train --name 50reg --reg 0.5 &
#srun --gres=gpu:4 -c 24 -l -w guppy31 bash run_wt103_deq_transformer.sh train --name 60reg --reg 0.6 &
#srun --gres=gpu:4 -c 24 -l -w guppy35 bash run_wt103_deq_transformer.sh train --name 70reg --reg 0.7 &
#srun --gres=gpu:4 -c 24 -l -w guppy31 bash run_wt103_deq_transformer.sh train --name 80reg --reg 0.8 &
#srun --gres=gpu:4 -c 24 -l -w guppy35 bash run_wt103_deq_transformer.sh train --name 90reg --reg 0.9 &
#srun --gres=gpu:4 -c 24 -l -w guppy16 bash run_wt103_deq_transformer.sh train --name 100reg --reg 1.0 &

# Regularization for making the equilibrium point objective smaller
#srun --gres=gpu:2 -c 8 -l -w guppy10 bash run_wt103_deq_transformer.sh train --name 0sgd_0.1fpm --reg_sgd 0.0 --reg_fpm 0.1
#srun --gres=gpu:2 -c 8 -l -w guppy10 bash run_wt103_deq_transformer.sh train --name 0sgd_0.2fpm --reg_sgd 0.0 --reg_fpm 0.2
#srun --gres=gpu:2 -c 8 -l -w guppy14 bash run_wt103_deq_transformer.sh train --name 0sgd_0.3fpm --reg_sgd 0.0 --reg_fpm 0.3
#srun --gres=gpu:2 -c 8 -l -w guppy14 bash run_wt103_deq_transformer.sh train --name 0sgd_0.4fpm --reg_sgd 0.0 --reg_fpm 0.4
#srun --gres=gpu:2 -c 8 -l -w guppy15 bash run_wt103_deq_transformer.sh train --name 0sgd_0.5fpm --reg_sgd 0.0 --reg_fpm 0.5
#srun --gres=gpu:2 -c 8 -l -w guppy15 bash run_wt103_deq_transformer.sh train --name 0sgd_0.6fpm --reg_sgd 0.0 --reg_fpm 0.6
#srun --gres=gpu:2 -c 8 -l -w guppy16 bash run_wt103_deq_transformer.sh train --name 0sgd_0.7fpm --reg_sgd 0.0 --reg_fpm 0.7
#srun --gres=gpu:2 -c 8 -l -w guppy16 bash run_wt103_deq_transformer.sh train --name 0sgd_0.8fpm --reg_sgd 0.0 --reg_fpm 0.8
#srun --gres=gpu:2 -c 8 -l -w guppy15 bash run_wt103_deq_transformer.sh train --name 0sgd_0.9fpm --reg_sgd 0.0 --reg_fpm 0.9
#srun --gres=gpu:2 -c 8 -l -w guppy34 bash run_wt103_deq_transformer.sh train --name 0sgd_1.0fpm --reg_sgd 0.0 --reg_fpm 1.0
#srun --gres=gpu:2 -c 8 -l -w guppy34 bash run_wt103_deq_transformer.sh train --name 0sgd_0.0fpm --reg_sgd 0.0 --reg_fpm 0.0

# Regularization term to the training (perplexity) objective
#srun --gres=gpu:4 -c 24 -l -w guppy35 bash run_wt103_deq_transformer.sh train --name 0.0sgd_0.0fpm --reg_sgd 0.0 --reg_fpm 0.0
#srun --gres=gpu:4 -c 24 -l -w guppy14 bash run_wt103_deq_transformer.sh train --name 0.2sgd_0.0fpm --reg_sgd 0.2 --reg_fpm 0.0
#srun --gres=gpu:4 -c 24 -l -w guppy15 bash run_wt103_deq_transformer.sh train --name 0.4sgd_0.0fpm --reg_sgd 0.4 --reg_fpm 0.0
srun --gres=gpu:4 -c 24 -l -w guppy16 bash run_wt103_deq_transformer.sh train --name 0.6sgd_0.0fpm --reg_sgd 0.6 --reg_fpm 0.0
#srun --gres=gpu:4 -c 24 -l -w guppy10 bash run_wt103_deq_transformer.sh train --name 0.8sgd_0.0fpm --reg_sgd 0.8 --reg_fpm 0.0
#srun --gres=gpu:4 -c 24 -l -w guppy34 bash run_wt103_deq_transformer.sh train --name 1.0sgd_0.0fpm --reg_sgd 1.0 --reg_fpm 0.0

# Neumann solver implementation
#srun --gres=gpu:4 -c 24 -l -w guppy19 bash run_wt103_deq_transformer.sh train --name 0.0sgd_0.0fpm_baseline --reg_sgd 0.0 --reg_fpm 0.0
