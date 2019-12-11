#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training (DEQ-Transformer)...'
    python train_transformer.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --div_val 4 \
        --n_layer 2 \
        --eval_n_layer 24 \
        --d_embed 700 \
        --d_model 700 \
        --n_head 10 \
        --d_head 70 \
        --d_inner 48000 \
        --dropout 0.05 \
        --dropatt 0.0 \
        --optim Adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --pretrain_steps 0 \
        --log_interval 200 \
        --eval_interval 2000 \
        --max_step 300000 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 150 \
        --wnorm \
        --f_thres 30 \
        --b_thres 40 \
        --subseq_len 75 \
        --batch_size 56 \
        --gpu0_bsz 14 \
        --multi_gpu \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Not supported yet'
else
    echo 'unknown argment 1'
fi

# if [[ $1 == 'train' ]]; then
#     echo 'Run training (DEQ-Transformer)...'
#     python train_transformer.py \
#         --cuda \
#         --data ../data/wikitext-103/ \
#         --dataset wt103 \
#         --adaptive \
#         --div_val 4 \
#         --n_layer 2 \
#         --eval_n_layer 24 \
#         --d_embed 700 \
#         --d_model 700 \
#         --n_head 10 \
#         --d_head 70 \
#         --d_inner 48000 \
#         --dropout 0.05 \
#         --dropatt 0.0 \
#         --optim Adam \
#         --lr 0.00025 \
#         --warmup_step 0 \
#         --pretrain_steps 0 \
#         --log_interval 1 \
#         --eval_interval 2 \
#         --max_step 5 \
#         --tgt_len 150 \
#         --mem_len 150 \
#         --eval_tgt_len 150 \
#         --wnorm \
#         --f_thres 1000 \
#         --b_thres 40 \
#         --subseq_len 75 \
#         --batch_size 56 \
#         --gpu0_bsz 14 \
#         --multi_gpu \
#         ${@:2}
# elif [[ $1 == 'eval' ]]; then
#     echo 'Not supported yet'
# else
#     echo 'unknown argment 1'
# fi