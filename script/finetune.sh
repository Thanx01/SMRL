#!/bin/sh

dataset='region_v6'
ind_dataset='region_6_ind'
kge='ComplEx'
gpu=0
pretrain_name="${dataset}_${kge}_smrl"

python main.py \
  --data_name ${dataset} \
  --ind_data_name ${ind_dataset} \
  --name ${dataset}_${kge}_smrl_finetune \
  --metatrain_state ./state/${pretrain_name}/${pretrain_name}.best \
  --step fine_tune \
  --kge ${kge} \
  --gpu cuda:${gpu} \
  --use_attr true