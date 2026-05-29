#!/bin/sh

dataset='region_v6'
ind_dataset='region_6_ind'
kge='ComplEx'
gpu=0
model_name="${dataset}_${kge}_smrl_finetune"

python main.py \
  --data_name ${dataset} \
  --ind_data_name ${ind_dataset} \
  --name ${model_name} \
  --metatrain_state ./state/${model_name}/${model_name}.best \
  --step fine_tune \
  --kge ${kge} \
  --gpu cuda:${gpu} \
  --posttrain_num_epoch 0 \
  --use_attr true