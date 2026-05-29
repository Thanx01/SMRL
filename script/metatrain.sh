#!/bin/sh

dataset='region_v6'
ind_dataset='region_6_ind'
kge='ComplEx'
gpu=0

python main.py \
  --data_name ${dataset} \
  --ind_data_name ${ind_dataset} \
  --name ${dataset}_${kge}_smrl \
  --step meta_train \
  --kge ${kge} \
  --gpu cuda:${gpu} \
  --use_attr true