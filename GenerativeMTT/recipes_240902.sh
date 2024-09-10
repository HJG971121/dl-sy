#!/bin/bash

# cropsize
#python lazyconfig_train_net.py \
#--config-file ../GenerativeMTT/configs/cropsize/baseline_MADenseUNet_64.py --resume

python lazyconfig_train_net.py \
--config-file ../GenerativeMTT/configs/cropsize/baseline_MADenseUNet_128.py

python lazyconfig_train_net.py \
--config-file ../GenerativeMTT/configs/cropsize/baseline_MADenseUNet_384.py

python lazyconfig_train_net.py \
--config-file ../GenerativeMTT/configs/cropsize/baseline_MADenseUNet_512.py

# loss
python lazyconfig_train_net.py \
--config-file ../GenerativeMTT/configs/loss/baseline_MADenseUNet_MSE.py

python lazyconfig_train_net.py \
--config-file ../GenerativeMTT/configs/loss/baseline_MADenseUNet_MSE+0.1vMSE.py

python lazyconfig_train_net.py \
--config-file ../GenerativeMTT/configs/loss/baseline_MADenseUNet_MSE+10vMSE.py

python lazyconfig_train_net.py \
--config-file ../GenerativeMTT/configs/loss/baseline_MADenseUNet_MSE+vMSEa0.5+vMSEv0.5.py

python lazyconfig_train_net.py \
--config-file ../GenerativeMTT/configs/loss/baseline_MADenseUNet_MSE+vMSEa0.25+vMSEv0.75.py