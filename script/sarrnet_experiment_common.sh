#!/bin/bash

set -e

SARRNET_GPU=${SARRNET_GPU:-0}
SARRNET_SEED=${SARRNET_SEED:-111}
SARRNET_EPOCHS=${SARRNET_EPOCHS:-144}
SARRNET_EVAL_EPOCH=${SARRNET_EVAL_EPOCH:-8}
SARRNET_IMG_SIZE=${SARRNET_IMG_SIZE:-256}
SARRNET_BATCH_SIZE=${SARRNET_BATCH_SIZE:-16}
SARRNET_MODEL=${SARRNET_MODEL:-wide_resnet50_2}
SARRNET_LAYERS=${SARRNET_LAYERS:-"1 2 3"}
SARRNET_CKPT_DIR=${SARRNET_CKPT_DIR:-/hy-tmp/checkpoints/}

sarrnet_category_extra_args() {
    local category="$1"
    local extra_args=""
    case ${category} in
        cable|toothbrush|transistor|wood)
            extra_args="${extra_args} --enable_enhancement"
            ;;
    esac
    echo "${extra_args}"
}

sarrnet_category_lr() {
    local category="$1"
    if [ "${category}" = "transistor" ]; then
        echo "1e-03"
    else
        echo "5e-03"
    fi
}

sarrnet_base_cmd() {
    local dataset="$1"
    local category="$2"
    local exp_name="$3"
    local lr="$4"

    echo "python ../main.py --dataset ${dataset} --batch_size ${SARRNET_BATCH_SIZE} --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${SARRNET_EPOCHS} --normal ${category} --seed ${SARRNET_SEED} --img_size ${SARRNET_IMG_SIZE} --labeled_anomaly_class_num 1 --labeled_anomaly_class 0 --labeled_anomaly_ratio 0 --log_dir ./log --exp_name ${exp_name} --ckpt_dir ${SARRNET_CKPT_DIR} --model ${SARRNET_MODEL} --eval_epoch ${SARRNET_EVAL_EPOCH} --layer ${SARRNET_LAYERS} --recon_loss_type combined --loss_alpha 1.0 --loss_beta 0 --loss_gamma 0 --save_best"
}
