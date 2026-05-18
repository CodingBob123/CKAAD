#!/bin/bash
set -e

EXP_NAME=${EXP_NAME:-sarrnet_main}
GPU=${GPU:-0}
SEED=${SEED:-111}
EPOCHS=${EPOCHS:-200}
EVAL_EPOCH=${EVAL_EPOCH:-8}
IMG_SIZE=${IMG_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-16}

for normal in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
do
    echo "Running MVTec category: ${normal}"

    lr=5e-03
    extra_args=""
    if [ "${normal}" = "transistor" ]; then
        lr=1e-03
    fi

    case ${normal} in
        cable|toothbrush|transistor|wood)
            extra_args="${extra_args} --enable_enhancement"
            ;;
    esac

    CUDA_VISIBLE_DEVICES=${GPU} python ../main.py \
        --dataset mvtec \
        --batch_size ${BATCH_SIZE} \
        --lr ${lr} \
        --d_lr 1e-04 \
        --adv_conf 0.02 \
        --epochs ${EPOCHS} \
        --normal ${normal} \
        --seed ${SEED} \
        --img_size ${IMG_SIZE} \
        --labeled_anomaly_class_num 1 \
        --labeled_anomaly_class 0 \
        --labeled_anomaly_ratio 0 \
        --log_dir ./log \
        --exp_name ${EXP_NAME} \
        --model wide_resnet50_2 \
        --eval_epoch ${EVAL_EPOCH} \
        --layer 1 2 3 \
        --eval_anomaly_map_source rrs_cos \
        --recon_loss_type combined \
        --loss_alpha 1.0 \
        --loss_beta 0 \
        --loss_gamma 0 \
        --use_rrs \
        --rrs_modes max mean \
        --rrs_anomaly_samples 20 \
        --rrs_lr 1e-3 \
        --rrs_loss_weight 1.0 \
        --use_afs \
        --afs_init_bsn 50 \
        --use_pixel_anomaly \
        --pixel_anomaly_mode mixed \
        --pixel_patchguard_prob 0.5 \
        --pixel_cutpaste_prob 0.3 \
        --pixel_cutout_prob 0.2 \
        --pixel_anomaly_prob 0.20 \
        --use_synthetic_anomaly \
        --anomaly_perturbation simplenet_noise \
        --anomaly_noise_std 0.05 \
        --anomaly_mix_noise 3 \
        --multi_scale_anomaly \
        --perlin_anomaly_prob 0.30 \
        --anomaly_strategy prob \
        --save_best \
        ${extra_args}
done
