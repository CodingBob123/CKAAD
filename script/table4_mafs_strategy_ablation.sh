#!/bin/bash
set -e
source "$(dirname "$0")/sarrnet_experiment_common.sh"

DATASET=${DATASET:-mvtec}
CATEGORIES=${CATEGORIES:-"bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"}

for normal in ${CATEGORIES}
do
    lr=$(sarrnet_category_lr "${normal}")
    extra_args=$(sarrnet_category_extra_args "${normal}")

    declare -A variants
    variants[all_channels]="--eval_anomaly_map_source rrs_cos --use_rrs --rrs_modes max mean --rrs_anomaly_samples 20 --rrs_lr 1e-3 --rrs_loss_weight 1.0 --use_afs --afs_select_planes 256 512 1024 --afs_init_bsn 50 --afs_selection_strategy fixed"
    variants[random]="--eval_anomaly_map_source rrs_cos --use_rrs --rrs_modes max mean --rrs_anomaly_samples 20 --rrs_lr 1e-3 --rrs_loss_weight 1.0 --use_afs --afs_select_planes 128 256 512 --afs_init_bsn 50 --afs_selection_strategy random"
    variants[fixed]="--eval_anomaly_map_source rrs_cos --use_rrs --rrs_modes max mean --rrs_anomaly_samples 20 --rrs_lr 1e-3 --rrs_loss_weight 1.0 --use_afs --afs_select_planes 128 256 512 --afs_init_bsn 50 --afs_selection_strategy fixed"
    variants[mafs]="--eval_anomaly_map_source rrs_cos --use_rrs --rrs_modes max mean --rrs_anomaly_samples 20 --rrs_lr 1e-3 --rrs_loss_weight 1.0 --use_afs --afs_select_planes 128 256 512 --afs_init_bsn 50 --afs_selection_strategy anomaly"

    common_hsakg="--use_pixel_anomaly --pixel_anomaly_mode mixed --pixel_patchguard_prob 0.5 --pixel_cutpaste_prob 0.3 --pixel_cutout_prob 0.2 --pixel_anomaly_prob 0.20 --use_synthetic_anomaly --anomaly_perturbation simplenet_noise --anomaly_noise_std 0.05 --anomaly_mix_noise 3 --multi_scale_anomaly --perlin_anomaly_prob 0.30 --anomaly_strategy prob"

    for variant in all_channels random fixed mafs
    do
        echo "Table4 ${variant}: ${DATASET}/${normal}"
        cmd=$(sarrnet_base_cmd "${DATASET}" "${normal}" "table4_mafs_${variant}" "${lr}")
        CUDA_VISIBLE_DEVICES=${SARRNET_GPU} ${cmd} ${variants[${variant}]} ${common_hsakg} ${extra_args}
    done
done
