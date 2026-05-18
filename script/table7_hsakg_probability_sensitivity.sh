#!/bin/bash
set -e
source "$(dirname "$0")/sarrnet_experiment_common.sh"

DATASET=${DATASET:-mvtec}
# CATEGORIES=${CATEGORIES:-"bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"}
CATEGORIES=${CATEGORIES:-"toothbrush transistor wood zipper"}

common_full="--eval_anomaly_map_source rrs_cos --use_rrs --rrs_modes max mean --rrs_anomaly_samples 20 --rrs_lr 1e-3 --rrs_loss_weight 1.0 --use_afs --afs_init_bsn 50 --afs_selection_strategy anomaly --use_pixel_anomaly --pixel_anomaly_mode mixed --pixel_patchguard_prob 0.5 --pixel_cutpaste_prob 0.3 --pixel_cutout_prob 0.2 --use_synthetic_anomaly --anomaly_perturbation simplenet_noise --anomaly_noise_std 0.05 --anomaly_mix_noise 3 --multi_scale_anomaly --anomaly_strategy prob"

for normal in ${CATEGORIES}
do
    lr=$(sarrnet_category_lr "${normal}")
    extra_args=$(sarrnet_category_extra_args "${normal}")

    declare -A variants
    variants[p10_p20]="--pixel_anomaly_prob 0.10 --perlin_anomaly_prob 0.20"
    variants[p20_p30]="--pixel_anomaly_prob 0.20 --perlin_anomaly_prob 0.30"
    variants[p30_p30]="--pixel_anomaly_prob 0.30 --perlin_anomaly_prob 0.30"
    variants[p40_p40]="--pixel_anomaly_prob 0.40 --perlin_anomaly_prob 0.40"

    for variant in p10_p20 p20_p30 p30_p30 p40_p40
    do
        echo "Table7 ${variant}: ${DATASET}/${normal}"
        cmd=$(sarrnet_base_cmd "${DATASET}" "${normal}" "table7_hsakg_${variant}" "${lr}")
        CUDA_VISIBLE_DEVICES=${SARRNET_GPU} ${cmd} ${variants[${variant}]} ${common_full} ${extra_args}
    done
done
