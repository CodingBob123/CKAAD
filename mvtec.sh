#!/bin/bash
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1

# ========== AFS 参数说明 ==========
# AFS auto-half: 对于 wide_resnet50_2 --layer 1 2 3:
#   layer1: 256 → 128 | layer2: 512 → 256 | layer3: 1024 → 512
# 可手动指定: --afs_select_planes 128 256 512
#
# 如需启用合成异常协同训练（可选），取消下面注释：
#   --use_synthetic_anomaly --anomaly_perturbation noise \
#   --anomaly_ratio 0.3 --anomaly_noise_std 0.15 --multi_scale_anomaly

for normal in 'bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
do
    echo $normal

    case $normal in 'carpet')
            epochs=10
            eval_epoch=1
            ;;
        *)
            epochs=104
            eval_epoch=8
            ;;
    esac

    case $normal in 'transistor')
            lr=1e-03
            ;;
        *)
            lr=5e-03
            ;;
    esac

CUDA_VISIBLE_DEVICES=0 python main.py --dataset mvtec --batch_size 16 \
 --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
--normal $normal --seed 111 --img_size 256 \
--labeled_anomaly_class_num ${labeled_anomaly_class_num} \
--labeled_anomaly_class 0 \
--labeled_anomaly_ratio ${labeled_anomaly_ratio} \
--log_dir ./log --model wide_resnet50_2 --eval_epoch ${eval_epoch} --layer 1 2 3 \
--recon_loss_type combined --loss_alpha 1.0 --loss_beta 0 --loss_gamma 0 \
--use_rrs --rrs_anomaly_samples 20 --rrs_lr 1e-3 --rrs_loss_weight 1.0 \
--use_afs --afs_init_bsn 50
done
