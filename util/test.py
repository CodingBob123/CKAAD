import torch
import numpy as np
from dataset.mvtec import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve
import cv2
from sklearn.metrics import auc
from skimage import measure
from numpy import ndarray
from scipy.ndimage import gaussian_filter
from statistics import mean
import os
from torchvision import transforms
from torchvision.utils import save_image


def ensure_anomaly_map_shape(anomaly_map):
    """Normalize anomaly maps to [B, H, W] numpy arrays."""
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.detach().cpu().numpy()
    anomaly_map = np.asarray(anomaly_map)
    if anomaly_map.ndim == 2:
        anomaly_map = anomaly_map[None, ...]
    elif anomaly_map.ndim == 4 and anomaly_map.shape[1] == 1:
        anomaly_map = anomaly_map[:, 0, :, :]
    elif anomaly_map.ndim == 4 and anomaly_map.shape[-1] == 1:
        anomaly_map = anomaly_map[..., 0]
    if anomaly_map.ndim != 3:
        raise ValueError(
            "anomaly_map must be convertible to [B, H, W], got shape {}".format(
                anomaly_map.shape
            )
        )
    return anomaly_map.astype(np.float32, copy=False)


def transform_invert(img_, transform_train):
    """
    reverse transfrom 
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None]) 
    return img_


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    batch = 1 if len(fs_list[0].size()) == 2 else fs_list[0].size(0)
    if amap_mode == 'mul':
        anomaly_map = torch.ones([batch, 1, out_size, out_size], device=fs_list[0].device)
    else:
        anomaly_map = torch.zeros([batch, 1, out_size, out_size], device=fs_list[0].device)
    a_map_list = []
    for i in range(len(ft_list)):
        
        fs = fs_list[i]
        ft = ft_list[i]
        
        a_map = 1 - F.cosine_similarity(fs, ft) # a_map: batch * H * W
        a_map = torch.unsqueeze(a_map, dim=1) # a_map: batch(1)  * 1  * H * W
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        elif amap_mode == 'max':
            anomaly_map = torch.max(anomaly_map, a_map)
        else:
            anomaly_map += a_map
    anomaly_map = anomaly_map.cpu().numpy()
    anomaly_map_list = []
    for i in range(len(anomaly_map)):
        amap = gaussian_filter(anomaly_map[i], sigma=4)
        anomaly_map_list.append(amap)
    anomaly_map = np.vstack(anomaly_map_list)
    return anomaly_map


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def calculate_metrics(scores, labels, acc=True):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)

    if len(np.unique(labels)) < 2:
        auroc_score = np.nan
        positive = int(labels[0]) if labels.size > 0 else 0
        binary_predictions = np.ones_like(labels) * positive
        f1 = 1.0 if positive == 1 else 0.0
    else:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-16)
        if len(thresholds) > 0:
            best_idx = np.argmax(f1_scores[:-1])
            best_threshold = thresholds[best_idx]
            binary_predictions = np.where(scores >= best_threshold, 1, 0)
        else:
            binary_predictions = np.zeros_like(labels)
        f1 = np.max(f1_scores)
        auroc_score = roc_auc_score(labels, scores)

    TP = np.sum((binary_predictions == 1) & (labels == 1))
    TN = np.sum((binary_predictions == 0) & (labels == 0))
    FP = np.sum((binary_predictions == 1) & (labels == 0))
    FN = np.sum((binary_predictions == 0) & (labels == 1))
    ACC = (TP + TN) / (TP + TN + FP + FN + 1e-16)
    if acc:
        res = {
            'AUROC': auroc_score,
            'F1': f1,
            'ACC': ACC,
        }
    else:
        res = {
            'AUROC': auroc_score,
        }
    return res

def get_eval_anomaly_map(inputs, outputs, img, args, rrs=None):
    source = getattr(args, 'eval_anomaly_map_source', 'recon')
    if source == 'rrs':
        if rrs is None:
            raise ValueError("eval_anomaly_map_source='rrs' requires --use_rrs")
        rrs.eval()
        rrs_out = rrs(inputs, outputs, image=img)
        return ensure_anomaly_map_shape(rrs_out['anomaly_score'])
    if source == 'rrs_cos':
        if rrs is None:
            raise ValueError("eval_anomaly_map_source='rrs_cos' requires --use_rrs")
        rrs.eval()
        anomaly_map = rrs.rrs_cosine_map(inputs, outputs, img.shape[-1], amap_mode='add')
        anomaly_map = ensure_anomaly_map_shape(anomaly_map)
        anomaly_map_list = []
        for i in range(len(anomaly_map)):
            amap = gaussian_filter(anomaly_map[i], sigma=4)
            anomaly_map_list.append(amap)
        return np.stack(anomaly_map_list, axis=0)
    return ensure_anomaly_map_shape(
        cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='add')
    )


def evaluation(encoder, ed, dataloader, device, args, afs=None, rrs=None):
    if args.dataset in ['mvtec', 'visa', 'btad']:
        return evaluation_pixel(encoder, ed, dataloader, device, args, afs=afs, rrs=rrs)
    else:
        return evaluation_semantic(encoder, ed, dataloader, device, args, afs=afs, rrs=rrs)

def evaluation_semantic(encoder, ed, dataloader, device, args, afs=None, rrs=None):
    encoder.eval()
    ed.eval()
    if rrs is not None:
        rrs.eval()
    gt_list = []
    sample_score_list = []
    metric_dict = {}
    
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            inputs_raw = encoder(img)
            if afs is not None:
                inputs = afs(inputs_raw)
            else:
                inputs = inputs_raw
            outputs = ed(inputs)
            gt_list.append(label != int(args.normal))
            anomaly_map = get_eval_anomaly_map(inputs, outputs, img, args, rrs=rrs)
            if args.dataset in ['isic']:
                fea_score = anomaly_map.reshape(img.size(0), -1).mean(axis=-1)
            else:
                fea_score = torch.topk(torch.from_numpy(anomaly_map.reshape(img.size(0), -1)), args.topk, dim=-1)[0].numpy().mean(axis=-1)
            sample_score_list.append(fea_score)
        gt_list = torch.cat(gt_list).cpu().numpy()
        sample_score_list = np.concatenate(sample_score_list)
        metric_dict['Image'] = calculate_metrics(sample_score_list, gt_list)
    return metric_dict

def evaluation_pixel(encoder, ed, dataloader, device, args, afs=None, rrs=None):
    encoder.eval()
    ed.eval()
    if rrs is not None:
        rrs.eval()
    pixel_gt_list = []
    pixel_score_list = []
    sample_gt_list = []
    sample_score_list = []
    aupro_list = []
    metrics = {}
    with torch.no_grad():
        for batch in dataloader:
            img, gt, label = batch[:3]
            img = img.to(device)
            inputs_raw = encoder(img)
            if afs is not None:
                inputs = afs(inputs_raw)
            else:
                inputs = inputs_raw
            outputs = ed(inputs)
            gt = gt.squeeze(1)
            anomaly_map = get_eval_anomaly_map(inputs, outputs, img, args, rrs=rrs)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            gt_np = gt.cpu().numpy().astype(int)

            pixel_gt_list.append(gt_np.reshape(-1))
            pixel_score_list.append(anomaly_map.reshape(-1))
            sample_gt_list.append(np.max(gt_np.reshape(gt.size(0), -1), axis=-1))
            sample_score = torch.topk(
                torch.from_numpy(anomaly_map.reshape(img.size(0), -1)),
                args.topk,
                dim=-1,
            )[0].numpy().mean(axis=-1)
            sample_score_list.append(sample_score)

            positive_mask = gt_np.reshape(gt_np.shape[0], -1).max(axis=-1).astype(bool)
            if positive_mask.any():
                for am, g in zip(anomaly_map[positive_mask], gt_np[positive_mask]):
                    aupro_list.append(
                        compute_pro(
                            g.reshape(1, *g.shape).astype(int),
                            am.reshape(1, *am.shape),
                        )
                    )

        pixel_gt_list = np.concatenate(pixel_gt_list).reshape(-1)
        pixel_score_list = np.concatenate(pixel_score_list).reshape(-1)
        sample_gt_list = np.concatenate(sample_gt_list)
        sample_score_list = np.concatenate(sample_score_list)
        pixel_aupro = round(float(np.mean(aupro_list)), 6) if aupro_list else 0.0
        metrics['Pixel'] = calculate_metrics(pixel_score_list, pixel_gt_list, False)
        metrics['Pixel']['PRO'] = pixel_aupro
        metrics['Image'] = calculate_metrics(sample_score_list, sample_gt_list, True)
    return metrics

def visualize(pfe, ae, dataloader: MVTecDataset, args, transform, device, postfix="", afs=None, rrs=None):
    pfe.eval()
    ae.eval()
    with torch.no_grad():
        cnt = 0
        for data in dataloader:
            imgs = data[0].to(device)
            inputs_raw = pfe(imgs)
            if afs is not None:
                inputs = afs(inputs_raw)
            else:
                inputs = inputs_raw
            outputs = ae(inputs)
            labels = data[-1]
            anomaly_maps = get_eval_anomaly_map(inputs, outputs, imgs, args, rrs=rrs)
            
            imgs = transform_invert(imgs, transform)
            
            if len(data) == 3:
                gts = data[1].squeeze(1)
                pack = zip(imgs, anomaly_maps, gts, labels)
            else:
                pack = zip(imgs, anomaly_maps, labels)
            for p in pack:
                ano_map = cvt2heatmap((p[1] / 2) * 255)
                img = cv2.cvtColor((p[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                result_path = './results/' + args.dataset +'_' + args.normal + postfix
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                if len(data) == 3:
                    gt = cv2.cvtColor((p[2].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    res = np.vstack((img, gt, ano_map))
                else:
                    res = np.vstack((img, ano_map))
                cv2.imwrite(result_path + '/' + str(cnt) + '_' + str(p[-1].item()) + '.png', res)
                cnt += 1


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(np.unique(masks)).issubset({0, 1}), "masks must be binary with values in {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    if delta == 0:
        return 0.0

    pros_list = []
    fprs_list = []

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)
        if len(pros) == 0:
            continue

        inverse_masks = 1 - masks
        inverse_pixels = inverse_masks.sum()
        if inverse_pixels == 0:
            continue
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_pixels

        pros_list.append(mean(pros))
        fprs_list.append(fpr)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    pros_arr = np.asarray(pros_list)
    fprs_arr = np.asarray(fprs_list)
    keep = fprs_arr < 0.3
    pros_arr = pros_arr[keep]
    fprs_arr = fprs_arr[keep]
    if len(fprs_arr) < 2 or fprs_arr.max() == 0:
        return 0.0
    fprs_arr = fprs_arr / fprs_arr.max()

    pro_auc = auc(fprs_arr, pros_arr)
    return pro_auc
