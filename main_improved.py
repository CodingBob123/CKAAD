import torch
import numpy as np
import random
import os
from util.test import evaluation
from model.model import PretrainedFeatureExtractor, ED, Discriminator
from model.adaptive_loss import AdaptiveReconstructionLoss, MultiScaleReconstructionLoss, PerceptualReconstructionLoss
import logging
from argparse import ArgumentParser
from dataset.dataset import OODDataSet
from itertools import cycle
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models

def parse_args():
    parser = ArgumentParser(description='Improved CKAAD with Enhanced Image Reconstruction Loss')
    
    # 基础参数
    parser.add_argument('--dataset', type=str, default='mvtec', help='training dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='training epoch')
    parser.add_argument('--img_size', type=int, default=32, help='img size')
    parser.add_argument('--normal', type=str, default='carpet', help='normal class')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--model', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'wide_resnet101_2', 'resnet152'])
    parser.add_argument('--layer', nargs='+', type=int, default=[2], help='choose pretrain resnet layer to reconstruct')
    
    # 重建损失参数
    parser.add_argument('--reconstruction_type', type=str, default='adaptive', 
                       choices=['simple', 'adaptive', 'multiscale', 'perceptual'], help='reconstruction loss type')
    parser.add_argument('--alpha', type=float, default=1.0, help='feature reconstruction weight')
    parser.add_argument('--beta', type=float, default=0.5, help='pixel reconstruction weight')
    parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='perceptual loss weight')
    
    # 其他参数
    parser.add_argument('--labeled_anomaly_class_num', type=int, default=0, help='labeled anomaly class num')
    parser.add_argument('--labeled_anomaly_class', type=int, default=1, help='labeled anomaly class')
    parser.add_argument('--labeled_anomaly_ratio', type=float, default=0.0, help='labeled anomaly ratio')
    parser.add_argument('--log_dir', type=str, default='./improved_log/', help='log dir')
    parser.add_argument('--eval_epoch', type=int, default=1, help='every eval_epoch to eval')
    parser.add_argument('--d_lr', type=float, default=1e-05, help='discriminator learning rate')
    parser.add_argument('--adv_conf', type=float, default=0.02, help='adversarial loss weight')
    parser.add_argument('--topk', type=int, default=100, help='calculate topk values')
    
    return parser.parse_args()

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def get_res_str(metrics):
    score_res_str = ""
    for key, value in metrics.items():
        for item, v in value.items():
            score_res_str += "{}_{}: {:.6f} ".format(key, item, v) 
    return score_res_str

def printPicture(losses_dict, save_path=None):
    """绘制损失曲线"""
    n_losses = len(losses_dict)
    n_cols = 3
    n_rows = (n_losses + n_cols - 1) // n_cols
    
    plt.figure(figsize=(18, 6 * n_rows))
    for i, (loss_name, loss_values) in enumerate(losses_dict.items()):
        plt.subplot(n_rows, n_cols, i+1)
        plt.plot(range(1, len(loss_values)+1), loss_values, marker='o')
        plt.title(loss_name)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def train(args):
    # 设置日志
    log_dir = os.path.join(args.log_dir, "lan{:.2f}_acn{}".format(args.labeled_anomaly_ratio, args.labeled_anomaly_class_num), args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger_filename = os.path.join(log_dir, 'improved_n_{}_a_{}_s_{}'.format(args.normal, args.labeled_anomaly_class, args.seed) + '.txt')
    logger = get_logger(logger_filename)
    logger.info("log file: {}".format(logger_filename))
    logger.info("class: {}".format(args.normal))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("device: {}".format(device))

    # 数据加载
    dataset = OODDataSet(root='./data', dataset=args.dataset, image_size=args.img_size, category=args.normal,
                         labeled_anomaly_ratio=args.labeled_anomaly_ratio,
                         labeled_anomaly_class_num=args.labeled_anomaly_class_num,
                         labeled_anomaly_class=args.labeled_anomaly_class)
    train_dataloader, valid_dataloader, anomaly_dataloader, test_dataloader = dataset.get_data_loader(batch_size=args.batch_size)

    # 模型初始化
    pfe = PretrainedFeatureExtractor(args.model, layers=args.layer, image_size=args.img_size).to(device)
    for param in pfe.parameters():
        param.requires_grad_(False)
    pfe.eval()
    
    ae = ED(backbone=args.model, input_channels=pfe.output_channels).to(device)
    discriminator = Discriminator(input_sizes=pfe.output_sizes, input_channels=pfe.output_channels, expansion=pfe.expansion).to(device)
    
    # 优化器
    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    
    # 重建损失函数
    if args.reconstruction_type == 'adaptive':
        reconstruction_loss_fn = AdaptiveReconstructionLoss(
            initial_alpha=args.alpha, 
            initial_beta=args.beta
        ).to(device)
    elif args.reconstruction_type == 'multiscale':
        reconstruction_loss_fn = MultiScaleReconstructionLoss().to(device)
    elif args.reconstruction_type == 'perceptual':
        # 加载VGG模型用于感知损失
        vgg_model = models.vgg16(pretrained=True).features[:16].to(device)
        for param in vgg_model.parameters():
            param.requires_grad_(False)
        vgg_model.eval()
        reconstruction_loss_fn = PerceptualReconstructionLoss(
            vgg_model=vgg_model, 
            lambda_perceptual=args.lambda_perceptual
        ).to(device)
    else:  # simple
        reconstruction_loss_fn = None
    
    # 训练参数
    gamma = 0.5
    true_label = 0
    fake_label = 1
    
    # 损失记录
    losses_dict = {
        'discriminator_loss': [],
        'reconstruction_loss': [],
        'feature_loss': [],
        'pixel_loss': [],
        'adversarial_loss': [],
        'total_loss': []
    }
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        ae.train()
        discriminator.train()
        
        epoch_losses = {k: [] for k in losses_dict.keys()}
        
        for normal, anomaly in tqdm.tqdm(zip(train_dataloader, cycle(anomaly_dataloader))):
            normal_img = normal[0].to(device)
            if anomaly is not None:
                anomaly_img = anomaly[0].to(device)
            elif args.dataset in ['mvtec', 'visa', 'btad'] and len(normal) == 4:
                anomaly_img = normal[1].to(device)
            else:
                anomaly_img = normal_img[:0]
            
            anomaly_size = anomaly_img.size(0)
            
            # 特征提取和重建
            normal_inputs = pfe(normal_img)
            normal_outputs, recon_img = ae(normal_inputs)
            
            if anomaly_size > 0:
                anomaly_inputs = pfe(anomaly_img)
                anomaly_outputs, _ = ae(anomaly_inputs)
                outputs = [torch.cat([n_o, a_o]) for n_o, a_o in zip(normal_outputs, anomaly_outputs)]
            else:
                outputs = normal_outputs
            
            # 判别器训练
            dis_loss = torch.tensor(0.0).to(device)
            adv_loss = torch.tensor(0.0).to(device)
            
            normal_inputs_detach = [i.detach() for i in normal_inputs]
            outputs_detach = [o.detach() for o in outputs]
            
            if anomaly_size > 0:
                anomaly_inputs_detach = [i.detach() for i in anomaly_inputs]
                dis_loss = (discriminator.calculate_loss(normal_inputs_detach, true_label) + 
                           (1 - gamma) * discriminator.calculate_loss(anomaly_inputs_detach, fake_label) + 
                           gamma * discriminator.calculate_loss(outputs_detach, fake_label))
                
                discriminator_optimizer.zero_grad()
                dis_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                discriminator_optimizer.step()
                
                adv_loss = discriminator.calculate_loss(outputs, true_label)
            
            # 重建损失计算
            if args.reconstruction_type == 'adaptive':
                recon_loss, loss_components = reconstruction_loss_fn(
                    normal_inputs, normal_outputs, recon_img, normal_img
                )
                feature_loss = loss_components['feature_loss']
                pixel_loss = loss_components['pixel_loss']
            elif args.reconstruction_type == 'multiscale':
                recon_loss = reconstruction_loss_fn(recon_img, normal_img)
                feature_loss = 0  # 多尺度损失只包含像素损失
                pixel_loss = recon_loss.item()
            elif args.reconstruction_type == 'perceptual':
                recon_loss, loss_components = reconstruction_loss_fn(recon_img, normal_img)
                feature_loss = 0  # 感知损失主要关注像素和感知层面
                pixel_loss = loss_components['pixel_loss']
            else:  # simple
                # 原始的特征重建损失
                cos_loss = torch.nn.CosineSimilarity()
                feature_loss = 0
                for item in range(len(normal_inputs)):
                    feature_loss += torch.mean(1-cos_loss(
                        normal_inputs[item].view(normal_outputs[item].shape[0], -1),
                        normal_outputs[item].view(normal_outputs[item].shape[0], -1)
                    ))
                
                # 像素重建损失
                if recon_img.shape[-2:] != normal_img.shape[-2:]:
                    recon_img = F.interpolate(recon_img, size=normal_img.shape[-2:], 
                                            mode='bilinear', align_corners=False)
                pixel_loss = F.mse_loss(recon_img, normal_img)
                
                recon_loss = args.alpha * feature_loss + args.beta * pixel_loss
            
            # 总损失
            total_loss = recon_loss + args.adv_conf * adv_loss
            
            # 生成器训练
            ae_optimizer.zero_grad()
            total_loss.backward()
            ae_optimizer.step()
            
            # 记录损失
            epoch_losses['discriminator_loss'].append(dis_loss.item())
            epoch_losses['reconstruction_loss'].append(recon_loss.item())
            epoch_losses['feature_loss'].append(feature_loss if isinstance(feature_loss, float) else feature_loss.item())
            epoch_losses['pixel_loss'].append(pixel_loss if isinstance(pixel_loss, float) else pixel_loss.item())
            epoch_losses['adversarial_loss'].append(adv_loss.item())
            epoch_losses['total_loss'].append(total_loss.item())
        
        # 计算平均损失
        for k in losses_dict.keys():
            losses_dict[k].append(np.mean(epoch_losses[k]))
        
        # 日志输出
        logger.info("epoch [{}/{}], d_loss: {:.6f}, recon_loss: {:.6f}, feature_loss: {:.6f}, "
                   "pixel_loss: {:.6f}, adv_loss: {:.6f}, total_loss: {:.6f}".format(
            epoch, args.epochs, 
            losses_dict['discriminator_loss'][-1],
            losses_dict['reconstruction_loss'][-1],
            losses_dict['feature_loss'][-1],
            losses_dict['pixel_loss'][-1],
            losses_dict['adversarial_loss'][-1],
            losses_dict['total_loss'][-1]
        ))
        
        # 评估
        if epoch % args.eval_epoch == 0:
            if valid_dataloader is not None:
                valid_metrics = evaluation(pfe, ae, valid_dataloader, device, args)
                valid_info = get_res_str(valid_metrics)
                logger.info("Valid: {}".format(valid_info))
                
            metrics = evaluation(pfe, ae, test_dataloader, device, args)
            infostr = get_res_str(metrics)
            logger.info("Test: {}".format(infostr))
    
    # 绘制损失曲线
    pic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pic')
    os.makedirs(pic_dir, exist_ok=True)
    save_path = os.path.join(pic_dir, 'improved_{}_{}.jpg'.format(args.normal, args.reconstruction_type))
    printPicture(losses_dict, save_path=save_path)

if __name__ == '__main__':
    args = parse_args()
    args.seed = setup_seed(args.seed)
    train(args)
