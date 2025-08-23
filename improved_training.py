import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from util.test import evaluation
from model.model import PretrainedFeatureExtractor, ED, Discriminator
from model.advanced_losses import AdvancedGANLosses, ImprovedDiscriminatorLoss, ImprovedGeneratorLoss
import logging
from argparse import ArgumentParser
from dataset.dataset import OODDataSet
from itertools import cycle
import tqdm
import matplotlib.pyplot as plt
from torchvision import models

def parse_args():
    parser = ArgumentParser(description='Improved CKAAD Training with Advanced GAN Losses')
    
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
    
    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='wasserstein', 
                       choices=['wasserstein', 'relativistic', 'feature_matching'], help='discriminator loss type')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='gradient penalty weight')
    parser.add_argument('--lambda_fm', type=float, default=1.0, help='feature matching weight')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='adversarial loss weight')
    parser.add_argument('--lambda_recon', type=float, default=10.0, help='reconstruction loss weight')
    parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='perceptual loss weight')
    parser.add_argument('--lambda_consistency', type=float, default=1.0, help='consistency loss weight')
    parser.add_argument('--lambda_diversity', type=float, default=0.1, help='diversity loss weight')
    
    # 训练参数
    parser.add_argument('--d_lr', type=float, default=1e-05, help='discriminator learning rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of discriminator updates per generator update')
    parser.add_argument('--use_perceptual', action='store_true', help='use perceptual loss')
    parser.add_argument('--use_consistency', action='store_true', help='use consistency loss')
    parser.add_argument('--use_diversity', action='store_true', help='use diversity loss')
    
    # 其他参数
    parser.add_argument('--labeled_anomaly_class_num', type=int, default=0, help='labeled anomaly class num')
    parser.add_argument('--labeled_anomaly_class', type=int, default=1, help='labeled anomaly class')
    parser.add_argument('--labeled_anomaly_ratio', type=float, default=0.0, help='labeled anomaly ratio')
    parser.add_argument('--log_dir', type=str, default='./improved_log/', help='log dir')
    parser.add_argument('--eval_epoch', type=int, default=1, help='every eval_epoch to eval')
    parser.add_argument('--topk', type=int, default=100, help='calculate topk values')
    
    return parser.parse_args()

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

def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')

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
    
    print_args(logger, args)
    
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
    
    # 损失函数
    d_loss_fn = ImprovedDiscriminatorLoss(
        loss_type=args.loss_type, 
        lambda_gp=args.lambda_gp, 
        lambda_fm=args.lambda_fm
    )
    
    g_loss_fn = ImprovedGeneratorLoss(
        lambda_adv=args.lambda_adv,
        lambda_recon=args.lambda_recon,
        lambda_perceptual=args.lambda_perceptual,
        lambda_consistency=args.lambda_consistency,
        lambda_diversity=args.lambda_diversity
    )
    
    # VGG模型（用于感知损失）
    vgg_model = None
    if args.use_perceptual:
        vgg_model = models.vgg16(pretrained=True).features[:16].to(device)
        for param in vgg_model.parameters():
            param.requires_grad_(False)
        vgg_model.eval()
    
    # 损失记录
    losses_dict = {
        'discriminator_loss': [],
        'generator_loss': [],
        'reconstruction_loss': [],
        'adversarial_loss': [],
        'perceptual_loss': [],
        'consistency_loss': [],
        'diversity_loss': []
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
            
            # 特征提取
            normal_inputs = pfe(normal_img)
            normal_outputs, recon_img = ae(normal_inputs)
            
            if anomaly_size > 0:
                anomaly_inputs = pfe(anomaly_img)
                anomaly_outputs, _ = ae(anomaly_inputs)
                outputs = [torch.cat([n_o, a_o]) for n_o, a_o in zip(normal_outputs, anomaly_outputs)]
            else:
                outputs = normal_outputs
            
            # 判别器训练
            for _ in range(args.n_critic):
                discriminator_optimizer.zero_grad()
                
                normal_inputs_detach = [i.detach() for i in normal_inputs]
                outputs_detach = [o.detach() for o in outputs]
                
                if anomaly_size > 0:
                    anomaly_inputs_detach = [i.detach() for i in anomaly_inputs]
                    d_loss = d_loss_fn(discriminator, normal_inputs_detach, outputs_detach, 
                                     normal_inputs_detach, outputs_detach)
                else:
                    d_loss = d_loss_fn(discriminator, normal_inputs_detach, outputs_detach)
                
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                discriminator_optimizer.step()
            
            # 生成器训练
            ae_optimizer.zero_grad()
            
            # 计算生成器损失
            g_loss, loss_components = g_loss_fn(
                discriminator=discriminator,
                real_samples=normal_inputs,
                fake_samples=outputs,
                real_img=normal_img,
                fake_img=recon_img,
                real_features=normal_inputs,
                fake_features=normal_outputs,
                vgg_model=vgg_model if args.use_perceptual else None
            )
            
            g_loss.backward()
            ae_optimizer.step()
            
            # 记录损失
            epoch_losses['discriminator_loss'].append(d_loss.item())
            epoch_losses['generator_loss'].append(g_loss.item())
            epoch_losses['reconstruction_loss'].append(loss_components['recon_loss'])
            epoch_losses['adversarial_loss'].append(loss_components['adv_loss'])
            epoch_losses['perceptual_loss'].append(loss_components['perceptual_loss'])
            epoch_losses['consistency_loss'].append(loss_components['consistency_loss'])
            epoch_losses['diversity_loss'].append(loss_components['diversity_loss'])
        
        # 计算平均损失
        for k in losses_dict.keys():
            losses_dict[k].append(np.mean(epoch_losses[k]))
        
        # 日志输出
        logger.info("epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f}, recon_loss: {:.6f}, adv_loss: {:.6f}, "
                   "perceptual_loss: {:.6f}, consistency_loss: {:.6f}, diversity_loss: {:.6f}".format(
            epoch, args.epochs, 
            losses_dict['discriminator_loss'][-1],
            losses_dict['generator_loss'][-1],
            losses_dict['reconstruction_loss'][-1],
            losses_dict['adversarial_loss'][-1],
            losses_dict['perceptual_loss'][-1],
            losses_dict['consistency_loss'][-1],
            losses_dict['diversity_loss'][-1]
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
    save_path = os.path.join(pic_dir, 'improved_{}.jpg'.format(args.normal))
    printPicture(losses_dict, save_path=save_path)

if __name__ == '__main__':
    args = parse_args()
    args.seed = setup_seed(args.seed)
    train(args)
