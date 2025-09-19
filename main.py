import torch
import numpy as np
import random
import os
from util.test import evaluation
from model.model import PretrainedFeatureExtractor, ED, Discriminator
import logging
from argparse import ArgumentParser
from dataset.dataset import OODDataSet
from itertools import cycle
import tqdm


def parse_args():
    
    parser = ArgumentParser(description='Pytorch implemention of Boosting Fine-Grained Visual Anomaly Detection with Coarse-Knowledge-Aware Adversarial Learning')

    parser.add_argument('--dataset', type=str, default='mvtec', help='training dataset')
    
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    
    parser.add_argument('--lr', type=float, default=0.005, help='learning tate')

    parser.add_argument('--epochs', type=int, default=200, help='training epoch')
    
    parser.add_argument('--img_size', type=int, default=32, help='img size')
    
    parser.add_argument('--normal', type=str, default='carpet', help='normal class')
    
    parser.add_argument('--seed', type=int, default=0, help='seed')
    
    parser.add_argument('--labeled_anomaly_class_num', type=int, default=0, help='labeled anomaly class num')
    
    parser.add_argument('--labeled_anomaly_class', type=int, default=1, help='labeled anomaly class')
    
    parser.add_argument('--labeled_anomaly_ratio', type=float, default=0.0, help='labeled anomaly ratio')
    
    parser.add_argument('--log_dir', type=str, default='./log/', help='log dir')
    
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'wide_resnet101_2', 'resnet152'])
    
    parser.add_argument('--eval_epoch', type=int, default=1, help='every eval_epoch to eval')
    
    parser.add_argument('--layer', nargs='+', type=int, default=[2], help='choose pretrain resnet layer to reconstruct')
    
    parser.add_argument('--d_lr', type=float, default=1e-05, help='discriminator learning rate')
    
    parser.add_argument('--adv_conf', type=float, default=0.02, help='adversial loss conf')
    
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

def loss_function(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(b[item].shape[0], -1),
                                      b[item].view(b[item].shape[0], -1)))
    return loss

def train(args):
    """
    CKAAD:整个模型的训练过程
    
    训练流程概述:
    1. 准备数据：正常样本和异常样本
    2. 特征提取：使用预训练模型提取多层级特征
    3. 自编码器训练：重建正常样本特征
    4. 判别器训练：区分正常特征、异常特征和重建特征
    5. 对抗训练：使重建特征更接近正常特征
    """
    # 1.设置日志的目录和文件名，并打印日志
    log_dir = os.path.join(args.log_dir, "lan{:.2f}_acn{}".format(args.labeled_anomaly_ratio,  args.labeled_anomaly_class_num), args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger_filename = os.path.join(log_dir, 'n_{}_a_{}_s_{}'.format(args.normal, args.labeled_anomaly_class, args.seed) + '.txt')
    logger = get_logger(logger_filename)
    logger.info("log file: {}".format(logger_filename))
    logger.info("class: {}".format(args.normal))
    
    print_args(logger, args)
    epochs = args.epochs
    batch_size = args.batch_size
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("device: {}".format(device))

    # 2.加载数据集，获取数据加载器
    dataset = OODDataSet(root='./data', dataset=args.dataset, image_size=args.img_size, category=args.normal,
                         labeled_anomaly_ratio=args.labeled_anomaly_ratio,
                         labeled_anomaly_class_num=args.labeled_anomaly_class_num,
                         labeled_anomaly_class=args.labeled_anomaly_class)
    train_dataloader, valid_dataloader, anomaly_dataloader, test_dataloader = dataset.get_data_loader(batch_size=batch_size)

    # 3.初始化模型
    # 3.1 初始化预训练特征提取器，冻结参数
    # 根据args.layer参数选择提取哪几层特征，例如[1,2,3]表示提取ResNet的第1、2、3层特征
    pfe = PretrainedFeatureExtractor(args.model, layers=args.layer, image_size=args.img_size).to(device)
    for param in pfe.parameters():
        param.requires_grad_(False)  # 冻结特征提取器参数
    pfe.eval()  # 设置为评估模式
    
    # 3.2 初始化编码器-解码器(自编码器)
    # 输入通道数由预训练模型的输出通道数决定，例如对于ResNet50和layers=[1,2,3]，为[256,512,1024]
    ae = ED(backbone=args.model, input_channels=pfe.output_channels).to(device)
    
    # 3.3 初始化判别器
    # input_sizes: 各层特征图的空间尺寸，例如[64,32,16]
    # input_channels: 各层特征图的通道数，例如[64,128,256]
    # expansion: 通道扩展系数，ResNet18/34为1，ResNet50/101等为4
    discriminator = Discriminator(input_sizes=pfe.output_sizes, input_channels=pfe.output_channels, expansion=pfe.expansion).to(device)
    
    # 3.4 初始化优化器
    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    
    # 设置权重系数和标签
    gamma = 0.5  # 控制重建特征损失的权重
    true_label = 0  # 正常样本的标签
    fake_label = 1  # 异常样本的标签
    
    # 4.开始训练循环
    for epoch in range(1, epochs+1):
        ae.train()
        discriminator.train()
        dis_loss_list = []
        recon_loss_list = []
        adv_loss_list = []
        ae_loss_list = []
        
        # 使用zip和cycle将正常数据和异常数据配对
        # cycle确保异常数据可以循环使用，即使异常数据少于正常数据
        for normal, anomaly in tqdm.tqdm(zip(train_dataloader, cycle(anomaly_dataloader))):
            # 5.1 准备输入数据
            normal_img = normal[0].to(device)  # 正常图像: [batch_size, 3, img_size, img_size]
            
            # 处理异常图像，根据不同情况获取异常样本
            if anomaly is not None:
                anomaly_img = anomaly[0].to(device)  # 异常图像: [batch_size, 3, img_size, img_size]
            elif args.dataset in ['mvtec', 'visa', 'btad'] and len(normal) == 4:
                anomaly_img = normal[1].to(device)  # 某些数据集中，normal包含正常和异常样本
            else:
                anomaly_img = normal_img[:0]  # 创建空张量，表示没有异常样本
                
            anomaly_size = anomaly_img.size(0)  # 异常样本的数量
            
            # 5.2 特征提取和重建
            # 使用预训练特征提取器提取正常样本的多层级特征
            normal_inputs = pfe(normal_img)  # 列表，包含多个特征图: [
                                            #   [batch_size, 64*exp, H1, W1], 
                                            #   [batch_size, 128*exp, H2, W2], 
                                            #   [batch_size, 256*exp, H3, W3]
                                            # ]
                                            # 其中exp是扩展系数，H1>H2>H3, W1>W2>W3
                                            # 例如对于img_size=256，可能为[64,32,16]
            
            # 使用自编码器重建正常样本特征
            normal_outputs = ae(normal_inputs)  # 列表，包含多个重建特征图，形状与normal_inputs相同
            
            # 如果有异常样本，则提取和重建异常样本特征
            if anomaly_size > 0: 
                anomaly_inputs = pfe(anomaly_img)  # 列表，形状与normal_inputs相同
                anomaly_outputs = ae(anomaly_inputs)  # 列表，形状与normal_outputs相同
                
                # 将正常样本重建特征和异常样本重建特征在批次维度上拼接
                # 对每个层级的特征分别拼接
                outputs = [torch.cat([n_o, a_o]) for n_o, a_o in zip(normal_outputs, anomaly_outputs)]  
                # outputs是列表，每个元素形状为: [batch_size*2, C, H, W]
            else:
                outputs = normal_outputs  # 如果没有异常样本，直接使用正常样本重建特征
                
            # 5.3 初始化损失值
            dis_loss = torch.tensor(0.0).to(device)
            adv_loss = torch.tensor(0.0).to(device)
            
            # 5.4 分离特征图的梯度，准备训练判别器
            # 分离正常样本特征的梯度
            normal_inputs_detach = [i.detach() for i in normal_inputs]  # 形状与normal_inputs相同，但不计算梯度
            
            # 如果有异常样本，分离异常样本特征的梯度
            if anomaly_size > 0:
                anomaly_inputs_detach = [i.detach() for i in anomaly_inputs]  # 形状与anomaly_inputs相同，但不计算梯度
                
            # 分离重建特征的梯度
            outputs_detach = [o.detach() for o in outputs]  # 形状与outputs相同，但不计算梯度
            
            # 5.5 训练判别器
            if anomaly_size > 0:
                # 判别器损失由三部分组成:
                # 1. 正常样本特征应被判为真(标签0)
                # 2. 异常样本特征应被判为假(标签1)，权重为(1-gamma)
                # 3. 重建特征应被判为假(标签1)，权重为gamma
                dis_loss = discriminator.calculate_loss(normal_inputs_detach, true_label) + \
                          (1 - gamma) * discriminator.calculate_loss(anomaly_inputs_detach, fake_label) + \
                          gamma * discriminator.calculate_loss(outputs_detach, fake_label)
                
                # 更新判别器参数
                discriminator_optimizer.zero_grad()
                dis_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
                discriminator_optimizer.step()
                    
                # 5.6 计算对抗损失
                # 希望重建特征能够欺骗判别器，被判为真(标签0)
                adv_loss = discriminator.calculate_loss(outputs, true_label)
            
            # 5.7 计算重建损失和自编码器总损失
            # 重建损失使用余弦相似度，衡量正常样本特征和重建特征的相似程度
            recon_loss = loss_function(normal_inputs, normal_outputs)
            
            # 自编码器总损失 = 重建损失 + 对抗损失*权重
            ae_loss = recon_loss + args.adv_conf * adv_loss
            
            # 更新自编码器参数
            ae_optimizer.zero_grad()
            ae_loss.backward()
            ae_optimizer.step()

            # 5.8 记录各项损失值
            dis_loss_list.append(dis_loss.item())
            ae_loss_list.append(ae_loss.item())
            recon_loss_list.append(recon_loss.item())
            adv_loss_list.append(adv_loss.item())

        # 6. 打印当前epoch的训练损失
        logger.info("epoch [{}/{}], dis_loss: {:.6f}, recon_loss:{:.6f}, adv_loss:{:.6f}, ae_loss: {:.6f}".format(epoch, epochs, np.mean(dis_loss_list),
                                                                                                                                 np.mean(recon_loss_list), np.mean(adv_loss_list), np.mean(ae_loss_list),
                                                                                                                                 ))
        # 7. 定期评估模型性能
        if (epoch) % args.eval_epoch == 0:
            if valid_dataloader is not None:
                valid_metrics = evaluation(pfe, ae, valid_dataloader, device, args)
                valid_info = get_res_str(valid_metrics)
                logger.info("Valid: {}".format(valid_info))
                
            metrics = evaluation(pfe, ae, test_dataloader, device, args)
            infostr = get_res_str(metrics)
            logger.info("Test: {}".format(infostr))
       
def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')


if __name__ == '__main__':

    args = parse_args()
    args.seed = setup_seed(args.seed)
    train(args)
   
