import torch
import torch.nn.functional as F  # 用于SSIM损失中的插值操作
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
    """
    原始重建损失函数：使用余弦相似度
    
    参数:
        a: 原始特征列表（正常样本的多层级特征）
        b: 重建特征列表（自编码器输出的多层级特征）
    
    返回:
        loss: 余弦相似度损失值
    """
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(b[item].shape[0], -1),
                                      b[item].view(b[item].shape[0], -1)))
    return loss


def ssim_loss(pred, target, window_size=11, size_average=True):
    """
    SSIM (Structural Similarity Index) 损失函数
    参考论文: "Image Quality Assessment: From Error Visibility to Structural Similarity" (Wang et al., TIP 2004)
    
    SSIM用于衡量两张图像的结构相似性，相比像素级损失（如L1、L2），SSIM更关注：
    - 亮度相似性 (luminance)
    - 对比度相似性 (contrast)  
    - 结构相似性 (structure) <- 这是关键！对局部结构模式（边缘、纹理）更敏感
    
    ========== 如何在现有模型中建立和使用SSIM损失 ==========
    
    1. 【建立SSIM损失函数】
       本函数已经实现，包含：
       - 高斯窗口生成
       - 亮度、对比度、结构相似性的计算
       - 最终的SSIM值和损失转换
    
    2. 【应用到特征级损失】
       由于当前模型是基于特征图（而非原始图像）进行训练，有两种使用方式：
       
       方式A: 在特征图上直接计算SSIM（推荐）
           - 在 loss_function() 或训练循环中，对每个层级的特征图计算SSIM
           - 优点：直接优化特征的结构相似性
           - 实现：见下面的示例代码
       
       方式B: 如果有重构图像，在图像空间计算SSIM
           - 如果decoder最终输出重构图像，可以在图像上计算SSIM
           - 优点：更直观，但需要完整的图像解码器
    
    3. 【如何加入到现有模型训练中】
       修改位置：main.py 的 train() 函数中，大约第248行附近
       
       原代码：
           recon_loss = loss_function(normal_inputs, normal_outputs)
           ae_loss = recon_loss + args.adv_conf * adv_loss
       
       修改为：
           # 原始余弦相似度损失
           recon_loss_cos = loss_function(normal_inputs, normal_outputs)
           
           # SSIM结构损失（需要先上采样特征图到相同尺寸，或直接对特征图计算）
           # 方式1: 对每个层级特征分别计算SSIM（推荐）
           ssim_loss_value = 0.0
           for feat_pred, feat_target in zip(normal_outputs, normal_inputs):
               # 如果特征图尺寸太小（如8x8），可以上采样到合适尺寸（如32x32）再计算
               if feat_pred.shape[-1] < 11:  # window_size需要小于特征图尺寸
                   feat_pred = F.interpolate(feat_pred, size=(32, 32), mode='bilinear', align_corners=False)
                   feat_target = F.interpolate(feat_target, size=(32, 32), mode='bilinear', align_corners=False)
               # 将多通道特征图转换为单通道（取平均）或保持多通道
               # 方法1: 对每个通道分别计算SSIM后平均
               ssim_channel_sum = 0
               for c in range(feat_pred.shape[1]):
                   ssim_channel_sum += ssim_loss(feat_pred[:, c:c+1], feat_target[:, c:c+1], 
                                                window_size=min(11, feat_pred.shape[-1]), size_average=True)
               ssim_loss_value += ssim_channel_sum / feat_pred.shape[1]
           
           # 组合损失（可以调整lambda_ssim权重，建议从0.1开始）
           lambda_ssim = 0.1  # SSIM损失的权重，可根据效果调整
           recon_loss = recon_loss_cos + lambda_ssim * ssim_loss_value
           ae_loss = recon_loss + args.adv_conf * adv_loss
       
       方式2（简化版，更快）：
           # 仅对最高分辨率的特征层计算SSIM（如第一层特征）
           if len(normal_outputs) > 0:
               feat_pred = normal_outputs[0]  # 假设是最高分辨率层
               feat_target = normal_inputs[0]
               # 如果尺寸合适，直接计算；否则上采样
               if feat_pred.shape[-1] >= 11:
                   ssim_loss_value = ssim_loss(feat_pred[:, 0:1], feat_target[:, 0:1])  # 仅第一个通道
               else:
                   feat_pred_up = F.interpolate(feat_pred[:, 0:1], size=(32, 32), mode='bilinear')
                   feat_target_up = F.interpolate(feat_target[:, 0:1], size=(32, 32), mode='bilinear')
                   ssim_loss_value = ssim_loss(feat_pred_up, feat_target_up)
               
               recon_loss = loss_function(normal_inputs, normal_outputs) + 0.1 * ssim_loss_value
           else:
               recon_loss = loss_function(normal_inputs, normal_outputs)
           
           ae_loss = recon_loss + args.adv_conf * adv_loss
    
    4. 【参数调整建议】
       - lambda_ssim: SSIM损失权重，建议从0.05-0.2范围尝试
       - window_size: 高斯窗口大小，默认为11，如果特征图很小可以调小（如7）
       - 如果训练不稳定，可以先用较小的lambda_ssim（如0.05），逐步增大
    
    5. 【注意事项】
       - SSIM对特征图尺寸有要求（至少 >= window_size）
       - 如果特征图尺寸太小，建议上采样后再计算
       - 多通道特征图可以逐通道计算SSIM后平均，或者取第一个通道
       - SSIM值的范围是[-1, 1]，我们计算 (1 - SSIM) / 2 作为损失值，范围是[0, 1]
    
    参数:
        pred: 预测特征图 [B, C, H, W] 或 [B, 1, H, W]
        target: 目标特征图 [B, C, H, W] 或 [B, 1, H, W]
        window_size: 高斯窗口大小，默认为11（必须是奇数）
        size_average: 是否对批次维度求平均，默认为True
    
    返回:
        loss: SSIM损失值，范围约[0, 1]，值越小表示结构越相似
    """
    def create_window(window_size, channel):
        """创建高斯窗口"""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) 
                                  for x in range(window_size)])
            return gauss/gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    # 获取通道数（如果pred是单通道[B,1,H,W]，则为1；如果是多通道[B,C,H,W]，则为C）
    channel = pred.size(1)
    
    # 创建高斯窗口（如果窗口大小变化或通道数变化，需要重新创建）
    # 这里简化处理，假设window_size和channel在训练过程中不变
    window = create_window(window_size, channel).to(pred.device)
    
    # 计算均值（使用高斯窗口进行加权平均）
    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # SSIM公式中的常数（防止分母为0）
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        # 返回平均SSIM损失：将SSIM值转换为损失（1 - ssim）/ 2，使其范围在[0, 1]
        # SSIM范围是[-1, 1]，我们将其转换为损失值[0, 1]
        return (1 - ssim_map.mean()) / 2
    else:
        # 返回每个位置的SSIM损失
        return (1 - ssim_map) / 2

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
            
            # ========== SSIM结构感知损失（可选，已实现但默认不启用）==========
            # 如需启用SSIM损失，请取消下面的注释，并根据特征图尺寸调整参数
            # SSIM损失可以增强对结构模式的感知能力，特别是边缘、纹理等局部结构
            
            # 【使用方法1：对所有层级特征计算SSIM（推荐，但计算量较大）】
            # ssim_loss_value = 0.0
            # for feat_pred, feat_target in zip(normal_outputs, normal_inputs):
            #     # 如果特征图尺寸小于11，需要上采样（window_size默认为11）
            #     if feat_pred.shape[-1] < 11:
            #         feat_pred_up = F.interpolate(feat_pred, size=(32, 32), mode='bilinear', align_corners=False)
            #         feat_target_up = F.interpolate(feat_target, size=(32, 32), mode='bilinear', align_corners=False)
            #     else:
            #         feat_pred_up = feat_pred
            #         feat_target_up = feat_target
            #     # 对每个通道分别计算SSIM后平均（或只取第一个通道：feat_pred_up[:, 0:1]）
            #     ssim_channel_sum = 0
            #     num_channels = min(feat_pred_up.shape[1], 8)  # 限制通道数，减少计算量
            #     for c in range(num_channels):
            #         ssim_channel_sum += ssim_loss(feat_pred_up[:, c:c+1], feat_target_up[:, c:c+1], 
            #                                      window_size=min(11, feat_pred_up.shape[-1]), size_average=True)
            #     ssim_loss_value += ssim_channel_sum / num_channels
            # # 组合SSIM损失（权重可调，建议从0.05-0.2范围尝试）
            # lambda_ssim = 0.1  # SSIM损失权重，可根据效果调整
            # recon_loss = recon_loss + lambda_ssim * ssim_loss_value
            
            # 【使用方法2：仅对最高分辨率特征层计算SSIM（更快，推荐先试）】
            # if len(normal_outputs) > 0:
            #     feat_pred = normal_outputs[0]  # 最高分辨率层（通常是第一层）
            #     feat_target = normal_inputs[0]
            #     # 如果特征图尺寸合适（>=11），直接计算；否则上采样
            #     if feat_pred.shape[-1] >= 11:
            #         # 仅使用第一个通道计算SSIM（加快计算）
            #         ssim_loss_value = ssim_loss(feat_pred[:, 0:1], feat_target[:, 0:1], 
            #                                     window_size=min(11, feat_pred.shape[-1]), size_average=True)
            #     else:
            #         # 上采样到合适尺寸
            #         target_size = max(32, feat_pred.shape[-1] * 2)  # 至少32x32
            #         feat_pred_up = F.interpolate(feat_pred[:, 0:1], size=(target_size, target_size), 
            #                                    mode='bilinear', align_corners=False)
            #         feat_target_up = F.interpolate(feat_target[:, 0:1], size=(target_size, target_size), 
            #                                      mode='bilinear', align_corners=False)
            #         ssim_loss_value = ssim_loss(feat_pred_up, feat_target_up, window_size=11, size_average=True)
            #     
            #     # 组合损失（权重可调）
            #     lambda_ssim = 0.1  # 建议从0.05开始，逐步调整到0.1-0.2
            #     recon_loss = recon_loss + lambda_ssim * ssim_loss_value
            
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
   
