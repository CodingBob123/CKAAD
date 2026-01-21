import torch
from contextlib import nullcontext
import numpy as np
import random
import os
from util.test import evaluation, visualize_anomaly_maps_simple
from model.model import PretrainedFeatureExtractor, ED, Discriminator
import logging
from argparse import ArgumentParser
from dataset.dataset import OODDataSet
from itertools import cycle
import tqdm
import matplotlib
matplotlib.use('Agg')  # 设置非GUI后端，避免WSL图形界面问题
import matplotlib.pyplot as plt
from torchvision import transforms


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

    parser.add_argument('--use_amp', action='store_true', help='enable mixed precision (AMP)')
    parser.add_argument('--compile', action='store_true', help='enable torch.compile for models if available')

    # 可视化相关参数
    parser.add_argument('--enable_epoch_viz', action='store_true', help='enable anomaly map visualization during training (every 8 epochs)')
    parser.add_argument('--viz_interval', type=int, default=8, help='interval for anomaly map visualization during training')

    # 新增：渐进式实验配置参数
    parser.add_argument('--enable_enhancement', nargs='*', type=lambda x: str(x).lower() in ('true', '1', 'yes', 't', 'y'),
                       default=[False, False, False],
                       help='enable enhancement for each branch [branch1, branch2, branch3]. Use --enable_enhancement True False True format')

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

def loss_draw(loss_history, save_path=None):
    """
    绘制损失曲线。
    - 横轴：epoch
    - 纵轴：不同损失值
    - 布局：2x2网格，四个子图
    - 比例尺较大：调整为较大的画布和线宽
    """
    if not loss_history:
        print("Warning: loss_history is empty, skipping plot generation")
        return

    try:
        items = list(loss_history.items())
        if len(items) != 4:
            print(f"Warning: Expected 4 loss items, got {len(items)}")
            return

        # 创建2x2子图布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 更大的画布尺寸
        axes = axes.ravel()  # 扁平化axes数组

        for idx, (name, values) in enumerate(items):
            ax = axes[idx]
            if values and len(values) > 0:
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, marker='o', linewidth=3, markersize=5, color='blue')
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel(name, fontsize=12)
                ax.set_title(f'{name} vs Epoch', fontsize=14, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.tick_params(axis='both', which='major', labelsize=10)
                # 设置更大的边距
                ax.margins(x=0.05, y=0.1)
            else:
                ax.set_title(f"{name}\n(No data)", fontsize=14)
                ax.axis('off')

        plt.tight_layout(pad=3.0)

        if save_path:
            # 确保目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            # 保存为jpg格式
            plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
            print(f"Loss curve saved to: {save_path}")
        else:
            # 在无图形界面环境中，不显示图片，直接跳过
            print("Warning: No save path provided, skipping plot display in headless environment")

        plt.close()

    except Exception as e:
        print(f"Error generating loss plot: {e}")
        plt.close()

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
    # 传递渐进式实验配置参数
    ae = ED(backbone=args.model, input_channels=pfe.output_channels, enable_enhancement=args.enable_enhancement).to(device)
    
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

    # AMP setup
    use_cuda_amp = args.use_amp and (device == 'cuda')
    if use_cuda_amp:
        from torch.cuda.amp import autocast, GradScaler
        scaler_ae = GradScaler()
        scaler_d = GradScaler()
        amp_ctx = autocast
        logger.info("AMP enabled (autocast + GradScaler)")
    else:
        scaler_ae = None
        scaler_d = None
        amp_ctx = nullcontext

    # 记录各类损失用于绘图
    loss_history = {
        "dis_loss": [],
        "recon_loss": [],
        "adv_loss": [],
        "ae_loss": [],
    }
    
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
            with amp_ctx():
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
                with amp_ctx():
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
                with amp_ctx():
                    dis_loss = discriminator.calculate_loss(normal_inputs_detach, true_label) + \
                              (1 - gamma) * discriminator.calculate_loss(anomaly_inputs_detach, fake_label) + \
                              gamma * discriminator.calculate_loss(outputs_detach, fake_label)
                discriminator_optimizer.zero_grad()
                if use_cuda_amp:
                    scaler_d.scale(dis_loss).backward()
                    # Unscale before clipping
                    scaler_d.unscale_(discriminator_optimizer)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
                    scaler_d.step(discriminator_optimizer)
                    scaler_d.update()
                else:
                    dis_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
                    discriminator_optimizer.step()
                    
                # 5.6 计算对抗损失
                # 希望重建特征能够欺骗判别器，被判为真(标签0)
                with amp_ctx():
                    adv_loss = discriminator.calculate_loss(outputs, true_label)

            # 5.7 计算重建损失和自编码器总损失
            # 重建损失使用余弦相似度，衡量正常样本特征和重建特征的相似程度
            with amp_ctx():
                recon_loss = loss_function(normal_inputs, normal_outputs)
                ae_loss = recon_loss + args.adv_conf * adv_loss
            ae_optimizer.zero_grad()
            if use_cuda_amp:
                scaler_ae.scale(ae_loss).backward()
                scaler_ae.step(ae_optimizer)
                scaler_ae.update()
            else:
                ae_loss.backward()
                ae_optimizer.step()

            # 5.8 记录各项损失值
            dis_loss_list.append(dis_loss.item())
            ae_loss_list.append(ae_loss.item())
            recon_loss_list.append(recon_loss.item())
            adv_loss_list.append(adv_loss.item())

        # 6. 打印当前epoch的训练损失并记录到历史
        epoch_dis = np.mean(dis_loss_list)
        epoch_recon = np.mean(recon_loss_list)
        epoch_adv = np.mean(adv_loss_list)
        epoch_ae = np.mean(ae_loss_list)

        logger.info("epoch [{}/{}], dis_loss: {:.6f}, recon_loss:{:.6f}, adv_loss:{:.6f}, ae_loss: {:.6f}".format(epoch, epochs, epoch_dis,
                                                                                                                                 epoch_recon, epoch_adv, epoch_ae,
                                                                                                                                 ))

        # 记录损失历史用于绘图
        loss_history["dis_loss"].append(epoch_dis)
        loss_history["recon_loss"].append(epoch_recon)
        loss_history["adv_loss"].append(epoch_adv)
        loss_history["ae_loss"].append(epoch_ae)

        # 7. 定期评估模型性能
        if (epoch) % args.eval_epoch == 0:
            if valid_dataloader is not None:
                with amp_ctx():
                    valid_metrics = evaluation(pfe, ae, valid_dataloader, device, args)
                valid_info = get_res_str(valid_metrics)
                logger.info("Valid: {}".format(valid_info))

            with amp_ctx():
                metrics = evaluation(pfe, ae, test_dataloader, device, args)
            infostr = get_res_str(metrics)
            logger.info("Test: {}".format(infostr))

            # 可选：在定期评估时绘制异常热力图
            if args.enable_epoch_viz and epoch % args.viz_interval == 0:
                try:
                    logger.info("Generating anomaly maps at epoch {}...".format(epoch))

                    # 使用简化的matplotlib可视化
                    visualize_anomaly_maps_simple(pfe, ae, test_dataloader, args, device, epoch)

                    viz_result_path = './results/{}_{}_epoch_{}'.format(args.dataset, args.normal, epoch)
                    logger.info("Anomaly maps saved to: {}".format(viz_result_path))

                except Exception as e:
                    logger.error("Failed to generate anomaly maps at epoch {}: {}".format(epoch, str(e)))

    # 8. 训练结束后保存最终损失曲线
    try:
        pic_dir = "./pic/"
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir, exist_ok=True)

        loss_img_name = "loss_curve_final_n_{}_a_{}_s_{}.jpg".format(args.normal, args.labeled_anomaly_class, args.seed)
        loss_save_path = os.path.join(pic_dir, loss_img_name)

        # 检查损失历史记录是否为空
        if any(loss_history.values()):
            loss_draw(loss_history, loss_save_path)
            logger.info("Final loss curve saved to: {}".format(loss_save_path))
        else:
            logger.warning("Loss history is empty, skipping plot generation")
    except Exception as e:
        logger.error("Failed to generate final loss curve: {}".format(str(e)))

    # 9. 训练结束后进行anomaly map可视化
    try:
        logger.info("Starting anomaly map visualization...")

        # 设置数据变换（与训练时相同）
        if args.dataset in ['mvtec', 'visa', 'btad']:
            img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            gt_transform = transforms.Compose([transforms.ToTensor()])
        else:
            img_transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            gt_transform = transforms.Compose([transforms.ToTensor()])

        # 创建测试数据集（只可视化前几个样本以节省时间）
        if args.dataset == 'mvtec':
            from dataset.mvtec import MVTecDataset
            viz_dataset = MVTecDataset(root='./data', category=args.normal, train=False,
                                     transform=img_transform, gt_target_transform=gt_transform,
                                     img_size=args.img_size)
            # 加载实际的数据到内存中
            viz_dataset.load_data()

            # 只可视化前5个样本（包括正常和异常样本）
            viz_indices = []
            normal_count = 0
            abnormal_count = 0
            for i, target in enumerate(viz_dataset.targets):
                if target == 0 and normal_count < 3:  # 正常样本
                    viz_indices.append(i)
                    normal_count += 1
                elif target > 0 and abnormal_count < 2:  # 异常样本
                    viz_indices.append(i)
                    abnormal_count += 1
                if len(viz_indices) >= 5:
                    break

            # 筛选数据
            viz_dataset.data = viz_dataset.data[viz_indices]
            viz_dataset.targets = viz_dataset.targets[viz_indices]
            viz_dataset.gt_targets = viz_dataset.gt_targets[viz_indices]

        viz_dataloader = torch.utils.data.DataLoader(viz_dataset, batch_size=4, shuffle=False)

        # 使用简化的matplotlib可视化（不依赖opencv）
        visualize_anomaly_maps_simple(pfe, ae, viz_dataloader, args, device, epochs)

        viz_result_path = './results/{}_{}_final_epoch_{}'.format(args.dataset, args.normal, epochs)
        logger.info("Anomaly map visualization completed. Results saved to: {}".format(viz_result_path))

    except Exception as e:
        logger.error("Failed to generate anomaly map visualization: {}".format(str(e)))
        logger.error("This might be due to missing visualization dependencies")
        logger.info("You can manually implement visualization using the anomaly_map data from evaluation_pixel()")

def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')


if __name__ == '__main__':

    args = parse_args()
    args.seed = setup_seed(args.seed)
    train(args)