import torch
from contextlib import nullcontext
import numpy as np
import random
import os
from util.test import evaluation
from model.model import PretrainedFeatureExtractor, ED, EnhancedDiscriminator
import logging
from argparse import ArgumentParser
from dataset.dataset import OODDataSet
from itertools import cycle
import tqdm
import torch.autograd as autograd


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
    
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='gradient penalty coefficient')
    
    parser.add_argument('--n_critic', type=int, default=5, help='number of critic iterations per generator iteration')
    
    parser.add_argument('--use_amp', action='store_true', help='enable mixed precision (AMP)')
    
    parser.add_argument('--compile', action='store_true', help='enable torch.compile for models if available')
    
    # New arguments for enhanced discriminator
    parser.add_argument('--discriminator_mode', type=str, default='image', choices=['image', 'patch'], 
                        help='discriminator mode: image-level or patch-level')
    
    parser.add_argument('--use_spectral_norm', action='store_true', default=True,
                        help='use spectral normalization in discriminator')
    
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='use multi-head attention in discriminator')
    
    parser.add_argument('--use_position_encoding', action='store_true', default=True,
                        help='use position encoding in discriminator')
    
    parser.add_argument('--margin', type=float, default=5.0, 
                        help='margin for hinge loss in patch-level discriminator')
    
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

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    batch_size = real_samples[0].size(0)
    alphas = torch.rand(batch_size, 1, 1, 1, device=device)

    # Get random interpolation between real and fake samples
    interpolates = []
    for real, fake in zip(real_samples, fake_samples):
        alpha = alphas.expand_as(real)
        interpolated_sample = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        interpolates.append(interpolated_sample)

    d_interpolates = discriminator(interpolates)
    
    # Handle different output types from different discriminator modes
    if isinstance(d_interpolates, list):  # Patch-level discriminator
        # Concatenate all patch scores for gradient calculation
        d_interpolates = torch.cat([score.view(batch_size, -1).mean(dim=1, keepdim=True) for score in d_interpolates], dim=1).mean(dim=1)
    
    grad_outputs = torch.ones_like(d_interpolates, requires_grad=False).to(device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )

    # Concatenate gradients from all interpolated tensors and calculate the norm
    gradients_flat = torch.cat([grad.view(batch_size, -1) for grad in gradients], dim=1)
    gradient_penalty = ((gradients_flat.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train(args):
    log_dir = os.path.join(args.log_dir, "lan{:.2f}_acn{}".format(args.labeled_anomaly_ratio,  args.labeled_anomaly_class_num), args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger_filename = os.path.join(log_dir, 'n_{}_a_{}_s_{}_{}_{}'.format(
        args.normal, args.labeled_anomaly_class, args.seed, 
        args.discriminator_mode, 'attn' if args.use_attention else 'no_attn') + '.txt')
    logger = get_logger(logger_filename)
    logger.info("log file: {}".format(logger_filename))
    logger.info("class: {}".format(args.normal))
    
    print_args(logger, args)
    epochs = args.epochs
    batch_size = args.batch_size
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("device: {}".format(device))

    dataset = OODDataSet(root='./data', dataset=args.dataset, image_size=args.img_size, category=args.normal,
                         labeled_anomaly_ratio=args.labeled_anomaly_ratio,
                         labeled_anomaly_class_num=args.labeled_anomaly_class_num,
                         labeled_anomaly_class=args.labeled_anomaly_class)
    train_dataloader, valid_dataloader, anomaly_dataloader, test_dataloader = dataset.get_data_loader(batch_size=batch_size)

    pfe = PretrainedFeatureExtractor(args.model, layers=args.layer, image_size=args.img_size).to(device)
    for param in pfe.parameters():
        param.requires_grad_(False)
    pfe.eval()
    ae = ED(backbone=args.model, input_channels=pfe.output_channels).to(device)
    
    # Use enhanced discriminator
    discriminator = EnhancedDiscriminator(
        input_sizes=pfe.output_sizes, 
        input_channels=pfe.output_channels, 
        expansion=pfe.expansion,
        mode=args.discriminator_mode,
        use_spectral_norm=args.use_spectral_norm,
        use_attention=args.use_attention,
        use_position_encoding=args.use_position_encoding,
        margin=args.margin
    ).to(device)
    
    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    
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

    for epoch in range(1, epochs + 1):
        ae.train()
        discriminator.train()
        d_loss_list, recon_loss_list, adv_loss_list, ae_loss_list = [], [], [], []

        for i, (normal, anomaly) in enumerate(tqdm.tqdm(zip(train_dataloader, cycle(anomaly_dataloader)))):
            normal_img = normal[0].to(device)
            if anomaly is not None:
                anomaly_img = anomaly[0].to(device)
            elif args.dataset in ['mvtec', 'visa', 'btad'] and len(normal) == 4:
                anomaly_img = normal[1].to(device)
            else:
                anomaly_img = normal_img[:0]
            anomaly_size = anomaly_img.size(0)

            # --------------------- #
            #  Train Discriminator  #
            # --------------------- #
            discriminator_optimizer.zero_grad()

            with amp_ctx():
                normal_inputs = pfe(normal_img)
                normal_outputs = ae(normal_inputs)

                # Detach to avoid training AE
                normal_outputs_detach = [o.detach() for o in normal_outputs]

                if args.discriminator_mode == 'patch':
                    # For patch-level discriminator, use calculate_loss method
                    real_loss = -discriminator.calculate_loss(normal_inputs, label_value=1)
                    fake_loss_recon = discriminator.calculate_loss(normal_outputs_detach, label_value=0)
                    
                    if anomaly_size > 0:
                        anomaly_inputs = pfe(anomaly_img)
                        fake_loss_anomaly = discriminator.calculate_loss(anomaly_inputs, label_value=0)
                    else:
                        fake_loss_anomaly = torch.tensor(0.0).to(device)
                        
                else:
                    # For image-level discriminator, use standard WGAN-GP approach
                    real_loss = -torch.mean(discriminator(normal_inputs))
                    fake_loss_recon = torch.mean(discriminator(normal_outputs_detach))
                    
                    if anomaly_size > 0:
                        anomaly_inputs = pfe(anomaly_img)
                        fake_loss_anomaly = torch.mean(discriminator(anomaly_inputs))
                    else:
                        fake_loss_anomaly = torch.tensor(0.0).to(device)

                # Prepare samples for gradient penalty
                if anomaly_size > 0:
                    fake_samples_for_gp = [torch.cat([no, ai]) for no, ai in zip(normal_outputs_detach, anomaly_inputs)]
                    real_samples_for_gp = [torch.cat([ni, ni]) for ni in normal_inputs]  # Use normal inputs twice to match size
                else:
                    fake_samples_for_gp = normal_outputs_detach
                    real_samples_for_gp = normal_inputs

                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_samples_for_gp, fake_samples_for_gp, device)

                # Total discriminator loss
                d_loss = fake_loss_recon + fake_loss_anomaly + real_loss + args.lambda_gp * gradient_penalty

            if use_cuda_amp:
                scaler_d.scale(d_loss).backward()
                scaler_d.step(discriminator_optimizer)
                scaler_d.update()
            else:
                d_loss.backward()
                discriminator_optimizer.step()
            
            d_loss_list.append(d_loss.item())

            # ----------------- #
            #  Train Generator  #
            # ----------------- #
            # Train the generator only once every n_critic iterations
            if i % args.n_critic == 0:
                ae_optimizer.zero_grad()

                with amp_ctx():
                    # We need to re-compute the outputs for the generator pass
                    normal_outputs_gen = ae(normal_inputs)
                    
                    # Reconstruction loss
                    recon_loss = loss_function(normal_inputs, normal_outputs_gen)
                    
                    # Adversarial loss
                    if args.discriminator_mode == 'patch':
                        # For patch-level discriminator
                        adv_loss = -discriminator.calculate_loss(normal_outputs_gen, label_value=1)
                    else:
                        # For image-level discriminator
                        adv_loss = -torch.mean(discriminator(normal_outputs_gen))
                    
                    # Total generator loss
                    ae_loss = recon_loss + args.adv_conf * adv_loss

                if use_cuda_amp:
                    scaler_ae.scale(ae_loss).backward()
                    scaler_ae.step(ae_optimizer)
                    scaler_ae.update()
                else:
                    ae_loss.backward()
                    ae_optimizer.step()

                recon_loss_list.append(recon_loss.item())
                adv_loss_list.append(adv_loss.item())
                ae_loss_list.append(ae_loss.item())

        logger.info(
            "epoch [{}/{}], d_loss: {:.6f}, recon_loss: {:.6f}, adv_loss: {:.6f}, ae_loss: {:.6f}".format(
                epoch, epochs, np.mean(d_loss_list), np.mean(recon_loss_list), np.mean(adv_loss_list), np.mean(ae_loss_list)
            )
        )
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
       
def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')


if __name__ == '__main__':

    args = parse_args()
    args.seed = setup_seed(args.seed)
    train(args) 