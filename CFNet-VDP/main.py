from __future__ import print_function, division
import argparse
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
import time
import cv2
from datetime import datetime
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss, reconstract_loss
from utils import *
from torch.utils.data import DataLoader
import gc

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
parser.add_argument('--model', default='cfnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--mode', default='train', help='train or eval')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
def denormalize(image_np):
    """将归一化的图像反归一化到 [0,1] 范围"""
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    return np.clip(image_np * std + mean, 0, 1)  # 限制到 [0,1]
def forward_warp_vertical_interp(feat, offset_y):
    """
    Y方向前向变形（完全保持输入特征图的数据类型和数值范围）
    Args:
        feat: 输入特征图 [B, C, H, W] (任意合法的torch dtype)
        offset_y: Y方向偏移量 [B, H, W, 1] 或 [B, 1, H, W]（正数向下）
    Returns:
        变形后的特征图 [B, C, H, W] (与feat同dtype同范围)
    """
    B, C, H, W = feat.shape
    device = feat.device
    dtype = feat.dtype  # 保持输入数据类型
    feat = feat.float() / 255.0  # 归一化到 [0, 1]
    # 统一offset_y形状为 [B, H, W] 并匹配设备
    offset_y = offset_y.to(device=device)
    if offset_y.dim() == 4:
        offset_y = offset_y.squeeze(-1) if offset_y.size(-1) == 1 else offset_y.squeeze(1)
    
    # 生成坐标网格（与feat同设备）
    y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
    x_coords = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)
    
    # 计算新坐标和插值权重（保持计算精度）
    y_offset = y_coords - offset_y
    y0 = torch.clamp(y_offset.floor().long(), 0, H - 1)
    y1 = torch.clamp(y0 + 1, 0, H - 1)
    w1 = (y_offset - y0.float()).unsqueeze(1)  # [B, 1, H, W]
    w0 = 1.0 - w1
    
    # 初始化输出（与feat同类型同设备）
    warped = torch.zeros_like(feat)
    count = torch.zeros_like(feat)
    
    # 向量化 scatter_add（Y方向）
    warped.scatter_add_(2, y0.unsqueeze(1).expand(-1, C, -1, -1), feat * w0.expand(-1, C, -1, -1))
    warped.scatter_add_(2, y1.unsqueeze(1).expand(-1, C, -1, -1), feat * w1.expand(-1, C, -1, -1))
    count.scatter_add_(2, y0.unsqueeze(1).expand(-1, C, -1, -1), w0.expand(-1, C, -1, -1))
    count.scatter_add_(2, y1.unsqueeze(1).expand(-1, C, -1, -1), w1.expand(-1, C, -1, -1))
    
    # 归一化（避免除零，保持数值范围）
    warped = warped / (count + 1e-6)
    
    return (warped * 255.0).clamp(0, 255).to(torch.uint8)
def save_images1(image_outputs, save_dir="saved_images", prefix="batch"):
    """
    保存 7 种图像：
    - imgL, imgR, disp_gt, disp_ests, n_imgR, recon_right, warp_left
    """
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    
    # 保存左图 (imgL)
    vutils.save_image(
        image_outputs["imgL"], 
        os.path.join(save_dir, f"{prefix}_imgL.png"),
        normalize=True  # 自动归一化到 [0, 1]
    )
    
    # 保存右图 (imgR)
    vutils.save_image(
        image_outputs["imgR"], 
        os.path.join(save_dir, f"{prefix}_imgR.png"),
        normalize=True
    )
    
    # 保存真实视差图 (disp_gt)
    vutils.save_image(
        image_outputs["disp_gt"], 
        os.path.join(save_dir, f"{prefix}_disp_gt.png"),
        normalize=True
    )
def custom_grid_sample(left_img_A, offset):
    """
    Args:
        left_img_A: Input image tensor of shape (b, c, h, w), range [0,255] (uint8) or [0,1] (float).
        offset: Offset tensor of shape (b, h, w, 2), where:
                offset[..., 0] is x (width) offset in PIXELS (not normalized).
                offset[..., 1] is y (height) offset in PIXELS (not normalized).
    Returns:
        Sampled image tensor of shape (b, c, h, w).
    """
    # 1. 设备和类型检查
    device = left_img_A.device
    offset = offset.to(device).float()  # 确保offset是float且在同一设备
    
    # 2. 标准化输入图像到[0,1]范围
    if left_img_A.dtype == torch.uint8:
        left_img_A = left_img_A.float() / 255.0
    else:
        left_img_A = left_img_A.float()
    
    b, c, h, w = left_img_A.shape

    # 3. 将offset从像素坐标归一化到[-1, 1]
    #    注意：PyTorch的grid_sample中x对应宽度方向（第二维），y对应高度方向（第一维）
    offset_normalized = torch.zeros_like(offset)
    offset_normalized[..., 0] = offset[..., 0] / (w - 1) * 2  # x: [-1, 1]
    offset_normalized[..., 1] = offset[..., 1] / (h - 1) * 2  # y: [-1, 1]

    # 4. 生成标准网格（[-1,1]范围）
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
    )
    grid = torch.stack((grid_x, grid_y), dim=-1)  # (h, w, 2)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)  # (b, h, w, 2)

    # 5. 将归一化后的offset加到网格上
    grid = grid + offset_normalized

    # 6. 执行采样
    sampled = F.grid_sample(
        left_img_A,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    # 7. 输出类型与输入一致
    if left_img_A.dtype == torch.uint8:
        sampled = (sampled.clamp(0, 1) * 255).byte()
    return sampled

def train():
    bestepoch = 0
    error = 100
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        #bestepoch = 0
        #error = 100
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["D1"][0]
        if  nowerror < error :
            bestepoch = epoch_idx
            error = avg_test_scalars["D1"][0]
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
        gc.collect()
    print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
def test():
    avg_test_scalars = AverageMeterDict()
        #bestepoch = 0
        #error = 100
    for batch_idx, sample in enumerate(TestImgLoader):
        global_step = batch_idx
        start_time = time.time()
        do_summary = global_step % args.summary_freq == 0
        loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            # loss, scalar_outputs = test_sample(sample, compute_metrics=do_summary)
        # if do_summary:
            # save_scalars(logger, 'test', scalar_outputs, global_step)
            # save_images(logger, 'test', image_outputs, global_step)
        avg_test_scalars.update(scalar_outputs)
        # del scalar_outputs
        del scalar_outputs, image_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx,
                                                                    len(TestImgLoader), loss,
                                                                    time.time() - start_time))
    avg_test_scalars = avg_test_scalars.mean()
    
    save_scalars(logger, 'fulltest', avg_test_scalars, 1)
    print("avg_test_scalars", avg_test_scalars)
    gc.collect()
def visualize_and_save_images(left_img_A, left_img_Ap, batch=0, save_dir="./outputs"):
    """
    Visualize and save two RGB images side by side.
    
    Args:
        left_img_A: Tensor of shape (b, 3, h, w).
        left_img_Ap: Tensor of shape (b, 3, h, w).
        batch: Batch index to visualize (default=0).
        save_dir: Directory to save the PNG file (default="./outputs").
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays (HWC format)
    img_A = left_img_A[batch].permute(1, 2, 0).cpu().detach().numpy()  # (h, w, 3)
    img_Ap = left_img_Ap[batch].permute(1, 2, 0).cpu().detach().numpy()  # (h, w, 3)
    
    # Clip values to [0, 1] if necessary (assuming input is float32)
    # img_A = img_A.clip(0, 1)
    # img_Ap = img_Ap.clip(0, 1)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot left_img_A
    ax1.imshow(img_A)
    ax1.set_title('left_img_A')
    ax1.axis('off')
    
    # Plot left_img_Ap
    ax2.imshow(img_Ap)
    ax2.set_title('left_img_Ap')
    ax2.axis('off')
    
    # Add timestamp and save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"compare_{timestamp}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved visualization to: {save_path}")

# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt, left_img_A, left_img_Ap, offset = sample['left'], sample['right'], sample['disparity'], sample['left_img_A'], sample['left_img_Ap'], sample['offset']

    left_img_A = left_img_A.cuda()
    left_img_A = left_img_A.permute(0, 3, 1, 2)
    left_img_Ap = left_img_Ap.cuda()
    left_img_Ap = left_img_Ap.permute(0, 3, 1, 2)
    offset = offset.cuda()
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    b,c,h,w = imgL.shape
    optimizer.zero_grad()

    disp_ests, offset_y = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    left_img_A = custom_grid_sample(left_img_A, offset)

    # recon_right_pil = Image.fromarray(recon_left[0].permute(1, 2, 0).cpu().numpy().astype('uint8'))
    # right_gt_pil = Image.fromarray(left_gt[0].permute(1, 2, 0).cpu().numpy().astype('uint8'))
    # # # 保存图像
    # recon_right_pil.save("recon_left_b0.png")
    # right_gt_pil.save("left_gt_b0.png")

    n_mask = (left_img_Ap > 0) & (left_img_A > 0)
    recon_loss = reconstract_loss(left_img_Ap, left_img_A, n_mask)
    # visualize_and_save_images(left_img_A, left_img_Ap, batch=0)

    loss = model_loss(disp_ests, disp_gt, mask) + recon_loss

    scalar_outputs = {"loss": loss, "recon_loss": recon_loss}

    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()

        # 可视化部分
    save_dir = './testing_vis2'
    final_disp = disp_ests[-1]
    with torch.no_grad():
        # 准备数据
        imgL_np = imgL[0].cpu().permute(1, 2, 0).numpy()  # [H,W,3]
        imgR_np = imgR[0].cpu().permute(1, 2, 0).numpy()  # [H,W,3]
        disp_pred_np = final_disp[0].cpu().numpy()  # [H,W]
        disp_gt_np = disp_gt[0].cpu().numpy()  # [H,W]
        left_img_Ap = left_img_Ap[0].cpu().permute(1, 2, 0).numpy() 
        left_img_A = left_img_A[0].cpu().permute(1, 2, 0).numpy() 
        # 创建可视化
        plt.figure(figsize=(20, 24))  # 增大画布高度以适应4行子图
        
        # 左图
        plt.subplot(4, 2, 1)
        plt.imshow(denormalize(imgL_np))
        plt.title('Left Image')
        plt.axis('off')
        
        # 右图
        plt.subplot(4, 2, 2)
        plt.imshow(denormalize(imgR_np))
        plt.title('Right Image')
        plt.axis('off')
        
        # 预测视差
        plt.subplot(4, 2, 3)
        plt.imshow(disp_pred_np, cmap='jet', vmin=0, vmax=args.maxdisp)
        plt.colorbar(label='Disparity (pixels)')
        plt.title('Predicted Disparity')
        plt.axis('off')
        
        # 真实视差
        plt.subplot(4, 2, 4)
        plt.imshow(disp_gt_np, cmap='jet', vmin=0, vmax=args.maxdisp)
        plt.colorbar(label='Disparity (pixels)')
        plt.title('Ground Truth Disparity')
        plt.axis('off')
        
        # 重建左图
        plt.subplot(4, 2, 5)
        plt.imshow(left_img_Ap)
        plt.title('left_img_Ap')
        plt.axis('off')
        
        # 真实左图
        plt.subplot(4, 2, 6)
        plt.imshow(left_img_A)
        plt.title('left_img_A')
        plt.axis('off')
        
       
        # 调整子图间距
        plt.tight_layout()
        
        # 保存结果
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f'train_vis_{timestamp}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"可视化结果已保存至: {save_path}")

    #     stats = {
    #     'Offset Map': {
    #         'min': offset_np.min(),
    #         'max': offset_np.max(),
    #         'mean': offset_np.mean(),
    #         'std': offset_np.std()  # 额外保存标准差
    #         }
    #     }
    
    #     # 将统计信息保存为文本文件
    #     stats_save_path = os.path.join(save_dir, f'stats_{timestamp}.txt')
    #     with open(stats_save_path, 'w') as f:
    #         for name, values in stats.items():
    #             f.write(f"{name}:\n")
    #             for stat, val in values.items():
    #                 f.write(f"  {stat}: {val:.6f}\n")
    #             f.write("\n")
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests, pred_s3, pred_s4 = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)
    save_dir='./testing_vis5'
    os.makedirs(save_dir, exist_ok=True)
    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["D1_pred_s2"] = [D1_metric(pred, disp_gt, mask) for pred in pred_s2]
    scalar_outputs["D1_pred_s3"] = [D1_metric(pred, disp_gt, mask) for pred in pred_s3]
    scalar_outputs["D1_pred_s4"] = [D1_metric(pred, disp_gt, mask) for pred in pred_s4]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
    # # 可视化部分
    # final_disp = disp_ests[0]
    # with torch.no_grad():
    #     # 准备数据
    #     imgL_np = imgL[0].cpu().permute(1, 2, 0).numpy()  # [H,W,3]
    #     imgR_np = imgR[0].cpu().permute(1, 2, 0).numpy()  # [H,W,3]
    #     disp_pred_np = final_disp[0].cpu().numpy()  # [H,W]
    #     disp_gt_np = disp_gt[0].cpu().numpy()  # [H,W]


    #     imgL_np = denormalize(imgL_np)
    #     imgR_np = denormalize(imgR_np)

    #     # 创建可视化
    #     plt.figure(figsize=(20, 12))
        
    #     # 左图
    #     plt.subplot(2, 2, 1)
    #     plt.imshow(imgL_np)
    #     plt.title('Left Image')
    #     plt.axis('off')
        
    #     # 右图
    #     plt.subplot(2, 2, 2)
    #     plt.imshow(imgR_np)
    #     plt.title('Right Image')
    #     plt.axis('off')
        
    #     # 预测视差
    #     plt.subplot(2, 2, 3)
    #     plt.imshow(disp_pred_np, cmap='jet', vmin=0, vmax=args.maxdisp)
    #     plt.colorbar(label='Disparity (pixels)')
    #     plt.title('Predicted Disparity')
    #     plt.axis('off')
        
    #     # 真实视差
    #     plt.subplot(2, 2, 4)
    #     plt.imshow(disp_gt_np, cmap='jet', vmin=0, vmax=args.maxdisp)
    #     plt.colorbar(label='Disparity (pixels)')
    #     plt.title('Ground Truth Disparity')
    #     plt.axis('off')

    #     # 保存结果
    #     os.makedirs(save_dir, exist_ok=True)
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     save_path = os.path.join(save_dir, f'train_vis_{timestamp}.png')
    #     plt.savefig(save_path, bbox_inches='tight', dpi=150)
    #     plt.close()
    #     print(f"可视化结果已保存至: {save_path}")
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        test()
