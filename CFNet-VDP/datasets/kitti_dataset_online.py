import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
from . import flow_transforms
import torchvision
import cv2
import copy
import torch.nn.functional as F
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
import random
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import glob
feed_width = 640
feed_height = 480

process_width = feed_width
process_height = feed_height
def get_occlusion_mask(shifted):
    mask_up = shifted > 0
    mask_down = shifted > 0
    #print(mask_down)

    shifted_up = np.ceil(shifted)
    shifted_down = np.floor(shifted)
    # 如果多个点落在一个位置，选视差最大的。
    for col in range(process_width - 2):
        loc = shifted[:, col:col + 1]  # keepdims
        loc_up = np.ceil(loc)
        loc_down = np.floor(loc)

        _mask_down = ((shifted_down[:, col + 2:] != loc_down) * (
            (shifted_up[:, col + 2:] != loc_down))).min(-1)
        _mask_up = ((shifted_down[:, col + 2:] != loc_up) * (
            (shifted_up[:, col + 2:] != loc_up))).min(-1)

        mask_up[:, col] = mask_up[:, col] * _mask_up
        mask_down[:, col] = mask_down[:, col] * _mask_down

    mask = mask_up + mask_down
    return mask


def project_image(image, disp_map):
    image = np.array(image)
    # background_image = np.array(background_image)
    xs, ys = np.meshgrid(np.arange(process_width), np.arange(process_height))
    #print(xs)
    # set up for projection
    warped_image = np.zeros_like(image).astype(float)
    warped_image = np.stack([warped_image] * 2, 0)
    pix_locations = xs - disp_map
    #print(pix_locations)

    # find where occlusions are, and remove from disparity map
    mask = get_occlusion_mask(pix_locations)
    #print(mask)
    # 用来输出mask
    occ_mask = mask.copy()
    # inputs['occ_mask'] = mask
    masked_pix_locations = pix_locations * mask - process_width * (1 - mask)

    # do projection - linear interpolate up to 1 pixel away
    weights = np.ones((2, process_height, process_width)) * 10000

    for col in range(process_width - 1, -1, -1):
        loc = masked_pix_locations[:, col]
        loc_up = np.ceil(loc).astype(int)
        loc_down = np.floor(loc).astype(int)
        weight_up = loc_up - loc
        weight_down = 1 - weight_up

        mask = (loc_up >= 0) & (loc_up < process_width)
        mask[mask] = \
            weights[0, np.arange(process_height)[mask], loc_up[mask]] > weight_up[mask]
        weights[0, np.arange(process_height)[mask], loc_up[mask]] = \
            weight_up[mask]
        warped_image[0, np.arange(process_height)[mask], loc_up[mask]] = \
            image[:, col][mask] / 255.

        mask = (loc_down >= 0) & (loc_down < process_width)
        mask[mask] = \
            weights[1, np.arange(process_height)[mask], loc_down[mask]] > weight_down[mask]
        weights[1, np.arange(process_height)[mask], loc_down[mask]] = weight_down[mask]
        warped_image[1, np.arange(process_height)[mask], loc_down[mask]] = \
            image[:, col][mask] / 255.

    weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
    weights = np.expand_dims(weights, -1)
    warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
    warped_image *= 255.

    # now fill occluded regions with random background
    # if background_image is not None:
    #     warped_image[warped_image.max(-1) == 0] = background_image[warped_image.max(-1) == 0]
    warped_image = warped_image.astype(np.uint8)

    return warped_image, occ_mask

def build_delta_pose(angle_deg=(0, 0, 0), trans=(0, 0, 0)):
    R_delta = R.from_euler("xyz", angle_deg, degrees=True).as_matrix().astype(np.float32)
    t_delta = np.array(trans, dtype=np.float32)
    return R_delta, t_delta

def warp_rgb_depth_fusion_zbuffer(rgb, depth, K, R_delta, t_delta, fill_value=0, depth_tol=0.05):
    H, W = depth.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    z = depth.ravel()
    valid = z > 0
    u, v, z = u.ravel()[valid], v.ravel()[valid], z[valid]

    # 3D变换 - 向量化
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    pts_new = (R_delta @ pts.T).T + t_delta

    # 投影回2D - 向量化
    x_n, y_n, z_n = pts_new[:, 0], pts_new[:, 1], pts_new[:, 2]
    u_f = (K[0, 0] * x_n / z_n) + K[0, 2]
    v_f = (K[1, 1] * y_n / z_n) + K[1, 2]
    
    # 向量化计算 in_view
    in_view = (u_f >= 0) & (u_f < W - 1) & (v_f >= 0) & (v_f < H - 1) & (z_n > 0)

    # 向量化计算偏移量 (from original to transformed)
    y_old = v.astype(int)
    x_old = u.astype(int)
    offset_map = np.zeros((H, W, 2), dtype=np.float32)
    offset_map[y_old, x_old, 0] = u_f - x_old
    offset_map[y_old, x_old, 1] = v_f - y_old

    u_f = u_f[in_view]
    v_f = v_f[in_view]
    z_n = z_n[in_view]
    
    # 向量化双线性插值
    x0 = np.floor(u_f).astype(int)
    y0 = np.floor(v_f).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    wx1 = u_f - x0
    wy1 = v_f - y0
    wx0 = 1 - wx1
    wy0 = 1 - wy1
    
    # 获取颜色 - 向量化
    colors = rgb[v[in_view], u[in_view]].astype(np.float32)
    
    # 初始化累加器
    rgb_accum = np.zeros((H, W, 3), dtype=np.float32)
    weight_accum = np.zeros((H, W), dtype=np.float32)
    depth_min = np.full((H, W), np.inf, dtype=np.float32)
    
    # 处理四个角点
    for (yy, xx, w) in [(y0, x0, wx0 * wy0),
                        (y0, x1, wx1 * wy0),
                        (y1, x0, wx0 * wy1),
                        (y1, x1, wx1 * wy1)]:
        mask = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        yy_m = yy[mask]
        xx_m = xx[mask]
        w_m = w[mask]
        
        # 向量化更新
        depth_val = z_n[mask]
        depth_mask = depth_val < (depth_min[yy_m, xx_m] + depth_tol)
        
        # 更新RGB
        rgb_accum[yy_m[depth_mask], xx_m[depth_mask]] += colors[mask][depth_mask] * w_m[depth_mask, None]
        weight_accum[yy_m[depth_mask], xx_m[depth_mask]] += w_m[depth_mask]
        
        # 更新深度
        depth_update = depth_val < depth_min[yy_m, xx_m]
        depth_min[yy_m[depth_update], xx_m[depth_update]] = depth_val[depth_update]

    # 归一化处理
    mask = weight_accum > 1e-5
    rgb_final = np.zeros_like(rgb_accum, dtype=np.uint8)
    rgb_final[mask] = (rgb_accum[mask] / weight_accum[mask, None]).astype(np.uint8)

    depth_final = np.zeros_like(depth_min)
    depth_final[mask] = depth_min[mask]
    depth_final[~mask] = 0
    
    # 计算反向偏移量 (from transformed back to original)
    # 首先为变换后的深度图中的每个有效像素创建坐标网格
    v_new, u_new = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    z_new = depth_final.ravel()
    valid_new = z_new > 0
    u_new, v_new, z_new = u_new.ravel()[valid_new], v_new.ravel()[valid_new], z_new[valid_new]
    
    # 3D逆变换 - 向量化
    x_new = (u_new - K[0, 2]) * z_new / K[0, 0]
    y_new = (v_new - K[1, 2]) * z_new / K[1, 1]
    pts_new = np.stack([x_new, y_new, z_new], axis=1)
    pts_original = (R_delta.T @ (pts_new - t_delta).T).T
    
    # 投影回原始2D - 向量化
    x_o, y_o, z_o = pts_original[:, 0], pts_original[:, 1], pts_original[:, 2]
    u_o = (K[0, 0] * x_o / z_o) + K[0, 2]
    v_o = (K[1, 1] * y_o / z_o) + K[1, 2]
    
    # 创建反向偏移图
    inverse_offset_map = np.zeros((H, W, 2), dtype=np.float32)
    inverse_offset_map[v_new.astype(int), u_new.astype(int), 0] = u_o - u_new
    inverse_offset_map[v_new.astype(int), u_new.astype(int), 1] = v_o - v_new
    # 创建空区域掩码 (1表示有像素值，0表示无像素值)
    empty_mask = np.zeros((H, W), dtype=np.uint8)
    empty_mask[mask] = 1  # mask来自weight_accum > 1e-5的判断
    return rgb_final, depth_final, offset_map, inverse_offset_map, empty_mask
def warp_disp_with_offset(disp, offset_t, mask):
    """
    Warp disparity using grid_sample.
    
    Args:
        disp: (H, W) original disparity map (numpy or torch.Tensor)
        offset_t: (H, W, 2) offset map [du, dv] (numpy or torch.Tensor)
    
    Returns:
        disp_new: (H, W) warped disparity (numpy)
    """
    # Convert inputs to PyTorch tensors if needed
    if not isinstance(disp, torch.Tensor):
        disp = torch.from_numpy(disp).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    if not isinstance(offset_t, torch.Tensor):
        offset_t = torch.from_numpy(offset_t).float()  # (H, W, 2)

    H, W = disp.shape[-2:]
    device = disp.device

    # Create normalized grid for left_new (in [-1, 1])
    grid_u = torch.linspace(-1, 1, W, device=device)
    grid_v = torch.linspace(-1, 1, H, device=device)
    grid_v, grid_u = torch.meshgrid(grid_v, grid_u)  # (H, W)
    
    grid = torch.stack([grid_u, grid_v], dim=-1)  # (H, W, 2)

    # Convert offset_t from pixel units to normalized units
    scale = torch.tensor([2./W, 2./H], device=device)
    offset_norm = offset_t * scale  # (H, W, 2)
    grid_warped = grid + offset_norm  # (H, W, 2)

    # Sample disparity
    disp_new = F.grid_sample(
        disp,  # (1, 1, H, W)
        grid_warped.unsqueeze(0),  # (1, H, W, 2)
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze()  # (H, W)
    disp_new = disp_new - offset_t[..., 0]
    disp_new = disp_new * mask
    return disp_new.cpu().numpy() if disp_new.is_cuda else disp_new.numpy()
class KITTIDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames, self.n_left_filenames, self.offests = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        n_left_images = [x[3] for x in splits]
        offsets = [x[4] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images, n_left_images, offsets

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32)
        #data = data*192. #warp最大视差为192
        #data = np.mean(data, axis=2)
        return data
    def stereo_aug_sym(self, img1, img2, disp):
        
        
        #img1,img2,disp = self.spatial_transform(img1, img2, disp)
        
        wW,hH = img1.size
        
        img = Image.new('RGB', (2*wW, hH), (255,0,0))
        img.paste(img1,(0,0)) 
        img.paste(img2,(wW,0)) 
        
        w,h = img.size
        
        
        stereo_brightness = (0.5, 1.5)
        stereo_contrast = (0.5, 1.5)
        stereo_saturation = (0.5, 1.5)
        stereo_hue = (-0.05, 0.05)
        
        
        # random_number_color = random.random()
        random_number = random.random()
        if random_number<=0.8:
            aug_color = transforms.ColorJitter(stereo_brightness, stereo_contrast, stereo_saturation,stereo_hue)
            img = aug_color(img)
        
        img_np = np.array(img)
        
        random_number = random.random()    
        if random_number<=0.8:
            noise = np.random.randn(h, w, 3) / 50.0  
            #print(img_np.shape)
            img_np = np.clip(img_np / 255 + noise, 0, 1) * 255
        
        random_number = random.random()
        if random_number<=0.8:
        
            img_np = gaussian_filter(img_np,sigma=random.random())
       

        img = Image.fromarray(img_np.astype('uint8'))
        
        trans_img1 = img.crop((0,0,wW,hH))
        trans_img2 = img.crop((wW,0,2*wW,hH))
        return trans_img1, trans_img2, disp
    
    def stereo_aug_asym(self, img):
        w,h = img.size
        
        stereo_brightness = (0.5, 1.5)
        stereo_contrast = (0.5, 1.5)
        stereo_saturation = (0.5, 1.5)
        stereo_hue = (-0.05, 0.05)
        aug_color = transforms.ColorJitter(stereo_brightness, stereo_contrast, stereo_saturation,stereo_hue)
        
        # random_number_color = random.random()
        random_number = random.random()
        if random_number<=0.8:
            img = aug_color(img)
        
        img_np = np.array(img)
        # random_number = random.random()  
        # if random_number<=0.8:
        #     img_np = self.eraser_transform(img_np)
           
        random_number = random.random()
        if random_number<=0.8:    
            noise = np.random.randn(h, w, 3) / 50.0
            
            img_np = np.clip(img_np / 255 + noise, 0, 1) * 255
        
        random_number = random.random()
        if random_number<=0.8:
            img_np = gaussian_filter(img_np,sigma=random.random())
        img = Image.fromarray(img_np.astype('uint8'))
        
        return img
    def forward_warp_horizontal_interp(self, img, offset_x):
        B, C, H, W = img.shape
        device = img.device
        img = img.float() / 255.0  # 归一化到 [0, 1]
        
        # 处理 offset_x 形状（假设输入是 [B, H, W, 1]）
        offset_x = offset_x.permute(0, 3, 1, 2).squeeze(1)  # [B, H, W]
        
        # 生成坐标网格
        y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
        x_coords = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)
        
        # 计算新坐标和插值权重
        x_offset = x_coords + offset_x
        x0 = torch.clamp(x_offset.floor().long(), 0, W - 1)
        x1 = torch.clamp(x0 + 1, 0, W - 1)
        w1 = (x_offset - x0.float()).unsqueeze(1)  # [B, 1, H, W]
        w0 = 1.0 - w1
        
        # 初始化输出
        warped = torch.zeros_like(img)
        count = torch.zeros_like(img)  # 用于归一化
        
        # 向量化 scatter_add
        warped.scatter_add_(3, x0.unsqueeze(1).expand(-1, C, -1, -1), img * w0.expand(-1, C, -1, -1))
        warped.scatter_add_(3, x1.unsqueeze(1).expand(-1, C, -1, -1), img * w1.expand(-1, C, -1, -1))
        count.scatter_add_(3, x0.unsqueeze(1).expand(-1, C, -1, -1), w0.expand(-1, C, -1, -1))
        count.scatter_add_(3, x1.unsqueeze(1).expand(-1, C, -1, -1), w1.expand(-1, C, -1, -1))
        
        # 归一化（解决冲突）
        warped = warped / (count + 1e-6)
        
        # 恢复范围并返回
        return (warped * 255.0).clamp(0, 255).to(torch.uint8)

    def warp_image_x(self, image, offset_x):
        """
        根据x方向偏移量变形图像（支持批量处理）
        
        参数:
            image: torch.Tensor [1, 3, H, W] 输入图像
            offset_x: torch.Tensor [1, H, W, 1] x方向偏移量（单位：像素）
        
        返回:
            warped: torch.Tensor [1, 3, H, W] 变形后的图像
        """
        # 确保输入形状正确
        assert image.dim() == 4 and offset_x.dim() == 4
        assert image.size(0) == offset_x.size(0)  # batch size匹配
        image = image.float() / 255.0
        # 准备采样网格
        B, C, H, W = image.shape

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=image.device),
            torch.arange(W, device=image.device),
        )
        # 归一化到[-1,1]范围（F.grid_sample的要求）
        grid_x = (2.0 * grid_x.float() / (W - 1)) - 1.0
        grid_y = (2.0 * grid_y.float() / (H - 1)) - 1.0
        
        # 应用x方向偏移（注意偏移方向与坐标系的转换）
        offset_x_norm = offset_x.squeeze(-1)
        offset_x_norm = (2.0 * offset_x_norm / (W - 1))  # 像素偏移 -> 归一化偏移

        grid_x = grid_x.unsqueeze(0) - offset_x_norm # [1,H,W]
        #warped[y,x] = image[y,x-offset_x[y,x]] 感觉代码没有实现这个思想！！！！！！！！！！

        # 创建采样网格 [1,H,W,2]
        grid = torch.stack((grid_x, grid_y.unsqueeze(0).expand(B,-1,-1)), dim=-1)
        
        # 双线性插值采样
        warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        warped = (warped * 255.0).clamp(0, 255).byte()
        return warped

    def load_offset(self, filename):
        data = np.load(filename)
        return data

    def add_noise(self, image):
        # add some noise to stereo image
        
        noise = np.random.randn(image.shape[0], image.shape[1], 3) / 50
        image = np.clip(image / 255 + noise, 0, 1) * 255

        # add blurring
        if random.random() > 0.5:
            image = gaussian(image,
                            sigma=random.random(),
                            channel_axis=-1)

        image = np.clip(image, 0, 255)
        return image

    # def RGB2GRAY(self, img):
    #     imgG = copy.deepcopy(img)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     imgG[:, :, 0] = img
    #     imgG[:, :, 1] = img
    #     imgG[:, :, 2] = img
    #     return imgG

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        #right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        #n_left_img = self.load_image(os.path.join(self.datapath, self.n_left_filenames[index]))
        #offset_ori = self.load_offset(os.path.join(self.datapath, self.offests[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        #====================求right_img,n_left_img,offset_ori,disparity===========================================================
        
        # 生成 185 到 215 之间的随机数
        dmax = random.randint(60, 246)
        dmin = random.randint(5, 10)
        disparity = disparity / 65536.0 #逆深度归一化 H,W
        disparity = disparity * dmax + dmin
        # print("d_true min:", np.min(d_true))
        # print("d_true max:", np.max(d_true))

        b = 0.25 #基线
        H, W = 480, 640
        fx = fy = 0.6 * W
        cx = W / 2
        cy = H / 2
        # 针孔内参矩阵
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0,  1]], dtype=np.float32)
        d_true = (b * fx) / disparity

        warp_image, occ_mask = project_image(left_img, disparity)
        right_img = Image.fromarray(warp_image.astype('uint8'))  # numpy -> PIL


        # 旋转扰动（单位：度）
        angle_x = np.random.uniform(-0.5, 0.5)     # X 旋转：小范围俯仰
        angle_y = np.random.uniform(-2.0, 2.0)     # Y 旋转：加大水流偏航
        angle_z = np.random.uniform(-1.5, 1.5)     # Z 旋转：加大波浪滚转
        # 平移扰动（单位：米）
        trans_x = np.random.uniform(-0.005, 0.005)   # X 平移：适中水平扰动
        trans_y = np.random.uniform(-0.015, 0.015)   # Y 平移：最大化垂直水流扰动
        trans_z = np.random.uniform(-0.005, 0.005)   # Z 平移：适当前后抖动
        # 生成当前图像的扰动矩阵
        R_delta, t_delta = build_delta_pose(
            angle_deg=(angle_x, angle_y, angle_z),
            trans=(trans_x, trans_y, trans_z)
        )
        # ---------------------------------------------------

        left_img1 = np.array(left_img)
        left_new, depth_new, offset, offset_t, mask = warp_rgb_depth_fusion_zbuffer(left_img1, d_true, K, R_delta, t_delta)
        n_left_img = Image.fromarray(left_new.astype('uint8'))  # numpy -> PIL


        disparity = warp_disp_with_offset(disparity, offset_t, mask)

        # right_gt, occ_mask = project_image(n_left_img, disparity1)

        #====================求right_img,n_left_img,offset_ori,disparity===========================================================
        # output_dir = "/mnt/sda/wuqizheng/underwater/CFNet-main_v2/3"
        
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # plt.figure(figsize=(18, 12))
        # # 1. 视差图
        # plt.subplot(3, 2, 5)
        # plt.imshow(depth_new, cmap="viridis")
        # plt.colorbar(label="depth_new (px)")
        # plt.title(f"depth_new Map (min={np.min(depth_new):.2f}, max={np.max(depth_new):.2f})")

        # plt.subplot(3, 2, 3)
        # plt.imshow(disparity, cmap="viridis")
        # plt.colorbar(label="Disparity (px)")
        # plt.title(f"Disparity Map (min={np.min(disparity):.2f}, max={np.max(disparity):.2f})")



        # # 2. 右图
        # plt.subplot(3, 2, 2)
        # plt.imshow(warp_image)

        # # 3. 扰动左图
        # plt.subplot(3, 2, 1)
        # plt.imshow(left_new)
        # plt.title("Warped Image")

        # plt.subplot(3, 2, 4)
        # plt.imshow(right_gt)
        # plt.title("right_gt")

        # plt.subplot(3, 2, 6)
        # plt.imshow(disparity1, cmap="viridis")
        # plt.colorbar(label="Disparity (px)")
        # plt.title(f"Disparity Map (min={np.min(disparity1):.2f}, max={np.max(disparity1):.2f})")

        # plt.tight_layout()
        # plt.savefig(f"{output_dir}/visualization_{timestamp}.png", dpi=300, bbox_inches='tight')



        left_img_A = np.array(left_img) 
    #===============================================================================
        left_img_Ap = n_left_img.copy()
        left_img_Ap = np.array(left_img_Ap)
    #===============================================================================

    #===============================================================================
        if self.training:
            th, tw = 256, 512
            #th, tw = 320, 704
            n_left_img, right_img,disparity = self.stereo_aug_sym(n_left_img, right_img,disparity)
            right_img = self.stereo_aug_asym(right_img)
            right_img = np.asarray(right_img)
            n_left_img = np.asarray(n_left_img)

            # random_brightness = np.random.uniform(0.5, 2.0, 3)
            # random_gamma = np.random.uniform(0.8, 1.2, 3)
            # random_contrast = np.random.uniform(0.8, 1.2, 3)
            # left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            # left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            # left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            # right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            # right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            # right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
            # n_right_img = torchvision.transforms.functional.adjust_brightness(n_right_img, random_brightness[2])
            # n_right_img = torchvision.transforms.functional.adjust_gamma(n_right_img, random_gamma[2])
            # n_right_img = torchvision.transforms.functional.adjust_contrast(n_right_img, random_contrast[2])
            # right_img = np.array(right_img)
            # left_img = np.array(left_img)
            # n_right_img = np.array(n_right_img)

            # # right_img = self.add_noise(right_img).astype(np.float32)
            # # n_right_img = self.add_noise(n_right_img).astype(np.float32)
            

            # # w, h  = left_img.size
            # # th, tw = 256, 512
            # #
            # # x1 = random.randint(0, w - tw)
            # # y1 = random.randint(0, h - th)
            # #
            # # left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            # # right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            # # dataL = dataL[y1:y1 + th, x1:x1 + tw]
            # # right_img = np.asarray(right_img)
            # # left_img = np.asarray(left_img)

            # # geometric unsymmetric-augmentation
            # angle = 0
            # px = 0
            # if np.random.binomial(1, 0.5):
            #     # angle = 0.1;
            #     # px = 2
            #     angle = 0.05
            #     px = 1
            co_transform = flow_transforms.Compose([
                # flow_transforms.RandomVdisp(angle, px),
                # flow_transforms.Scale(np.random.uniform(self.rand_scale[0], self.rand_scale[1]), order=self.order),
                flow_transforms.RandomCrop((th, tw)),
            ])
    #===============================================================================
            augmented, disparity = co_transform([n_left_img, right_img, left_img_Ap, left_img_A, offset_t], disparity)
            n_left_img = augmented[0]
            right_img = augmented[1]
            left_img_Ap = augmented[2]
            left_img_A = augmented[3]
            offset = augmented[4]
            #===============================================================================
            # augmented, disparity = co_transform([left_img, right_img, n_right_img, offset], disparity)
            # left_img = augmented[0]
            # right_img = augmented[1]
            # n_right_img = augmented[2]
            # offset = augmented[3]
            
            
            # right_img.flags.writeable = True
            
            
            # if np.random.binomial(1,0.2):
            #   sx = int(np.random.uniform(35,100))
            #   sy = int(np.random.uniform(25,75))
            #   cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
            #   cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
            #   right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]
            #   n_right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(n_right_img,0),0)[np.newaxis,np.newaxis]

            # to tensor, normalize
            # offset = np.ascontiguousarray(offset, dtype=np.float32)
            disparity = np.ascontiguousarray(disparity, dtype=np.float32)
            processed = get_transform()
            n_left_img = processed(n_left_img)
            right_img = processed(right_img)

            return {"left": n_left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "left_img_A": left_img_A,
                    "left_img_Ap": left_img_Ap,
                    "offset": offset
                    }
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(n_left_img)
            right_img = processed(right_img)

            # pad to size 640x480
            top_pad = 480 - h
            right_pad = 640 - w
            assert top_pad >= 0 and right_pad >= 0
            
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                # print(disparity.shape)
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
                

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]
                        }
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
