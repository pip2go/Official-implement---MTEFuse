import kornia
import torch
import torch.nn as nn
import cv2
import numpy as np
# import pytorch_msssim
import torch.nn.functional as F
from torchvision import models
import torch.fft as fft
from torch.distributions import Normal
from kornia.filters import SpatialGradient
from utils import *
from torchmetrics import StructuralSimilarityIndexMeasure


# class Fusionloss(nn.Module):
#     def __init__(self):
#         super(Fusionloss, self).__init__()
#         self.sobelconv = Sobelxy()
#
#     def forward(self, image_vis, image_ir, generate_img):
#         image_y = image_vis[:, :1, :, :]
#         x_in_max = torch.max(image_y, image_ir)
#         loss_in = F.l1_loss(x_in_max, generate_img)
#         y_grad = self.sobelconv(image_y)
#         ir_grad = self.sobelconv(image_ir)
#         generate_img_grad = self.sobelconv(generate_img)
#         x_grad_joint = torch.max(y_grad, ir_grad)
#         loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
#         loss_total = loss_in + 10 * loss_grad
#         return loss_total, loss_in, loss_grad

def mse(img1, img2, window_size=9):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2

    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)

    res = F.fold(res, output_size=(height, width), kernel_size=(1, 1))
    return res

def std(img,  window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1,1, 11, 11]
    return window

def final_mse1(img_ir, img_vis, img_fuse, mask=None):
    mse_ir = mse(img_ir, img_fuse)
    mse_vi = mse(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)
    # std_ir = sum(img_ir)
    # std_vi = sum(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    m = torch.mean(img_ir)
    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    # map2 = torch.where((std_ir - std_vi) >= 0, zero, one)
    map_ir=torch.where(map1+mask>0, one, zero)
    map_vi= 1 - map_ir

    res = map_ir * mse_ir + map_vi * mse_vi
    # res = res * w_vi
    return res.mean()

class REG(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(REG, self).__init__()

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1*img2)/torch.sqrt(torch.sum(img1*img1)*torch.sum(img2*img2))
        return r

    def forward(self, a, b, c):
        return self.corr2(a, c) + self.corr2(b, c)

def corr_loss(image_ir, img_vis, img_fusion, eps=1e-6):
    reg = REG()
    corr = reg(image_ir, img_vis, img_fusion)
    corr_loss = 1./(corr + eps)
    return corr_loss

def Fusion_loss_PS(ir, vi, fu, mask, weights=[1, 10, 10], device=None):
    # grad_ir =  KF.spatial_gradient(IR, order=2).abs().sum(dim=[1,2])
    # grad_vi = KF.spatial_gradient(VI_Y, order=2).abs().sum(dim=[1,2])
    # grad_fus = KF.spatial_gradient(fu, order=2).abs().sum(dim=[1,2])
    # grad_joint = torch.max(grad_ir, grad_vi)
    sobelconv = Sobelxy(device)
    vi_grad_x, vi_grad_y = sobelconv(vi)
    ir_grad_x, ir_grad_y = sobelconv(ir)
    fu_grad_x, fu_grad_y = sobelconv(fu)
    grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
    grad_joint_y = torch.max(vi_grad_y, ir_grad_y)

    loss_grad = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)
    ## 梯度损失
    # loss_grad = F.l1_loss(grad_fus, grad_joint)
    ## SSIM损失
    loss_ssim = corr_loss(ir, vi, fu)
    ## 强度损失
    loss_intensity = final_mse1(ir, vi, fu, mask) + 0 * F.l1_loss(fu, torch.max(ir, vi))
    loss_total = weights[0] * loss_ssim + weights[1] * loss_grad + weights[2] * loss_intensity
    return loss_total, loss_intensity, loss_grad, loss_ssim

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        # save_images(image_vis)
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        loss_total = loss_in + 10*loss_grad
        return loss_total, loss_in, loss_grad


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]

        # prewitt_edge
        # kernelx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        # kernely = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

class Laplacian(nn.Module):
    def __init__(self):
        super(Laplacian, self).__init__()
        kernel = [[-1, -1, -1],
                  [-1,  8, -1],
                  [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def forward(self, x):
        laplacian = F.conv2d(x, self.weight, padding=1)
        return torch.abs(laplacian)

def charbonnier_loss(input, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((input - target)**2 + epsilon**2))

def l1_loss(pred, target):
    """
        平均绝对误差
    """
    l1Loss = nn.L1Loss(pred, target)
    return l1Loss


def l2_loss(pred, target):
    """
        均方误差
    """
    criterion = nn.MSELoss()
    return criterion(pred, target)

def gradient_loss(pred, target):
    """
        梯度损失
    """
    l1Loss = nn.L1Loss()
    Gradient_loss = l1Loss(SpatialGradient()(pred), SpatialGradient()(target))
    return Gradient_loss

# def gradient_loss(pred, vis, ir):
#     """
#     pred: 融合结果
#     vis: 可见光图像
#     ir: 红外图像
#     """
#     # 对可见光使用严格梯度约束
#     grad_vis_loss = F.l1_loss(SpatialGradient()(pred), SpatialGradient()(vis))
#
#     # 对红外使用松弛约束（仅强边缘区域）
#     grad_ir = SpatialGradient()(ir)
#     grad_ir_mask = (grad_ir > 0.1).float()  # 阈值过滤弱梯度
#     grad_ir_loss = F.l1_loss(pred * grad_ir_mask, ir * grad_ir_mask)
#
#     return grad_vis_loss + 2 * grad_ir_loss  # 降低红外权重

def ssim_loss(pred, target):
    """
        结构相似性损失
    """
    # ssim_loss = pytorch_msssim.msssim
    # ssim = StructuralSimilarityIndexMeasure()
    ssim = kornia.losses.SSIMLoss(11, reduction='mean')
    return 1 - ssim(pred, target)

def perceptual_loss(pred, target):
    """
        感知损失
    """
    vgg = models.vgg16(pretrained=True).features[:16].eval()
    pred_features = vgg(pred)
    target_features = vgg(target)
    return F.mse_loss(pred_features, target_features)

def adversarial_loss(discriminator, pred):
    """
        对抗损失
    """
    real_labels = torch.ones(pred.size(0), 1)
    return torch.nn.BCEWithLogitsLoss()(discriminator(pred), real_labels)

def deep_feature_loss(pred, target, model):
    """
        深度特征损失
    """
    pred_features = model(pred)
    target_features = model(target)
    return F.mse_loss(pred_features, target_features)

def edge_preserving_loss(pred, target):
    """
        边缘保留损失
    """
    device = pred.device
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(device)
    grad_loss = 0
    for i in range(pred.size(0)):
        pred_grad_x = F.conv2d(pred[i:i+1,:,:,:], sobel_x, padding=1)
        target_grad_x = F.conv2d(target[i:i+1,:,:,:], sobel_x, padding=1)

        pred_grad_y = F.conv2d(pred[i:i+1,:,:,:], sobel_y, padding=1)
        target_grad_y = F.conv2d(target[i:i+1,:,:,:], sobel_y, padding=1)

        grad_loss += F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)
    return grad_loss / pred.size(0)

def content_loss(pred, target, model):
    """
        内容损失
    """
    pred_features = model(pred)
    target_features = model(target)
    return F.mse_loss(pred_features, target_features)

def multi_scale_loss(pred, target, model, scales=[1, 2, 4]):
    """
        多尺度损失
    """
    total_loss = 0
    for scale in scales:
        pred_resized = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
        target_resized = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
        pred_features = model(pred_resized)
        target_features = model(target_resized)
        total_loss += F.mse_loss(pred_features, target_features)
    return total_loss


def hybrid_loss(pred, target, discriminator, model, lambda_mse=1, lambda_ssim=1, lambda_adv=1):
    """
        混合损失
    """
    mse = F.mse_loss(pred, target)
    ssim = ssim_loss(pred, target)
    adv = adversarial_loss(discriminator, pred)
    return lambda_mse * mse + lambda_ssim * ssim + lambda_adv * adv


class FrequencyDomainPhaseConsistencyLoss(nn.Module):
    def __init__(self):
        """
            频域相位一致性损失（Frequency Domain Phase Consistency Loss）
        """
        super(FrequencyDomainPhaseConsistencyLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        # 将图片转换到频域
        fft_x = fft.fft2(x)
        fft_y = fft.fft2(y)

        # 提取相位信息
        phase_x = torch.angle(fft_x)
        phase_y = torch.angle(fft_y)

        # 计算相位一致性损失
        phase_loss = self.criterion(phase_x, phase_y)
        return phase_loss

class GANLoss(nn.Module):
    """
        对抗性损失（Adversarial Loss）
    """
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def forward(self, prediction, target_is_real):
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)

class StructureAwareMultiScaleLoss(nn.Module):
    def __init__(self, num_scales=3, scale_weights=None):
        """
            结构感知多尺度损失（Structure-Aware Multi-Scale Loss）
        """
        super(StructureAwareMultiScaleLoss, self).__init__()
        self.num_scales = num_scales
        if scale_weights is None:
            self.scale_weights = [1.0] * num_scales
        else:
            self.scale_weights = scale_weights
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        total_loss = 0.0
        for scale in range(self.num_scales):
            # 下采样
            scale_factor = 1.0 / (2 ** scale)
            x_scaled = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            y_scaled = F.interpolate(y, scale_factor=scale_factor, mode='bilinear', align_corners=False)

            # 计算结构感知损失
            loss = self.criterion(x_scaled, y_scaled)
            total_loss += self.scale_weights[scale] * loss

        return total_loss

class MutualInformationLoss(nn.Module):
    def __init__(self, num_bins=256, sigma=1, normalized=True):
        """
            互信息最大化损失（Mutual Information Maximization Loss）
        """
        super(MutualInformationLoss, self).__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.normalized = normalized

    def _compute_histogram(self, x, y):

        # 将输入归一化到[0, 1]范围
        device = x.device

        x.to(device)
        y.to(device)

        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8)
        y = (y - torch.min(y)) / (torch.max(y) - torch.min(y) + 1e-8)

        # 将输入量化到num_bins个bin中
        x_bin = torch.floor(x * self.num_bins).long()
        y_bin = torch.floor(y * self.num_bins).long()

        # 计算联合直方图和边缘直方图
        bin_size = x_bin.size(0) * x_bin.size(1)
        hist_xy = torch.zeros(self.num_bins, self.num_bins).to(device)
        hist_x = torch.zeros(self.num_bins).to(device)
        hist_y = torch.zeros(self.num_bins).to(device)

        for i in range(bin_size):
            x_idx = x_bin[i // x_bin.size(1), i % x_bin.size(1)].to(device)
            y_idx = y_bin[i // y_bin.size(1), i % y_bin.size(1)].to(device)

            # 确保索引在合法范围内
            x_idx = torch.clamp(x_idx, 0, hist_x.size(0) - 1)
            y_idx = torch.clamp(y_idx, 0, hist_y.size(0) - 1)

            hist_xy[x_idx, y_idx] += 1
            hist_x[x_idx] += 1
            hist_y[y_idx] += 1

        # 计算概率分布
        p_xy = hist_xy / bin_size
        p_x = hist_x / bin_size
        p_y = hist_y / bin_size

        # 计算互信息
        mi = 0.0
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * torch.log(p_xy[i, j] / (p_x[i] * p_y[j]))

        if self.normalized:
            h_x = -torch.sum(p_x * torch.log(p_x + 1e-8))
            h_y = -torch.sum(p_y * torch.log(p_y + 1e-8))
            mi = mi / (torch.sqrt(h_x * h_y) + 1e-8)

        return mi

    def forward(self, x, y):
        # 计算互信息损失
        mi = self._compute_histogram(x, y)
        return -mi

class SemanticAwarePerceptualLoss(nn.Module):
    """
        语义感知损失（Semantic-Aware Perceptual Loss）
    """
    def __init__(self, semantic_weight=1.0):
        super(SemanticAwarePerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()
        self.semantic_weight = semantic_weight

    def forward(self, x, y, semantic_x, semantic_y):
        # 计算感知损失
        features_x = self.vgg(x)
        features_y = self.vgg(y)
        perceptual_loss = self.criterion(features_x, features_y)

        # 计算语义损失
        semantic_loss = self.criterion(semantic_x, semantic_y)

        # 综合损失
        total_loss = perceptual_loss + self.semantic_weight * semantic_loss
        return total_loss

class SparseRepresentationLoss(nn.Module):
    """
        稀疏表示与字典学习损失（Sparse Representation and Dictionary Learning Loss）
    """
    def __init__(self, dictionary_size=100, sparsity_weight=0.1):
        super(SparseRepresentationLoss, self).__init__()
        self.dictionary_size = dictionary_size
        self.sparsity_weight = sparsity_weight
        self.criterion = nn.MSELoss()

    def forward(self, x, dictionary):
        # 计算稀疏编码
        coefficients = torch.zeros(x.size(0), self.dictionary_size).to(x.device)
        for i in range(x.size(0)):
            # 使用OMP算法计算稀疏系数
            residual = x[i]
            selected_indices = []
            for _ in range(self.sparsity):
                inner_products = torch.matmul(residual, dictionary.T)
                max_idx = torch.argmax(torch.abs(inner_products))
                selected_indices.append(max_idx)
                residual -= torch.matmul(inner_products[max_idx], dictionary[max_idx])
            coefficients[i, selected_indices] = inner_products[selected_indices]

        # 计算重建损失
        x_reconstructed = torch.matmul(coefficients, dictionary)
        reconstruction_loss = self.criterion(x_reconstructed, x)

        # 计算稀疏性损失
        sparsity_loss = torch.norm(coefficients, p=1) / x.size(0)

        # 综合损失
        total_loss = reconstruction_loss + self.sparsity_weight * sparsity_loss
        return total_loss

class ContrastiveLoss(nn.Module):
    """
        对比学习损失（Contrastive Loss）
    """
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2, labels):
        # 计算相似性矩阵
        similarity_matrix = torch.mm(features1, features2.T) / self.temperature
        similarity_matrix = torch.exp(similarity_matrix)

        # 计算归一化
        mask = torch.eye(features1.size(0)).bool().to(features1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)

        # 计算正样本相似性
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        positive_mask = positive_mask.masked_fill(mask, 0)
        positive_sim = (similarity_matrix * positive_mask).sum(dim=1)

        # 计算负样本相似性
        negative_sim = similarity_matrix.sum(dim=1) - positive_sim

        # 计算损失
        loss = -torch.log(positive_sim / (positive_sim + negative_sim)).mean()
        return loss

class PhysicalConstraintsLoss(nn.Module):
    """
        能量函数引导的物理约束损失（Energy Function Guided Physical Constraints Loss）
    """
    def __init__(self, energy_weight=1.0):
        super(PhysicalConstraintsLoss, self).__init__()
        self.energy_weight = energy_weight
        self.criterion = nn.MSELoss()

    def forward(self, x, y, physical_constraints):
        # 计算感知损失
        features_x = self.vgg(x)
        features_y = self.vgg(y)
        perceptual_loss = self.criterion(features_x, features_y)

        # 计算物理约束损失
        energy_loss = self.criterion(physical_constraints(x), physical_constraints(y))

        # 综合损失
        total_loss = perceptual_loss + self.energy_weight * energy_loss
        return total_loss

class DynamicWeightAdaptiveLoss(nn.Module):
    """
        动态权重自适应损失（Dynamic Weight Adaptive Loss）
    """
    def __init__(self, initial_weights=[1.0, 1.0], learning_rate=0.01):
        super(DynamicWeightAdaptiveLoss, self).__init__()
        self.weights = torch.tensor(initial_weights, requires_grad=True)
        self.learning_rate = learning_rate

    def forward(self, loss1, loss2):
        # 动态调整权重
        with torch.no_grad():
            self.weights[0] += self.learning_rate * (loss1 - loss2)
            self.weights[1] += self.learning_rate * (loss2 - loss1)
            self.weights = F.softmax(self.weights, dim=0)

        # 计算加权损失
        total_loss = self.weights[0] * loss1 + self.weights[1] * loss2
        return total_loss

class HumanVisualSystemLoss(nn.Module):
    """
        人眼视觉系统启发损失（Human Visual System Inspired Loss）
    """
    def __init__(self, contrast_weight=1.0, color_weight=1.0):
        super(HumanVisualSystemLoss, self).__init__()
        self.contrast_weight = contrast_weight
        self.color_weight = color_weight
        self.criterion = nn.MSELoss()

    def _compute_contrast(self, x):
        # 计算对比度
        mean_x = torch.mean(x, dim=(2,3), keepdim=True)
        contrast = torch.mean((x - mean_x).pow(2), dim=(2,3))
        return contrast

    def _compute_color(self, x):
        # 计算颜色一致性
        color_mean = torch.mean(x, dim=(0,2,3), keepdim=True)
        color_loss = torch.mean((x - color_mean).pow(2))
        return color_loss

    def forward(self, x, y):
        # 计算对比度损失
        contrast_loss = self._compute_contrast(x) + self._compute_contrast(y)
        contrast_loss = self.criterion(contrast_loss, torch.zeros_like(contrast_loss))

        # 计算颜色损失
        color_loss = self._compute_color(x) + self._compute_color(y)
        color_loss = self.criterion(color_loss, torch.zeros_like(color_loss))

        # 综合损失
        total_loss = self.contrast_weight * contrast_loss + self.color_weight * color_loss
        return total_loss

def roberts_edge(image):
    """
        基于交叉差分，检测对角线边缘
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)  # 45度方向
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)  # 135度方向
    edge_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    edge_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
    edge = np.sqrt(edge_x**2 + edge_y**2).astype(np.uint8)
    return edge

def sobel_edge(image):
    """
        强调水平和垂直边缘
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edge_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向
    edge_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向
    edge = np.sqrt(edge_x**2 + edge_y**2).astype(np.uint8)
    return edge

def prewitt_edge(image):
    """
        类似 Sobel，但核不同
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    edge_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    edge_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
    edge = np.sqrt(edge_x**2 + edge_y**2).astype(np.uint8)
    return edge

def kirsch_edge(image):
    """
        使用 8 个方向模板，取最大值
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    kernels = [
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),  # 北
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32),  # 东北
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),  # 东
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32),  # 东南
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),  # 南
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),  # 西南
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),  # 西
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32)   # 西北
    ]
    edges = [cv2.filter2D(gray, cv2.CV_64F, kernel) for kernel in kernels]
    edge = np.max(np.abs(edges), axis=0).astype(np.uint8)
    return edge

def log_edge(image, sigma=1.0):
    """
        先高斯平滑，再拉普拉斯边缘检测
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # 生成 LoG 核
    kernel_size = int(6 * sigma) + 1
    kernel = cv2.getGaussianKernel(kernel_size, sigma) @ cv2.getGaussianKernel(kernel_size, sigma).T
    kernel = cv2.Laplacian(kernel, cv2.CV_64F)
    edge = cv2.filter2D(gray, cv2.CV_64F, kernel)
    edge = np.abs(edge).astype(np.uint8)
    return edge