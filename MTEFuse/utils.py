import datetime
import logging
import os
import platform
import random
import subprocess
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from natsort import natsorted
from scipy.misc import imsave, imresize
from torchvision import transforms
from PIL import Image, ImageOps
from torchvision.ops import DeformConv2d
import cv2 as cv
# from self1.plot import colorstr

def save_checkpoint(model, path):
    torch.save({
        'MTEFuse': model.state_dict(),
    }, path)


def is_pil_image(img):
    return isinstance(img, Image.Image)


def To_pil_image(img):
    return transforms.ToPILImage(img)

class DeformAlign(nn.Module):
    """可变形特征对齐模块"""
    def __init__(self, channels):
        super().__init__()
        self.offset_conv = nn.Conv2d(channels, 18, 3, padding=1)
        self.conv = DeformConv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.conv(x, offset)

class UpReshape(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=True)

    def forward(self, x1, x2):
        x2 = self.upsample(x2)

        # 获取目标尺寸
        _, _, H1, W1 = x1.shape
        _, _, H2, W2 = x2.shape

        # 动态尺寸调整（同时处理尺寸过大和过小的情况）
        if H2 != H1 or W2 != W1:
            # 高度调整
            if H2 > H1:
                # 中心裁剪高度
                diff = H2 - H1
                x2 = x2[:, :, diff // 2: diff // 2 + H1, :]
            else:
                # 反射填充高度
                pad_h = H1 - H2
                top = pad_h // 2
                bottom = pad_h - top  # 包含余数
                x2 = F.pad(x2, (0, 0, top, bottom), mode='replicate')

            # 宽度调整
            if W2 > W1:
                # 中心裁剪宽度
                diff = W2 - W1
                x2 = x2[:, :, :, diff // 2: diff // 2 + W1]
            else:
                # 反射填充宽度
                pad_w = W1 - W2
                left = pad_w // 2
                right = pad_w - left  # 包含余数
                x2 = F.pad(x2, (left, right, 0, 0), mode='replicate')

        return x2


class DownReshape(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.downsample = nn.MaxPool2d(scale_factor)

    def forward(self, x1, x2):
        x2 = self.downsample(x2)

        # 获取目标尺寸
        _, _, H1, W1 = x1.shape
        _, _, H2, W2 = x2.shape

        # 动态尺寸调整（同时处理尺寸过大和过小的情况）
        if H2 != H1 or W2 != W1:
            # 高度调整
            if H2 > H1:
                # 中心裁剪高度
                diff = H2 - H1
                x2 = x2[:, :, diff // 2: diff // 2 + H1, :]
            else:
                # 反射填充高度
                pad_h = H1 - H2
                top = pad_h // 2
                bottom = pad_h - top  # 包含余数
                x2 = F.pad(x2, (0, 0, top, bottom), mode='replicate')

            # 宽度调整
            if W2 > W1:
                # 中心裁剪宽度
                diff = W2 - W1
                x2 = x2[:, :, :, diff // 2: diff // 2 + W1]
            else:
                # 反射填充宽度
                pad_w = W1 - W2
                left = pad_w // 2
                right = pad_w - left  # 包含余数
                x2 = F.pad(x2, (left, right, 0, 0), mode='replicate')

        return x2

# def UpReshape(x1, x2):
#
#         # 获取尺寸
#         _, _, H1, W1 = x1.shape
#         _, _, H2, W2 = x2.shape
#
#         # 如果尺寸不同，直接插值成目标尺寸，避免裁剪和pad误差
#         if (H1 != H2) or (W1 != W2):
#             x2 = F.interpolate(x2, size=(H1, W1), mode='bilinear', align_corners=False)
#
#         return x2
#
# def DownReshape(x1, x2):
#
#         # 获取尺寸
#         _, _, H1, W1 = x1.shape
#         _, _, H2, W2 = x2.shape
#
#         # 如果尺寸不同，直接插值成目标尺寸，避免裁剪和pad误差
#         if (H1 != H2) or (W1 != W2):
#             x2 = F.interpolate(x2, size=(H1, W1), mode='bilinear', align_corners=False)
#
#         return x2


def load_image(path, mode='L', array=True):
    assert mode == 'L' or mode == 'RGB' or mode == 'CMYK' or 'YCbCr' or 'RGB_y', f"Unsupported mode: {mode}"
    with Image.open(path) as img:
        if mode != "RGB_y":
            image = img.convert(mode)
            if array:
                image = np.array(image)/255
        elif mode == "RGB_y":
            transform = transforms.ToTensor()
            tensor_img = transform(img)
            vi_Y, vi_Cb, vi_Cr = rgb_to_ycrcb(tensor_img.unsqueeze(0))
            if array:
                image = np.array(vi_Y)
    return image


def load_test_images(paths, height=None, width=None, mode='RGB'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = load_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy() * 255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def save_images(path, fuse_norm):
    # 确保转换为 CPU 张量并分离计算图
    # if fuse_norm.min() < 0 or fuse_norm.max() > 1:
    #     fuse_norm = torch.sigmoid(fuse_norm)
    img_fuse = np.round(np.squeeze((fuse_norm * 255).detach().cpu().numpy()))
    img = np.clip(img_fuse, 0, 255).astype(np.uint8)
    imsave(path, img)
    return img


def tensor_save_rgb(tensor, filename, normalize=False):
    img = tensor.detach().cpu()
    if normalize:
        img *= 255.0
    img = torch.clamp(img, 0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    Image.fromarray(img).save(filename)  # 保存转换后的RGB图像


def rgb_to_ycrcb(tensor, rgb=True):
    if tensor.size(0) != 3:
        raise ValueError("输入张量必须有 3 个通道（RGB）")
    # 提取 R, G, B 通道
    if rgb:
        r, g, b = tensor[0:1, :, :], tensor[1:2, :, :], tensor[2:3, :, :]
    else:
        b, g, r = tensor[0:1, :, :], tensor[1:2, :, :], tensor[:, 2:3, :, :]
    # 计算 Y 分量
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 0.5
    cb = (b - y) * 0.564 + 0.5
    y = torch.clamp(y, 0.0, 1.0)
    cr = torch.clamp(cr, 0.0, 1.0)
    cb = torch.clamp(cb, 0.0, 1.0)
    return y, cr, cb


def ycrcb_to_rgb(tensor, rgb=True):
    if tensor.size(1) != 3:
        raise ValueError("输入张量必须有 3 个通道（YCrCb）")
    y, cr, cb = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :]
    cr -= 0.5
    cb -= 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    if rgb:
        rgb = torch.cat([r, g, b], dim=1)
    else:
        rgb = torch.cat([b, g, r], dim=1)
    rgb = rgb.clamp(0.0, 1.0)
    return rgb


def append_image(directory):
    images = []
    dir = natsorted(os.listdir(directory))
    for file in dir:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            images.append(os.path.join(directory, file))
    return images


def tensor_load_rgb(imgname, size=None, scale=None, keep_asp=False, normalize=False):
    try:
        img = Image.open(imgname).convert('RGB')
    except Exception as e:
        raise IOError(f"无法加载图片文件{imgname}:{str(e)}")
    if size is not None:
        if keep_asp:
            new_size = int(size * 1.0 * img.size[1] / img.size[0])
            img = img.resize((size, new_size), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)
    if scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    if normalize:  # 图片归一化
        img /= 255.0
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()  # 将array转换为Pytorch的tensor
    return img


def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    for h in logging.root.handlers:
        logging.root.removeHandler(h)  # remove all handlers associated with the root logger object
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


def split_image(image, patch_size=256):
    w, h = image.size
    # 计算右侧和底部填充量
    pad_w = (patch_size - w % patch_size) if w % patch_size != 0 else 0
    pad_h = (patch_size - h % patch_size) if h % patch_size != 0 else 0
    # 使用灰色填充右和下侧
    padded = ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=0)
    # 计算分块数量
    num_w = (w + pad_w) // patch_size
    num_h = (h + pad_h) // patch_size
    # 分块
    patches = []
    positions = []
    for y in range(num_h):
        for x in range(num_w):
            left = x * patch_size
            upper = y * patch_size
            right = left + patch_size
            lower = upper + patch_size
            patch = padded.crop((left, upper, right, lower))
            patches.append(patch)
            positions.append((left, upper))
    return patches, positions, (w, h, pad_w, pad_h)


def merge_patches(patches, positions, meta):
    w_orig, h_orig, pad_w, pad_h = meta
    # 创建填充后的底图
    merged = Image.new("L", (w_orig + pad_w, h_orig + pad_h))
    for patch, (x, y) in zip(patches, positions):
        merged.paste(patch, (x, y, x + patch.width, y + patch.height))
    # 裁剪回原始尺寸
    merged = merged.crop((0, 0, w_orig, h_orig))
    return merged


LOGGER = set_logging(__name__)  # define globally (used in train.py, val.py, detect.py, etc.)
def colorstr(string):
    pass


def print_args(name, opt):
    LOGGER.info(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))


def date_modified(path=__file__):
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):  # path must be a directory
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''


def select_device(device='', batch_size=0):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'MTEFuse 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


def get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        min_lr: float = 0.0,
        last_epoch: int = -1
):
    """
    Args:
        optimizer: 优化器对象
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        num_cycles: 余弦周期数（默认0.5即半个周期）
        min_lr: 最小学习率
        last_epoch: 恢复训练时的起始epoch
    """

    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))

        # 余弦退火阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))) * (1 - min_lr) + min_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)