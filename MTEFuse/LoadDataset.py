import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from natsort import natsorted
import os
from torchvision import transforms
from utils import *

class MSRSDataset(Dataset):
    def __init__(self, vi_dir, ir_dir, transform=None):
        """
        vi_dir:可见光图像文件夹
        ir_dir:红外文件夹路径
        """
        self.vi_dir = vi_dir
        self.ir_dir = ir_dir
        self.vi_dirs = natsorted(os.listdir(vi_dir))
        self.ir_dirs = natsorted(os.listdir(ir_dir))
        self.transform_vi = transforms.Compose([transforms.ToTensor(),
                                             ])

        self.transform_ir = transforms.Compose([transforms.ToTensor(),
                                             ])

    def __len__(self):
        return len(self.vi_dirs)

    def __getitem__(self, idx):
        vi_dir = os.path.join(self.vi_dir, self.vi_dirs[idx])
        ir_dir = os.path.join(self.ir_dir, self.ir_dirs[idx])
        vi_img = Image.open(vi_dir)
        ir_img = Image.open(ir_dir)
        W1, H1 = vi_img.size
        W2, H2 = ir_img.size
        assert (W1 == W2) and (H1 == H2), "可见光与红外图像的尺寸不匹配"
        # vi_img = vi_img.resize((new_W, new_H)).convert("RGB")
        # ir_img = ir_img.resize((new_W, new_H)).convert("L")
        vi_img = vi_img.convert("RGB")
        ir_img = ir_img.convert("L")
        vi_img_y, _, _ = rgb_to_ycrcb(self.transform_vi(vi_img))
        ir_img = self.transform_ir(ir_img)

        return vi_img_y, ir_img

if __name__ == '__main__':
    # 创建数据集对象
    dataset = MSRSDataset(vi_dir=r'/home/demo/MSRS/vi',
                          ir_dir=r'/home/demo/MSRS/ir')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for vi, ir in dataloader:
        print(vi, ir)  # 每个批次包含 RGB 和深度图像
