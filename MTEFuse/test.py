import sys
from utils import *
from tqdm import tqdm
from model import MTEFuse_model
import warnings

warnings.filterwarnings("ignore")
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['MTEFuse'])

def test(data_path, checkpoint_path, save_path):
    eps = torch.finfo(torch.float32).eps
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    vi_dir = os.path.join(data_path, "vi")
    ir_dir = os.path.join(data_path, "ir")
    vi_files = [os.path.join(vi_dir, i) for i in os.listdir(vi_dir)]
    ir_files = [os.path.join(ir_dir, i) for i in os.listdir(ir_dir)]
    # vi_files = [os.path.join(vi_dir, i) for i in os.listdir(vi_dir)][]
    # ir_files = [os.path.join(ir_dir, i) for i in os.listdir(ir_dir)][]
    # vi_files = [vi_files]
    # ir_files = [ir_files]
    os.makedirs(save_path, exist_ok=True)
    model = MTEFuse_model().to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    with torch.no_grad():
        for vi, ir in tqdm(zip(vi_files, ir_files), total=len(vi_files), desc="MTEFuse testing"):

            img_vi = load_image(os.path.join(vi_dir, vi), 'L')
            img_ir = load_image(os.path.join(ir_dir, ir), 'L')
            img_vi = torch.FloatTensor(img_vi).to(device).unsqueeze(0).unsqueeze(0)
            img_ir = torch.FloatTensor(img_ir).to(device).unsqueeze(0).unsqueeze(0)

            pre_img11 = model.encoder(img_vi)
            pre_img12 = model.encoder(img_ir)
            pre_img21 = model.fuses1(torch.concat((pre_img11[0], pre_img12[0]), dim=1))
            pre_img22 = model.fuses2(torch.concat((pre_img11[1], pre_img12[1]), dim=1))
            pre_img23 = model.fuses3(torch.concat((pre_img11[2], pre_img12[2]), dim=1))
            pre_img24 = model.fuses4(torch.concat((pre_img11[3], pre_img12[3]), dim=1))
            img_fuse = model.decoder([pre_img21, pre_img22, pre_img23, pre_img24])
            fuse_norm = (img_fuse - torch.min(img_fuse))/(torch.max(img_fuse) - torch.min(img_fuse)+eps)
            img_fuse = save_images(save_path+f"{os.path.basename(vi)}", fuse_norm)


if __name__ == '__main__':
    checkpoint_path=r"C:\Users\Administrator\Desktop\MTEFuse\checkpoint\MTEFuse.pt"
    # for i in range(len(model_file)):
    #     model_path = os.path.join(model_file, os.listdir(model_file)[i])
    #     print(model_path)
    #     test(num_mec=13,
    #          modelname="MTEFuse_model",
    #          data_path=r"C:/Users/Administrator/Desktop/bilibili_PILI_code/image_processing_DL/pycharm_Image_fusion/Evaluate/datasets/TNO",
    #          model_path=model_path,
    #          save_path="outputs/")

    test(data_path=r"C:\Users\Administrator\Desktop\MTEFuse\TNO",
         checkpoint_path=checkpoint_path,
         save_path="outputs/")
# data_path="/home/pycharm_Image_fusion/Evaluate/datasets/TNO",
# model_path="/home/demo/densefuse222/checkpoint/checkpoint_MTEFuse_epoch-05-12-07-38.pt",