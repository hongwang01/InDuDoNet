import os.path
import os
import os.path
import argparse
import numpy as np
import torch
from CLINIC_metal.preprocess_clinic.preprocessing_clinic import clinic_input_data
from network.indudonet import InDuDoNet
import nibabel
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description="YU_Test")
parser.add_argument("--model_dir", type=str, default="models", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="CLINIC_metal/test/", help='path to training data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--save_path", type=str, default="results/CLINIC_metal/", help='path to training data')
parser.add_argument('--num_channel', type=int, default=32, help='the number of dual channels')
parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='the number of total iterative stages')
parser.add_argument('--eta1', type=float, default=1, help='initialization for stepsize eta1')
parser.add_argument('--eta2', type=float, default=5, help='initialization for stepsize eta2')
parser.add_argument('--alpha', type=float, default=0.5, help='initialization for weight factor')
opt = parser.parse_args()
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")
Pred_nii = opt.save_path +'/X_mar/'
mkdir(Pred_nii)

def image_get_minmax():
    return 0.0, 1.0
def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max) 
    data = (data - data_min) / (data_max - data_min)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

def test_image(allXma, allXLI, allM, allSma, allSLI, allTr, vol_idx, slice_idx):
    Xma = allXma[vol_idx][...,slice_idx]
    XLI = allXLI[vol_idx][...,slice_idx]
    M = allM[vol_idx][...,slice_idx]
    Sma = allSma[vol_idx][...,slice_idx]
    SLI = allSLI[vol_idx][...,slice_idx]
    Tr = allTr[vol_idx][...,slice_idx]
    Xma = normalize(Xma, image_get_minmax())  # *255
    XLI = normalize(XLI, image_get_minmax())
    Sma = normalize(Sma, proj_get_minmax())
    SLI = normalize(SLI, proj_get_minmax())
    Tr = 1-Tr.astype(np.float32)
    Tr = np.expand_dims(np.transpose(np.expand_dims(Tr, 2), (2, 0, 1)),0) # 1*1*h*w
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)),0)
    return torch.Tensor(Xma).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(Mask).cuda(), \
       torch.Tensor(Sma).cuda(), torch.Tensor(SLI).cuda(), torch.Tensor(Tr).cuda()

def main():
    # Build model
    print('Loading model ...\n')
    net = InDuDoNet(opt).cuda()
    net.load_state_dict(torch.load(os.path.join(opt.model_dir)))
    net.eval()
    print('--------------load---------------all----------------nii-------------')
    allXma, allXLI, allM, allSma, allSLI, allTr, allaffine, allfilename = clinic_input_data(opt.data_path)
    print('--------------test---------------all----------------nii-------------')
    for vol_idx in range(len(allXma)):
        print('test %d th volume.......' % vol_idx)
        num_s = allXma[vol_idx].shape[2]
        pre_Xout = np.zeros_like(allXma[vol_idx])
        pre_name = allfilename[vol_idx]
        for slice_idx in range(num_s):
            Xma, XLI, M, Sma, SLI, Tr  = test_image(allXma, allXLI, allM, allSma, allSLI, allTr, vol_idx, slice_idx)
            with torch.no_grad():
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()
                ListX, ListS, ListYS= net(Xma, XLI, M, Sma, SLI, Tr)
            Xout= ListX[-1] / 255.0
            pre_Xout[..., slice_idx] = Xout.data.cpu().numpy().squeeze()
        nibabel.save(nibabel.Nifti1Image(pre_Xout, allaffine[vol_idx]), Pred_nii + pre_name)
if __name__ == "__main__":
    main()

