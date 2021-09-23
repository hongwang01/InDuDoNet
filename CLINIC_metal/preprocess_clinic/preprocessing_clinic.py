# Given clinical Xma, generate data,including: XLI, M, Sma, SLI, Tr for infering InDuDoNet
import nibabel
import numpy as np
import os
from scipy.interpolate import interp1d
from .utils import get_config
from .build_gemotry import initialization, imaging_geo
import PIL
from PIL import Image
config = get_config('CLINIC_metal/preprocess_clinic/dataset_py_640geo.yaml')
CTpara = config['CTpara']  # CT imaging parameters
mask_thre = 2500 /1000 * 0.192 + 0.192  # taking 2500HU as a thresholding to segment the metal region
param = initialization()
ray_trafo, FBPOper = imaging_geo(param)  # CT imaging geometry, ray_trafo is fp, FBPoper is fbp
allXma = []
allXLI = []
allM = []
allSma = []
allSLI = []
allTr = []
allaffine = []
allfilename=[]
# process and save all the to-the-tested volumes
def clinic_input_data(test_path):
    for file_name in os.listdir(test_path):
        file_path = test_path+'/'+file_name
        img = nibabel.load(file_path)
        imag = img.get_fdata()  # imag with pixel as HU unit
        affine = img.affine
        allaffine.append(affine)
        num_s = imag.shape[2]
        M  = np.zeros((CTpara['imPixNum'], CTpara['imPixNum'], num_s), dtype='float32')
        Xma = np.zeros_like(M)
        XLI = np.zeros_like(M)
        Tr = np.zeros((CTpara['sinogram_size_x'], CTpara['sinogram_size_y'], num_s), dtype='float32')
        Sma = np.zeros_like(Tr)
        SLI =  np.zeros_like(Tr)
        for i in range(num_s):
            image = np.array(Image.fromarray(imag[:,:,i]).resize((CTpara['imPixNum'], CTpara['imPixNum']), PIL.Image.BILINEAR))
            image[image < -1000] = -1000
            image = image / 1000 * 0.192 + 0.192
            Xma[...,i] = image
            [rowindex, colindex] = np.where(image > mask_thre)
            M[rowindex, colindex, i] = 1
            Pmetal_kev = np.asarray(ray_trafo(M[:,:,i]))
            Tr[...,i] = Pmetal_kev > 0
            Sma[...,i] = np.asarray(ray_trafo(image))
            SLI[...,i] = interpolate_projection(Sma[...,i], Tr[...,i])
            XLI[...,i] = np.asarray(FBPOper(SLI[...,i]))
        allXma.append(Xma)
        allXLI.append(XLI)
        allM.append(M)
        allSma.append(Sma)
        allSLI.append(SLI)
        allTr.append(Tr)
        allfilename.append(file_name)
    return allXma, allXLI, allM, allSma, allSLI, allTr, allaffine, allfilename


def interpolate_projection(proj, metalTrace):
    # projection linear interpolation
    # Input:
    # proj:         uncorrected projection
    # metalTrace:   metal trace in projection domain (binary image)
    # Output:
    # Pinterp:      linear interpolation corrected projection
    Pinterp = proj.copy()
    for i in range(Pinterp.shape[0]):
        mslice = metalTrace[i]
        pslice = Pinterp[i]

        metalpos = np.nonzero(mslice==1)[0]
        nonmetalpos = np.nonzero(mslice==0)[0]
        pnonmetal = pslice[nonmetalpos]
        pslice[metalpos] = interp1d(nonmetalpos,pnonmetal)(metalpos)
        Pinterp[i] = pslice

    return Pinterp
