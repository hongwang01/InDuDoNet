# InDuDoNet: An Interpretable Dual Domain Network for CT Metal Artifact Reduction (MICCAI2021)
[Hong Wang](https://hongwang01.github.io/), Yuexiang Li, Haimiao Zhang, Jiawei Chen, Kai Ma, [Deyu Meng](http://gr.xjtu.edu.cn/web/dymeng), [Yefeng Zheng](https://sites.google.com/site/yefengzheng/)

[[PDF&&SupplementaryMaterial]](https://arxiv.org/pdf/2109.05298.pdf)

## Abstract
For the task of metal artifact reduction (MAR), although deep learning (DL)-based methods have achieved promising performances, most of them suffer from two problems: 1) the CT imaging geometry constraint is not fully embedded into the network during training, leaving room for further performance improvement; 2) the model interpretability is lack of sufficient consideration. Against these issues, we propose a novel interpretable dual domain network, termed as InDuDoNet, which combines the advantages of model-driven and data-driven methodologies. Specifically, we build a joint spatial and Radon domain reconstruction model and utilize the proximal gradient technique to design an iterative algorithm for solving it. The optimization algorithm only consists of simple computational operators, which facilitate us to correspondingly unfold iterative steps into network modules and thus improve the interpretability of the framework. Extensive experiments on synthesized and clinical data show the superiority of our InDuDoNet.

## Overview of InDuDoNet
<div  align="center"><img src="figs/net.png" height="100%" width="100%" alt=""/></div>

## Dependicies

Refer to **requirements.txt**. The following project links are needed for installing ODL and astra:

ODL: https://github.com/odlgroup/odl
Astra: https://github.com/astra-toolbox/astra-toolbox


## Folder Directory 
```
.
|-- train.py             
|-- test_deeplesion.py
|-- test_clinic.py
|-- results                   # reconstructed images 
|-- network                   # InDuDoNet
|-- deeplesion                # for train and test
|   |-- Dataset.py          
|   |-- __init__.py
|   |-- build_gemotry.py      # imaging paramter (FP/FBP)
|   |-- train                 # synthesized data for train
|   |-- test                  # synthesized data for test
|-- CLINIC_metal              # for clinical evaluation  
|   |-- preprocess_clinic     # processing CLINIC_metal
|   |-- test                  # clinical data for test
```
## Benchmark Dataset

**DeepLesion:** Download the [DeepLesion dataset](https://nihcc.app.box.com/v/DeepLesion) and synthesize the metal-corrupted ones with the [simulation protocol](https://github.com/liaohaofu/adn). Refer to [1][2][3].
Note that you can also refer to `bulid_geometory.py` and flexibly finish the data synthesis with `Python`.

**CLINIC-metal:** Download the clinical metal-corrupted [CLINIC-metal dataset](https://github.com/ICT-MIRACLE-lab/CTPelvic1K) with mutli-bone segmentation. In our experiments, we only adopt the testing set with 14 volumes for evaluation.
 
## Training
```
CUDA_VISIBLE_DEVICES=0 python train.py --data_path "deeplesion/train/" --batchnum 2
```
*For the demo, the hyper-parameter batchnum is set as 2. Please change it according to your own training set.*
## Testing

### For DeepLesion
```
CUDA_VISIBLE_DEVICES=0 python test_deeplesion.py --data_path "deeplesion/test/" --model_dir "models/" --save_path "results/deeplesion/" 
```
### For CLINIC-metal
```
CUDA_VISIBLE_DEVICES=0 python test_clinic.py --data_path "CLINIC_metal/test/" --model_dir "models/" --save_path "results/CLINIC_metal/"
```
## Model Verification
<div  align="center"><img src="figs/visualization.png" height="100%" width="100%" alt=""/></div>
## Experiments on Synthesized Data
<div  align="center"><img src="figs/syn.png" height="100%" width="100%" alt=""/></div>
## Experiments on CLINIC-metal
<div  align="center"><img src="figs/clinic.png" height="100%" width="100%" alt=""/></div>

## Metric
[PSNR/SSIM](https://github.com/hongwang01/RCDNet/tree/master/Performance_evaluation) 

## Acknowledgement
The authors would like to thank [Dr. Lequan Yu](https://yulequan.github.io/) for providing the code `bulid_geometory.py` released in this repository.

## Citations

```
@inproceedings{wang2021indudonet,
  title={InDuDoNet: An Interpretable Dual Domain Network for CT Metal Artifact Reduction},
  author={Wang, Hong and Li, Yuexiang and Zhang, Haimiao and Chen, Jiawei and Ma, Kai and Meng, Deyu and Zheng, Yefeng},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2021},
  organization={Springer}
}
```

## Contact
If you have any question, please feel free to concat Hong Wang (Email: hongwang01@stu.xjtu.edu.cn)