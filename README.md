# VF-HM
The offical code of "VF-HM: Vision Loss Estimation using Fundus Photograph for High Myopia" in MICCAI, 2023.

Note: we are preparing for the camera-ready submission, the code will be released ASAP.

## Experiments

First, prepare your dataset as:
```
.
└── data.csv         # data file list
    ├── fundus       # fundus dir
    │   ├── aaa.tif  # fundus image
    │   └── bbb.tif  # fundus image
    └── vf           # vf dir
        ├── aaa.json # vf file
        └── bbb.json # vf file
```

data.csv contains the diagnosis information for each eye, including the fundus file name and corresponding vf file name, in addition, there is one MM category for training data only.

the fundus dir contains fundus images in tif format with RGB colorful mode.

the vf dir contains the vf map in JSON format with 52 effective points.


run baseline regression:
```bash
python train_reg.py
```

run VF-HM:
```bash
python train_vfhm.py
```



## Citation
If you use this method or this code in your research, then please kindly cite it:
```
@inproceedings{yan2023vfhm,
title={VF-HM: Vision Loss Estimation using Fundus Photograph for High Myopia},
author={Zipei Yan, Dong Liang, Linchuan Xu, Jiahang Li, Zhengji Liu, Shuai Wang, Jiannong Cao, Chea-su KEE},
booktitle = {MICCAI},
year={2023},
}
```

