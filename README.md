# VF-HM
The offical PyTorch implementation of "VF-HM: Vision Loss Estimation using Fundus Photograph for High Myopia" in MICCAI, 2023.

<!-- Note: we are preparing for the camera-ready submission, the code will be released ASAP. -->

## Experiments
### Setup
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

`data.csv` contains the diagnosis information for each eye, including the fundus file name and corresponding vf file name, eye_type: Left (L) or Right(R); in addition, there is one MM category (C0/C1/C2/C3/C4) for training data only.

| fundus_id | vf_id    | mm | eye_type |
|-----------|----------|----|----------|
| aaa.tif   | aaa.json | C0 | L        |
| bbb.tif   | bbb.json | C1 | L        |
| ...       | ...      | .  | .        |


`fundus` dir contains fundus images in tif format with RGB colorful mode. An example of fundus (Left eye) is shown as follows (From [Wikipedia](https://en.wikipedia.org/wiki/Fundus_photography)):

<!-- ![fundus](https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Fundus_photograph_of_normal_left_eye.jpg/500px-Fundus_photograph_of_normal_left_eye.jpg) -->
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Fundus_photograph_of_normal_left_eye.jpg/500px-Fundus_photograph_of_normal_left_eye.jpg" width="200">

`vf` dir contains the vf map in JSON format with 52 effective points. An example of VF (Left eye) is shown as follows:
```
[[nan nan nan nan nan nan nan nan nan nan]
 [nan nan nan 24. 26. 24. 21. nan nan nan]
 [nan nan 27. 26. 24. 25. 25. 25. nan nan]
 [nan 26. 24. 26. 27. 26. 26. 27. 23. nan]
 [nan 18. nan 25. 29. 29. 29. 28. 26. 23.]
 [nan  0. nan 26. 30. 30. 29. 28. 27. 20.]
 [nan 24. 27. 27. 29. 29. 25. 28. 24. nan]
 [nan nan 27. 29. 29. 29. 28. 29. nan nan]
 [nan nan nan 31. 26. 27. 27. nan nan nan]
 [nan nan nan nan nan nan nan nan nan nan]]
```

### Run
run regression baseline:
```bash
bash train_reg.sh > train_reg.out &
```

run VF-HM:
```bash
bash train_vfhm.sh > train_vfhm.out &
```



## Citation
If you use our method or our code in your research, please kindly cite it:
```
@inproceedings{yan2023vfhm,
title={{VF-HM}: Vision loss estimation using fundus photograph for high myopia},
author={Zipei Yan, Dong Liang, Linchuan Xu, Jiahang Li, Zhengji Liu, Shuai Wang, Jiannong Cao, Chea-su Kee},
booktitle={MICCAI},
year={2023},
}
```

