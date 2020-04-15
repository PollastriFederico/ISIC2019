# A Deep Analysis on High Resolution Dermoscopic Images Classification

To use this code first unzip dataset_files.zip, place its content in a directory,
and add a subdirectory named images containing every sample from both the training 
and the test set of task 3 of the 2019 isic challenge (https://challenge2019.isic-archive.com/data.html).
Then specify the name of the new directory as the value of **data_root** in both 
isic_classification_dataset.py and dataset_partitioning.py.

To train the networks described in Table 5 of the paper:

``` bash
classification_net.py --network densenet201 --epochs 120 --save_dir /my_dir/MODELS/ --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 16 --learning_rate 0.001 --size 512 --batch_size 8
classification_net.py --network seresnext50 --epochs 120 --save_dir /my_dir/MODELS/ --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 84 --learning_rate 0.01 --size 512 --batch_size 12
classification_net.py --network resnet152 --epochs 120 --save_dir /my_dir/MODELS/ --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 116 --learning_rate 0.001 --size 512 --batch_size 8
```

augm config is a *code* obtained as the sum of employed data augmentation strategy following values:

``` python

self.possible_aug_list = [
            None,  # dummy for padding mode                                                         # 1
            None,  # placeholder for future inclusion                                               # 2
            sometimes(ia.augmenters.AdditivePoissonNoise((0, 10), per_channel=True)),               # 4
            sometimes(ia.augmenters.Dropout((0, 0.02), per_channel=False)),                         # 8
            sometimes(ia.augmenters.GaussianBlur((0, 0.8))),                                        # 16
            sometimes(ia.augmenters.AddToHueAndSaturation((-20, 10))),                              # 32
            sometimes(ia.augmenters.GammaContrast((0.5, 1.5))),                                     # 64
            None,  # placeholder for future inclusion                                               # 128
            None,  # placeholder for future inclusion                                               # 256
            sometimes(ia.augmenters.PiecewiseAffine((0, 0.04))),                                    # 512
            sometimes(ia.augmenters.Affine(shear=(-20, 20), mode=self.mode)),                       # 1024
            sometimes(ia.augmenters.CropAndPad(percent=(-0.2, 0.05), pad_mode=self.mode))           # 2048
        ]

```
Random Flipping and Rotations are **always** employed. Images are not simply resized, yet filled to be 
perfectly squared and **then** resized to 512x512

Every network is trained with:
- Optimizer: SGD
- Scheduler: Plateau (on Validation Balanced Accuracy)

Networks are trained for a maximum of 120 epochs, checkpoint that obtain a new best weighted accuracy on the 
**validation set** are saved.

The ensemble result is obtained by employing the Data Augmentation ensemble technique.

To calculate the results of the ensemble in Table 5 of the paper:

``` bash
models_ensemble.py --avg ensemble.txt --da_n_iter 30
```

where ensemble.txt is a simple txt file containing only the following lines.

``` txt
--network densenet201 --save_dir /my_dir/MODELS/ --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 16 --learning_rate 0.001 --size 512 --batch_size 8 --load_epoch *best saved epoch*
--network seresnext50 --save_dir /my_dir/MODELS/ --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 84 --learning_rate 0.01 --size 512 --batch_size 12 --load_epoch *best saved epoch*
--network resnet152 --save_dir /my_dir/MODELS/ --optimizer SGD --scheduler plateau --cutout_holes 0 0 1 2 3 --cutout_pad 20 50 100  --augm_config 116 --learning_rate 0.001 --size 512 --batch_size 8 --load_epoch *best saved epoch*
```
