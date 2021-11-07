# siamese.pytorch
Siamese Network for Omniglot dataset

![](https://github.com/gaungalif/siamese.pytorch/blob/main/results/results.gif)

This repository contains the code to train Omniglot datasets taken from kaggle as an image simmiliarity recognition using pytorch. We use a Siamese Network that contains a backbone and classifier from Convolutional Siamese Network architecture. Unlike the traditional CNNs that take an input of 1 image to generate a one-shot vector suggesting the category, the image belongs to, the Siamese architecture takes in 2 images and feeds them into 2 CNNs with the same structure, then the output will be merged together and feed into fully connected layers to output 32 encoder digits that represents the distances of the two vector of the images with pairwise distances.



Instead of learning which image belongs to which class, the Siamese Architecture learns how to detect different handwritten characters from omniglot datasets. I got 71% of accuracy with the omniglot Alphabet_of_the_Magi datasets
feel free to try, comment, and share! XD


## Requirment:
- onnx==1.10.1
- onnxruntime==1.9.0
- torch==1.8.2
- torchvision==0.9.2 
- torchaudio==0.8.2
- python==3.8.0
- pytorch-lightning==1.4.5
- opencv-python==4.4.0.46
- opencv-python-headless==4.5.1.48
- fire==0.4.0

## Hardware Requirment:
- Computer with decend RAM and CPU
- GPU (optional)

## How to Use:
### Dataset:
- We build the General Datamodule that would load and automatically created 2 pair of images list for each classes and then multipy them so that for  every single images in the dataset, it will create and combine with all the images in classes inside the dataset. (eg if there is 2 classes with 3 images, there will be 30 images in total. (simmilar classes images : 1a+1b, 1a+1c and different classes images 1a+2a, 1a+2b and so on...)) *don't worry, it will automatically balance the simmilar and different type for every classes*
- Download the dataset manually from here: https://www.kaggle.com/watesoyan/omniglot/download
- you can also use any other datasets for this networks, just create the structure like one of the Omniglot Character Type in omniglot datasets, or same as ImageNet dataset

- Store dataset at `dataset/omniglot/`
- There are few types of the characters included in the omniglot datasets, you can use one of the character type like 'Alphabet_of_the_Magi', just change the path to the character folders path 


### Training:
- Use `train.py` to train the model.
- Change `dataset` path to the appropriate path if needed
- You can modify the Hyperparameter and Augmentation if needed
- Use this command 'python train.py --help' for help

example command: 
```
python train.py  --data_dir './dataset/omniglot/Alphabet_of_the_Magi' --max_epoch 100 --batch_size 64 --num_workers 2 --backbone_name siamese --simmilar_data_multiplier 20
```


### Test:
- Use 'test.py' to test the model that you have trained.
- modify the data_dir(default: ./dataset/omniglot/) and weight (default: weights/*.pth) or weight_onnx(default: weights/*.pth.onnx) to the specific path location to test the image
- change the `--pic_idx` to change the images 
- the program will show 2 pairs of images and with the predicted numbers, labels, and the threshold for the simmilarity

- download the onnx weight that i have been trained for the 'omniglot/Alphabet_of_the_Magi' datasets here: 
https://drive.google.com/file/d/1PMXItnbH0NjAGT7np_75yEovuH_PZkDE/view?usp=sharing
- store the weight at `weights/`


example command: 
```
python test.py --data_dir ./dataset/omniglot/Alphabet_of_the_Magi --weight_onnx ./weights/Alphabet-of-the-Magi-siamese-10epochs.onnx --pic_idx 23
```
- output :
![](https://github.com/gaungalif/siamese.pytorch/blob/main/results/results.gif)

## Reference:

- Watesoyan, omniglot, https://www.kaggle.com/watesoyan/omniglot/download
- Torchvision Documentation, Pytorch, https://pytorch.org/vision/stable/index.html
- Pytorch Lightning Documentation, https://pytorch-lightning.readthedocs.io/en/latest/
- Pytorch ONNX, https://pytorch.org/docs/stable/onnx.html