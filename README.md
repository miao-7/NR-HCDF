# CD-NeRF #

### Dependencies ###
* Ubuntu 
* NVIDIA GPU + CUDA
  
### Requirements ###
* torch-ema 
* ninja 
* trimesh 
* opencv-python 
* tensorboardX 
* torch 
* numpy 
* pandas 
* tqdm 
* matplotlib  
* packaging 
* scipy 
* lpips 
* imageio 
* torchmetrics 
* configargparse

### Dataset ###
We used LLFF and NeRF-Synthetic dataset for training and evaluation.
* Download [LLFF](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7) dataset folder and put it under `./data`.
* Download [NeRF-Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) dataset folder and put it under `./data`.

### Run CDNeRF ###
* Use `./run_CDNeRF.sh` script to run CDNeRF
