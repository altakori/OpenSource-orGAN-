# OpenSource-orGAN-

## Project
```
🖼️change real pictures to comics(real time)
```
## Demo Link
[Google Drive](https://drive.google.com/file/d/1gf8eFGyDYY8mwgcsxuldv8F55P2HH5gf/view?usp=sharing)

## Table of Contents

- [Project](#Project)
- [Demo](#Demo)
- [Training envirionment](#Training-envirionment)
- [Procedure](#Procedure)
- [Package](#Package)
- [Dataset](#Dataset)
- [Model](#Model)
- [Folder structure](#Folder-structure)
- [Training Result](#Training-Result)
- [Reference](#Reference)
- [Team members](#Team-members)
- [License](#license)

## 

## Training envirionment
|CPU|GPU|RAM|SSD|HDD|OS|
|---|---|---|---|---|---|
|i9-11900K|RTX 3080|128G|2TB|18T|Ubuntu 22.04|

## Procedure
0. Load generator model
1. Read camera through Opencv
2. Resize & Normalize
3. Insert image into model
4. Show image through Opencv
5. repeat 1~5 until press esc key

## Package
#### Train
```
conda env create --file conda-environment.yaml
conda activate jy
 ```

#### Only Test
```
python=3.11
torchvision
torch
wandb
argparse
numpy
opencv-python
pillow
```

## Dataset
[Dataset(github)](https://github.com/Sxela/face2comics)     
[Comic faces v1(kaggle)](https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic)     
[Comic faces v2(kaggle)](https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic-v2)

## Model
[Google Drive](https://drive.google.com/file/d/1sgxcW-dCx7w06zFWDDoRJzl6q3dfhsf5/view?usp=drive_link)

## Folder structure
```
data
|    faces
|    |  0.jpg
|    |  1.jpg
|    |  ...
|    comics
|    |  0.jpg
|    |  1.jpg
|    |  ...
|
model1.pt
|
test.py
```

## Training Result
[Train Result(wandb)](https://wandb.ai/takeout/face2comic?workspace=user-takeout)

![image](https://github.com/altakori/OpenSource-orGAN-/assets/92903593/a240cc22-b9e7-4287-8273-20dbedbc36b1)

## Reference
https://www.tensorflow.org/tutorials/generative/pix2pix

https://github.com/Seonghoon-Yu/AI_Paper_Review/blob/master/GAN/pix2pix(2016).ipynb

https://arxiv.org/pdf/1611.07004.pdf


## Team members

<table>
  <tr>
    <td align="center" width="150px">
      <a href="https://github.com/altakori" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126345795?v=4" alt="박상혁" />
      </a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/be0k" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/141907272?v=4" alt="김준영" />
      </a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/qnq314" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/143786169?v=4" alt="김정우" />
      </a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/isaac8570" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/92903593?v=4" alt="김이삭" />
      </a>
    </td>
  </tr>
  <tr>
   <td align="center">
      <a href="https://github.com/altakori" target="_blank">
        박상혁
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/be0k" target="_blank">
        김준영
      </a>
    </td>
     <td align="center">
      <a href="https://github.com/qnq314" target="_blank">
        김정우
      </a>
    </td>
     <td align="center">
      <a href="https://github.com/isaac8570" target="_blank">
        김이삭
      </a>
    </td>
  </tr>
<table>


## License
See the [`LICENSE`](https://github.com/altakori/OpenSource-orGAN-/blob/main/LICENSE) file.
