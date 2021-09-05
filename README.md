# Thesis Normals Estimation
Engineering Thesis - Surface Normals Estimation using U-Net CNN

## Table of contents
* [Quick start](#quick-start)
* [General info](#general-info)
* [Results](#results)
* [Robot picking the item](#robot-picking-the-item)
* [Notebooks](#notebooks)


--------------
## Quick start

Nice virtualenv tutorial [here](https://computingforgeeks.com/fix-mkvirtualenv-command-not-found-ubuntu/)
```bash
pip3 install --upgrade pip
```

```bash
which python3.7
mkvirtualenv -p <path to python3> <name>
workon <name>
```

```bash
pip install -r requirements.txt
```

--------------
## General info

The goal of this thesis is to show that one can transfer the model trained purely on synthetic data into the real world. I will conduct a qualitative test showing that the robot is able to pick an item using the presented network.


Network was based on U-Net model.
![Model 2 - prod](/images/thesis_figures/Figure_14.png)

The models were trained on two datasets.
Each dataset contained RGB and depth images of the items inside the containers.
Both datasets had 10000 images. 1000 of those images in each dataset were taken to a validation set. The first dataset was quite simple, mostly boxes of different sizes aligned flat in the container. In the second dataset, items types were varying, and their positions were diversified.
![Dataset](/images/thesis_figures/Figure_2.png)

Normal vectors are created straight from depth.
![Normals1](/images/thesis_figures/Figure_9.png)

--------------
## Results

Three main models were trained.
* Model 1 - using only Synthetic Dataset 1
* Model 2 - using both Synthetic Dataset 1 and Synthetic Dataset 2
* Model 3 - same as Model 2, but normals were generated using a slight modification of the algorithm

Model 1 results:
![Model 2 - prod](/images/thesis_figures/Figure_18.png)
![Model 2 - lab](/images/thesis_figures/Figure_19.png)

Model 2 results:
![Model 2 - prod](/images/thesis_figures/Figure_20.png)
![Model 2 - lab](/images/thesis_figures/Figure_21.png)

Model 3 results:
![Model 3 - prod](/images/thesis_figures/Figure_22.png)
![Model 3 - lab](/images/thesis_figures/Figure_23.png)


#### Comparison against a Realsense camera
The normal vector from my model at the grasp point was  `[0.157   0.514  0.282]`, which gives the following Euler ZYX rotation in degrees `(-8.9, 15.0, -61.2)`.
But when I took the average value of the normal vector in the radius of 15 pixels from the grasp point, I got vector `[0.1 0.21 0.54]`. It is equal to Euler ZYX rotation: `(-1.8780727547595142, 9.6, -21.9)`.
Normals generated from the Realsense are much worst in this case. At the grasp point, the vector was `[-0.7 -0.7  0.14]`, giving `(-37, -44.4, 78.6)` Euler ZYX rotation. Taking the average in the radius of 15 pixels gives a vector `[-0.27 -0.66   0.31]`, which equals rotation by `(-13.09, -20.44, 64.96)` degrees in Euler ZYX.
It is worth mentioning that the robot would fail to pick an item with rotation above ~45-50 degrees as it would have problems with picking the item or hitting the tote.
![Model 3 - lab](/images/thesis_figures/Figure_24.png)

--------------
## Robot picking the item
![Robot picking 1](/images/experiment_on_robot/item_left_videos/item_left_height_1_source.gif)

![Robot picking 2](/images/experiment_on_robot/item_left_videos/item_left_height_3_source.gif)

--------------
## Notebooks

1. Data analysis [here](https://github.com/TheFebrin/depth-reconstruction/blob/master/notebooks/analyse_data.ipynb).
2. Unet demo [here](https://github.com/TheFebrin/depth-reconstruction/blob/master/notebooks/unet_demo.ipynb).
3. Dataset augmentation [here](https://github.com/TheFebrin/depth-reconstruction/blob/master/notebooks/augmentations.ipynb).
4. Depth to normals conversion [here](https://github.com/TheFebrin/depth-reconstruction/blob/master/notebooks/depth_to_normals.ipynb).
4. Normals from the model vs normals from depth from Realsense camera [here](https://github.com/TheFebrin/depth-reconstruction/blob/master/notebooks/compare_normals.ipynb).
4. Comparison with MiDas model [here](https://github.com/TheFebrin/depth-reconstruction/blob/master/notebooks/MiDaS_model.ipynb).
