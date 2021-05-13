# Depth Reconstruction
Engineering Thesis - Depth reconstruction using synthetic data.

## Table of contents
* [Quick start](#quick-start)
* [General info](#general-info)
* [Folder structure](#folder-structure)
* [Training](#training)
* [Results](#results)
* [Notebooks](#notebooks)


--------------
## Quick start

Nice virtualenv tutorial [here](https://computingforgeeks.com/fix-mkvirtualenv-command-not-found-ubuntu/)
```bash
pip3 install --upgrade pip
```

```bash
which python3.8
mkvirtualenv -p <path to python3> <name>
workon <name>
```

```bash
pip install -r requirements.txt
```

--------------
## General info

TODO

--------------
## Folder structure

```
├── /models                 - folder for keeping models
│    ├── model.py
│    └── unet.py
│   
├──  /mains                - main files responsible for the whole pipeline
│    └── main.py 
|
├──  /figures              - figures used in README
│    └── ...
│ 
├──  /notebooks 
│    ├── /Convert_to_COCO_format.ipynb
│    └── /Data_analysis.ipynb
| 
├──  /dataloaders  
│    └── dataloader.py 
│
└── /utils 
     ├── download_data_from_bucket.py
     └── ...
```

--------------
## Training

TODO

--------------
## Results

TODO


--------------
## Notebooks

1. Data analysis [here](https://github.com/TheFebrin/depth-reconstruction/blob/master/notebooks/analyse_data.ipynb).
2. Unet demo [here](https://github.com/TheFebrin/depth-reconstruction/blob/master/notebooks/unet_demo.ipynb).

