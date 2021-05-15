import os
import sys
import numpy as np
from numpy import savez_compressed
from PIL import Image
from tqdm import tqdm
import multiprocessing

from util_functions import get_normals_from_depth

PRODUCTION = False

if PRODUCTION:
    FOLDER_NAME = 'depth_production'
    TARGET_FOLDER = 'normals_production'
else:
    FOLDER_NAME = 'depth'
    TARGET_FOLDER = 'normals'


def create_normals_from_img(img_name: str, current_idx: int, n_total: int):
    print(f'[Starting] [{current_idx + 1} / {n_total}] creating normals from: {img_name}')
    IMG_PATH = os.path.join('..', FOLDER_NAME, img_name)

    img_arr = np.array(Image.open(IMG_PATH), dtype=np.float32)
    img_arr_normals = np.array(get_normals_from_depth(depth_img_arr=img_arr), dtype=np.float32)

    outfile = img_name.replace('.png', '')
    savez_compressed(f'../{TARGET_FOLDER}/{outfile}', img_arr_normals)
    print(f'[Finished] [{current_idx + 1} / {n_total}] creating normals from: {img_name}')

def main():
    depth_images = sorted(os.listdir(f'../{FOLDER_NAME}'))

    iter_from, iter_to = 0, len(depth_images)

    if len(sys.argv) > 1:
        iter_from = int(sys.argv[1])

    if len(sys.argv) > 2:
        iter_to = int(sys.argv[2])

    print(f'Creating normals. Iterating from:  {iter_from} to: {iter_to} ...')

    for i in tqdm(range(iter_from, iter_to), desc='Creating normals...'):
        file_name = depth_images[i]
        create_normals_from_img(
            img_name=file_name,
            current_idx=i - iter_from,
            n_total=iter_to - iter_from + 1
        )

if __name__ == '__main__':
    main()