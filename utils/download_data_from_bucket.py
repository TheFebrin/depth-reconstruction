"""
Script to download the data from gcs.

This will download images with ids: 0, 1, 2, ..., 9
python utils/download_data_from_bucket.py 0 10 color
"""

import sys
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

def download_image(id: int, image_type: str) -> None:

    BASE_PATH = f'gs://staging-other-matrix-source-data-nomagic-ai/SBX_COCO_NoMagic_V3_2021_01_26/{image_type}/Top_NoMagic_'
    BASE_TARGET = f'{image_type}/Top_NoMagic_'

    path   = BASE_PATH   + (8 - len(str(id))) * '0' + f'{id}.png'
    target = BASE_TARGET + (8 - len(str(id))) * '0' + f'{id}.png'

    bashCommand = 'gsutil cp ' + path + ' ' + target
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()  
    print(bashCommand)


if __name__ == '__main__':
    image_type = 'color'
    id_image_from, id_image_to = 0, 10

    if len(sys.argv) > 1:
        id_image_from = int(sys.argv[1])

    if len(sys.argv) > 2:
        id_image_to = int(sys.argv[2])

    if len(sys.argv) > 3:
        image_type = sys.argv[3]
        assert image_type in ['color', 'depth'], f'Wrong image_type: {image_type}'

    with ThreadPoolExecutor(max_workers=50) as executor:
        for id in range(id_image_from, id_image_to):
            try:
                executor.submit(download_image, id, image_type)
            except:
                print(f'Error while downloading: {id}')
