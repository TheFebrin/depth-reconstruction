"""
Script to download the data from gcs.

HAS TO BE EXECUTED FROM /depth-reconstruction/ FOLDER!

This will download images with ids: 0, 1, 2, ..., 9
python utils/download_data_from_bucket.py 0 10 color
"""

import sys
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def download_image(id: int, image_type: str) -> None:

    BASE_PATH = f'gs://staging-other-matrix-source-data-nomagic-ai/SBX_COCO_NoMagic_V3_2021_01_26/{image_type}/Top_NoMagic_'
    BASE_TARGET = f'{image_type}/Top_NoMagic_'

    path   = BASE_PATH   + (8 - len(str(id))) * '0' + f'{id}.png'
    target = BASE_TARGET + (8 - len(str(id))) * '0' + f'{id}.png'

    bashCommand = 'gsutil cp ' + path + ' ' + target
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()  
    print(bashCommand)

def download_from_bucket():
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


def download_image_from_csv(index: int, run_id: str, rgb_file: str, depth_file: str):
    RGB_PATH = f'gs://results.gripper-ros.nomagic.ai/metrics/{run_id}_control-server/{rgb_file}'
    DEPTH_PATH = f'gs://results.gripper-ros.nomagic.ai/metrics/{run_id}_control-server/{depth_file}'

    RGB_TARGET = f'color_production/color_production_{index}.png'
    DEPTH_TARGET = f'depth_production/depth_production_{index}.png'

    rgb_bash_command = 'gsutil cp ' + RGB_PATH + ' ' + RGB_TARGET
    depth_bash_command = 'gsutil cp ' + DEPTH_PATH + ' ' + DEPTH_TARGET

    # print(rgb_bash_command)
    process = subprocess.Popen(rgb_bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # print(depth_bash_command)
    process = subprocess.Popen(depth_bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(f'[DONE] Image: {index + 1}!')

def download_from_csv():
    PATH = 'utils/production_data.csv'

    raw_data = pd.read_csv(PATH)
    df = raw_data.dropna()

    print(f'Df size: {len(df)} | raw_data size: {len(raw_data)} | Dropped: {len(raw_data) - len(df)}')

    with ThreadPoolExecutor(max_workers=50) as executor:
        for index, row in df.iterrows():
            run_id = row['run_id']
            depth_file = row['depth_file']
            rgb_file = row['rgb_file']

            try:
                print(f'Submitting image: {index + 1} / {len(df)}')
                executor.submit(download_image_from_csv, index, run_id, rgb_file, depth_file)
            except:
                print(f'Error while downloading: {id}')


if __name__ == '__main__':
    download_from_csv()
