"""
TODO
"""

import sys
import subprocess
import multiprocessing
import concurrent
import concurrent.futures
from multiprocessing import Pool

def download_image(id: int, type: str):

    BASE_PATH = f'gs://staging-other-matrix-source-data-nomagic-ai/SBX_COCO_NoMagic_V3_2021_01_26/{type}/Top_NoMagic_'
    BASE_TARGET = f'{type}/Top_NoMagic_'

    def download(id: int):
        path   = BASE_PATH   + (8 - len(str(id))) * '0' + f'{id}.png'
        target = BASE_TARGET + (8 - len(str(id))) * '0' + f'{id}.png'

        bashCommand = 'gsutil cp ' + path + ' ' + target
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    # TODO: USE THREADS
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.submit(download, id)

    download(id=id)

    print(f'{id} downloaded.')


if __name__ == '__main__':
    type = 'color'
    n_pictures = 10
    if len(sys.argv) > 1:
        n_pictures = int(sys.argv[1])

    if len(sys.argv) > 2:
        type = sys.argv[2]
        assert type in ['color', 'depth'], f'Wrong type: {type}'

    for id in range(n_pictures):
        try:
            download_image(id=id, type=type)
        except:
            print(f'Error while downloading: {id}')
