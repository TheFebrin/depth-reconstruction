import numpy as np
from PIL import Image
from tqdm import tqdm
import numba
import matplotlib.pyplot as plt

@numba.jit(nopython=True)
def get_normals_from_depth(depth_img_arr: np.ndarray) -> np.ndarray:
    n, m = depth_img_arr.shape
    normals_img_arr = np.zeros((n, m, 3), dtype=np.float32)

    for x in range(1, n - 1):
        for y in range(1, m - 1):
            dzdx = (depth_img_arr[x + 1, y] - depth_img_arr[x - 1, y]) / 2.0
            dzdy = (depth_img_arr[x, y + 1] - depth_img_arr[x, y - 1]) / 2.0

            direction = np.array([-dzdx, -dzdy, 1.0], dtype=np.float32)
            magnitude = np.sqrt((direction ** 2).sum())
            normals_img_arr[x, y] = direction / magnitude
    return normals_img_arr


if __name__ == '__main__':
    puma = np.array(Image.open('../images/puma.png'), dtype=np.float32)

    plt.figure(figsize=(10, 5))
    plt.imshow(puma)
    plt.show()

    puma_normals = get_normals_from_depth(depth_img_arr=puma)

    plt.figure(figsize=(10, 5))
    plt.imshow(puma_normals)
    plt.show()