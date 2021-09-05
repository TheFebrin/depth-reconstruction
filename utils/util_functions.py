import numpy as np
from PIL import Image
from tqdm import tqdm
import numba
import matplotlib.pyplot as plt
import PyKDL


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


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


@numba.jit(nopython=True)
def get_normals_from_depth_avg(depth_img_arr: np.ndarray, k: int = 1) -> np.ndarray:
    n, m = depth_img_arr.shape
    normals_img_arr = np.zeros((n, m, 3), dtype=np.float32)

    for x in range(k, n - k):
        for y in range(k, m - k):
            right_x = depth_img_arr[x + 1: x + k + 1, y]
            left_x = depth_img_arr[x - k: x, y]

            top_y = depth_img_arr[x, y + 1: y + k + 1]
            bot_y = depth_img_arr[x, y - k: y]

            dzdx = (right_x.mean() - left_x.mean()) / 2.0
            dzdy = (top_y.mean() - bot_y.mean()) / 2.0

            direction = np.array([-dzdx, -dzdy, 1.0], dtype=np.float32)
            magnitude = np.sqrt((direction ** 2).sum())
            normals_img_arr[x, y] = direction / magnitude
    return normals_img_arr


def depth_image_to_pointcloud(
    depth_in_m: np.ndarray,
    fx: float = 913.22943,  # Camera calibration for production data
    cx: float = 636.6216,
    fy: float = 912.227,
    cy: float = 354.47263,
):
    rows = (np.arange(0, depth_in_m.shape[0]) - cy) / fy
    cols = (np.arange(0, depth_in_m.shape[1]) - cx) / fx

    meshed_cols, meshed_rows = np.meshgrid(cols, rows)
    result = np.empty((depth_in_m.shape[0], depth_in_m.shape[1], 3))

    result[:, :, 0] = depth_in_m * meshed_cols
    result[:, :, 1] = depth_in_m * meshed_rows
    result[:, :, 2] = depth_in_m

    return result


def rotation_between_normals(a, b):
    assert len(b) == len(a) == 3
    assert a.dtype == b.dtype == np.float32

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    v = np.cross(a, b)

    s = np.linalg.norm(v)

    c = np.dot(a, b)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    #     r = np.eye(3) + vx + vx @ vx * (1 - c) / (s ** 2)
    r = np.eye(3) + vx + vx @ vx * (1 / (1 + c))  # better formula
    return r


def rotation_to_quaternion(r):
    return PyKDL.Rotation(*r.ravel()).GetQuaternion()


def rotation_to_euler(r):
    return PyKDL.Rotation(*r.ravel()).GetEulerZYX()


def rotation_to_euler_zyx(r):
    euler = PyKDL.Rotation(*r.ravel()).GetEulerZYX()
    return tuple(map(lambda x: x * 180 / np.pi, euler))


if __name__ == '__main__':
    puma = np.array(Image.open('../images/puma.png'), dtype=np.float32)

    plt.figure(figsize=(10, 5))
    plt.imshow(puma)
    plt.show()

    puma_normals = get_normals_from_depth(depth_img_arr=puma)

    plt.figure(figsize=(10, 5))
    plt.imshow(puma_normals)
    plt.show()
