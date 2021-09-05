import numpy as np
from PIL import Image
import numba
import matplotlib.pyplot as plt
# import albumentations as A
# import cv2
import pptk

from util_functions import (
    get_normals_from_depth,
    depth_image_to_pointcloud,
    normalize,
    get_normals_from_depth_avg,
)


def plot_images():
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.tight_layout()

    img1 = Image.open('../images/dataset2/SBXCameraSensor_Top_PolyBag_00000009.png')
    img2 = Image.open('../images/production/color_production_1.png')

    depth1 = Image.open('../images/dataset2/depth_SBXCameraSensor_Top_PolyBag_00000009.png')
    depth2 = Image.open('../images/production/depth_production_1.png')

    depth1 = np.array(depth1, dtype=np.float32)
    depth2 = np.array(depth2, dtype=np.float32)

    normals1 = get_normals_from_depth(depth1)
    normals21 = get_normals_from_depth(depth2)


    normals2 = get_normals_from_depth_avg(depth2, k=2)
    normals3 = get_normals_from_depth_avg(depth2, k=3)
    normals4 = get_normals_from_depth_avg(depth2, k=4)
    normals5 = get_normals_from_depth_avg(depth2, k=5)
    normals6 = get_normals_from_depth_avg(depth2, k=6)


    # pointcloud1 = depth_image_to_pointcloud(depth1)
    #
    # fx, fy, cx, cy = 1338.7076416015625, 1338.7076416015625, 960.0, 540.0
    # pointcloud2 = depth_image_to_pointcloud(
    #     depth2, fx=fx, fy=fy, cx=cx, cy=cy
    # )

    # normals1 = pptk.estimate_normals(
    #     normalize(pointcloud1.reshape((-1, 3))),
    #     k=9, r=0.3, output_eigenvalues=False,
    #     output_all_eigenvectors=False, output_neighborhood_sizes=False,
    #     verbose=True
    # ).reshape(*pointcloud1.shape)
    #
    # normals2 = pptk.estimate_normals(
    #     normalize(pointcloud2.reshape((-1, 3))),
    #     k=9, r=0.2, output_eigenvalues=False,
    #     output_all_eigenvectors=False, output_neighborhood_sizes=False,
    #     verbose=True
    # ).reshape(*pointcloud2.shape)

    axes[0][0].imshow(normals21)
    axes[0][0].set_title('k = 1')

    axes[0][1].imshow(normals2)
    axes[0][1].set_title('k = 2')

    axes[0][2].imshow(normals3)
    axes[0][2].set_title('k = 3')

    axes[1][0].imshow(normals4)
    axes[1][0].set_title('k = 4')

    axes[1][1].imshow(normals5)
    axes[1][1].set_title('k = 5')

    axes[1][2].imshow(normals6)
    axes[1][2].set_title('k = 6')

    plt.show()


def plot_transforms():
	transform1 = A.Compose([
		#A.HorizontalFlip(p=1),
		#A.VerticalFlip(p=1),
		#A.RandomCrop(width=1500, height=900),
		#A.Resize(width=1280, height=720)
		A.GaussianBlur(
		    blur_limit=(11, 15), sigma_limit=10, always_apply=True, p=1
		),
	])

	transform = A.Compose([
		A.ColorJitter(
		    brightness=0.2, contrast=0.2,  saturation=0.2, hue=0.2, always_apply=False, p=0.5
		),
		A.RandomBrightnessContrast(
		    brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5
		),
		A.HueSaturationValue(
		    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5
		),
		A.GaussianBlur(
		    blur_limit=(3, 5), sigma_limit=0, always_apply=False, p=0.5
		),
	])

	# Read an image with OpenCV and convert it to the RGB colorspace
	color_img = cv2.imread('../images/dataset2/SBXCameraSensor_Top_PolyBag_00000009.png')
	color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

	transformed = transform1(image=color_img)
	transformed_color_img = transformed["image"]

	fig, axes = plt.subplots(1, 2, figsize=(10, 5))
	fig.tight_layout()


	axes[0].imshow(color_img)
	axes[0].set_title('Original image')
	axes[1].imshow(transformed_color_img)
	axes[1].set_title('Transformed image after GaussianBlur')

	plt.show()


def main():
	plot_images()

if __name__ == '__main__':
    main()
