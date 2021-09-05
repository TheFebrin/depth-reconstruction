
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


    

def depth_image_to_pointcloud(depth_in_m):
    rows = (np.arange(0, depth_in_m.shape[0]) - cy) / fy
    cols = (np.arange(0, depth_in_m.shape[1]) - cx) / fx

    meshed_cols, meshed_rows = np.meshgrid(cols, rows)
    result = np.empty((depth_in_m.shape[0], depth_in_m.shape[1], 3))

    result[:, :, 0] = depth_in_m * meshed_cols
    result[:, :, 1] = depth_in_m * meshed_rows
    result[:, :, 2] = depth_in_m

    return result
    

def normalize(img):
    return (img - img.min()) / (img.max() - img.min())
    
    
fx, fy, cx, cy = (1414.16796875, 1414.16796875, 960.0, 540.0)


color = Image.open('color2/SBX_COCO_NoMagic_Polybag_V2_20210315_color_SBXCameraSensor_Top_PolyBag_00000490.png')
depth = Image.open('depth2/SBX_COCO_NoMagic_Polybag_V2_20210315_depth_SBXDepthSensor_Top_PolyBag_00000490.png')

depth = np.array(depth, dtype=np.float32)
depth_ptc = depth_image_to_pointcloud(depth)

print(depth.min(), depth.max())

import pptk

filtered_normals = pptk.estimate_normals(
    normalize(depth_ptc).reshape(-1, 3), 
    k=8, r=0.3, output_eigenvalues=False,
    output_all_eigenvectors=False, output_neighborhood_sizes=False,
    verbose=True
)

print(filtered_normals.shape)

print(filtered_normals.min(), filtered_normals.max())

filtered_normals = filtered_normals.reshape(1080, 1920, 3)

plt.imshow(filtered_normals)
plt.show()

print('DONE')














