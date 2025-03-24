# visualize both image and point cloud with color label from SemanticKITTI dataset

import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d
import numpy as np
import open3d as o3d
import cv2
import glob

dataset_path ='/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/SemanticKITTI/data_odometry_velodyne'
image_path=r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/SemanticKITTI/data_odometry_color/dataset/sequences/00/image_2"

label_to_names = {
            0: 'unlabeled',
            1: 'car',
            2: 'bicycle',
            3: 'motorcycle',
            4: 'truck',
            5: 'other-vehicle',
            6: 'person',
            7: 'bicyclist',
            8: 'motorcyclist',
            9: 'road',
            10: 'parking',
            11: 'sidewalk',
            12: 'other-ground',
            13: 'building',
            14: 'fence',
            15: 'vegetation',
            16: 'trunk',
            17: 'terrain',
            18: 'pole',
            19: 'traffic-sign'
        }
 
Colors = [[0., 0., 0.], [0.96078431, 0.58823529, 0.39215686],
              [0.96078431, 0.90196078, 0.39215686],
              [0.58823529, 0.23529412, 0.11764706],
              [0.70588235, 0.11764706, 0.31372549], [1., 0., 0.],
              [0.11764706, 0.11764706, 1.], [0.78431373, 0.15686275, 1.],
              [0.35294118, 0.11764706, 0.58823529], [1., 0., 1.],
              [1., 0.58823529, 1.], [0.29411765, 0., 0.29411765],
              [0.29411765, 0., 0.68627451], [0., 0.78431373, 1.],
              [0.19607843, 0.47058824, 1.], [0., 0.68627451, 0.],
              [0., 0.23529412,
               0.52941176], [0.31372549, 0.94117647, 0.58823529],
              [0.58823529, 0.94117647, 1.], [0., 0., 1.], [1.0, 1.0, 0.25],
              [0.5, 1.0, 0.25], [0.25, 1.0, 0.25], [0.25, 1.0, 0.5],
              [0.25, 1.0, 1.25], [0.25, 0.5, 1.25], [0.25, 0.25, 1.0],
              [0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.375, 0.375, 0.375],
              [0.5, 0.5, 0.5], [0.625, 0.625, 0.625], [0.75, 0.75, 0.75],
              [0.875, 0.875, 0.875]]

print(len(Colors))
print(Colors[0])

for filename in glob.glob(image_path+'/'+'*.png'):
    break

image = cv2.imread(filename)
cv2.imshow("Image", image)
cv2.waitKey()

# construct a dataset by specifying dataset_path
dataset = ml3d.datasets.SemanticKITTI(dataset_path)
#print('dataset',dataset)
#dataset = ml3d.datasets.SemanticKITTI()

# get the 'all' split that combines training, validation and test set
all_split = dataset.get_split('train')
print(len(all_split))

# print the attributes of the first datum
print(all_split.get_attr(0))

# print the shape of the first point cloud
print(all_split.get_data(0).keys())
print(all_split.get_data(0)['point'].shape)
print(all_split.get_data(0)['label'].shape)
print(all_split.get_data(0)['feat'].shape)
print(all_split.get_data(0)['label'])

colorlist = []
for label in all_split.get_data(0)['label']:
    colorlist.append(Colors[label])

print(len(colorlist))

# create open3d tensor
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_split.get_data(0)['point'])
pcd.colors = o3d.utility.Vector3dVector(colorlist)

# create open3d window
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='semantickitti',width=1241, height=376)

# set background_color and point_size
vis.get_render_option().background_color = np.asarray([0,0,0]).astype(float)
vis.get_render_option().point_size = 2

# add lidar axis
#axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
#vis.add_geometry(axis_pcd)

# add point cloud
vis.add_geometry(pcd)

# set zoom, front, up, and lookat
vis.get_view_control().set_zoom(0.02)
vis.get_view_control().set_front([-2, 0, 1])
vis.get_view_control().set_up([1, 0, 1])
vis.get_view_control().set_lookat([-10, 0, 0])

vis.run()
vis.destroy_window()
