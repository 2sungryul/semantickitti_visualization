# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# semantickitti dataset animation example(point cloud with color label)

import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d
import numpy as np
import open3d as o3d
import threading
import time
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

CLOUD_NAME = "points"
FRAME_NUM = 1000

class MultiWinApp:

    def __init__(self):
        self.is_done = False
        self.cloud = None
        self.main_vis = None
        self.frame_index = 0
        self.first = False
        self.bbox_num = 0
        #self.n_snapshots = 0
        #self.snapshot_pos = None

        # construct a dataset by specifying dataset_path
        dataset = ml3d.datasets.SemanticKITTI(dataset_path)
        #print('dataset',dataset)
        
        # get the 'all' split that combines training, validation and test set
        self.all_split = dataset.get_split('train')
        print('train dataset size:',len(self.all_split))

        # print the attributes of the first datum
        #print(all_split.get_attr(0))

        # print the shape of the first point cloud
        print(self.all_split.get_data(0).keys())
        print(self.all_split.get_data(0)['point'].shape)
        print(self.all_split.get_data(0)['label'].shape)
        print(self.all_split.get_data(0)['feat'].shape)
        #print(all_split.get_data(0)['label'])

        #colorlist = []
        #for label in self.all_split.get_data(0)['label']:
        #    colorlist.append(Colors[label])

        #print('colorlist size:',len(colorlist))
        

    def run(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.main_vis = o3d.visualization.O3DVisualizer("semantickitti", width=1241, height=376)
        self.main_vis.reset_camera_to_default()
        #self.main_vis.setup_camera(80, [0, 0, 0], [-15, 0, 10], [5, 0, 10]) # center, eye, up
        self.main_vis.setup_camera(100, [0, 0, 0], [-10, 0, 7], [1, 0, 1]) # center, eye, up
        
        self.main_vis.set_background(np.array([0, 0, 0, 0]), None)
        self.main_vis.show_skybox(False)
        self.main_vis.point_size = 2
        self.main_vis.show_settings = False
                
        self.main_vis.set_on_close(self.on_main_window_closing)
        app.add_window(self.main_vis)
        threading.Thread(target=self.update_thread).start()

        #self.main_vis.add_action("Take snapshot in new window", self.on_snapshot)
        #self.snapshot_pos = (self.main_vis.os_frame.x, self.main_vis.os_frame.y)

        app.run()
    
    
    def on_main_window_closing(self):
        self.is_done = True
        return True  # False would cancel the close

    def update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.        
           
        # Initialize point cloud geometry
        point_cloud = o3d.geometry.PointCloud()
           
        while not self.is_done:
                       
            time.sleep(1)
                   
            def update_cloud():
                print("frame_index:",self.frame_index,self.all_split.get_data(self.frame_index)['point'].shape)
                if self.first:
                    self.main_vis.remove_geometry("axis")
                    self.main_vis.remove_geometry("pc")
                                                    
                axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.6, origin=[0, 0, 0])
                self.main_vis.add_geometry("axis", axis_pcd)

                # Update point cloud with new data
                point_cloud.points = o3d.utility.Vector3dVector(self.all_split.get_data(self.frame_index)['point'])
                colorlist = []
                for label in self.all_split.get_data(self.frame_index)['label']:
                    colorlist.append(Colors[label])

                point_cloud.colors = o3d.utility.Vector3dVector(colorlist)
                self.main_vis.add_geometry("pc", point_cloud)
                                
                # save screen image to jpg                
                self.main_vis.export_current_image("pc_%06d.png" % self.frame_index)
                
                # Move to the next frame
                self.frame_index = (self.frame_index + 1) % FRAME_NUM
                self.first = True
                
            o3d.visualization.gui.Application.instance.post_to_main_thread(self.main_vis, update_cloud)            

            if self.is_done:  # might have changed while sleeping
                break


def main():
    MultiWinApp().run()

if __name__ == "__main__":
    main()