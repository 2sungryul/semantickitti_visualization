# Running a pretrained model for 3d object detection

import os
import sys
from os.path import exists, join, dirname
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import numpy as np
from open3d.ml.torch.dataloaders import ConcatBatcher
import torch
import open3d as o3d

"""
    test_split: ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    training_split: ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    all_split: ['00', '01', '02', '03', '04', '05', '06', '07', '09',
                '08', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    validation_split: ['08']
"""

example_dir = os.path.dirname(os.path.realpath(__file__))

def get_segmentation_torch_ckpts():
    kpconv_url = "https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202009090354utc.pth"
    randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth"

    ckpt_path_r = example_dir + "/vis_weights_{}.pth".format('RandLANet')
    if not exists(ckpt_path_r):
        cmd = "wget {} -O {}".format(randlanet_url, ckpt_path_r)
        os.system(cmd)

    ckpt_path_k = example_dir + "/vis_weights_{}.pth".format('KPFCNN')
    if not exists(ckpt_path_k):
        cmd = "wget {} -O {}".format(kpconv_url, ckpt_path_k)
        print(cmd)
        os.system(cmd)

    return ckpt_path_r, ckpt_path_k

def get_detection_torch_ckpts():
    pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_kitti_202012221652utc.pth"
    pointrcnn_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointrcnn_kitti_202105071146utc.pth"
    
    ckpt_path_r = example_dir + "/vis_weights_{}.pth".format('pointpillar')
    if not exists(ckpt_path_r):
        cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path_r)
        os.system(cmd)

    ckpt_path_k = example_dir + "/vis_weights_{}.pth".format('pointrcnn')
    if not exists(ckpt_path_k):
        cmd = "wget {} -O {}".format(pointrcnn_url, ckpt_path_k)
        print(cmd)
        os.system(cmd)

    return ckpt_path_r, ckpt_path_k


vis_points = []

def pred_custom_data(index, pcs, pipeline_r, pipeline_k):
    #vis_points = []
    
    name = "{}".format(index)

    results_r = pipeline_r.run_inference(pcs)
    pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
    # Fill "unlabeled" value because predictions have no 0 values.
    pred_label_r[0] = 0

    results_k = pipeline_k.run_inference(pcs)
    pred_label_k = (results_k['predict_labels'] + 1).astype(np.int32)
    # Fill "unlabeled" value because predictions have no 0 values.
    pred_label_k[0] = 0

    label = pcs['label']
    pts = pcs['point']

    vis_d = {
        "name": name,
        "points": pts,
        "labels": label,
        "pred": pred_label_k,
    }
    vis_points.append(vis_d)

    vis_d = {
        "name": name + "_randlanet",
        "points": pts,
        "labels": pred_label_r,
    }
    vis_points.append(vis_d)

    vis_d = {
        "name": name + "_kpconv",
        "points": pts,
        "labels": pred_label_k,
    }
    vis_points.append(vis_d)

    #return vis_points


cfg_file = "../ml3d/configs/pointpillars_kitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)
model = ml3d.models.PointPillars(**cfg.model)
#print(model)

dataset_path = "/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/KITTI/data_object_velodyne"
cfg.dataset['dataset_path'] = dataset_path
dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)

# download the weights.
ckpt_folder = "./logs/"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "pointpillars_kitti_202012221652utc.pth"
pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_kitti_202012221652utc.pth"
if not os.path.exists(ckpt_path):
    cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path)
    os.system(cmd)

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)

test_split = dataset.get_split("test")
data = test_split.get_data(0)
print(data.keys())
print(data['full_point'].shape)

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
result,box = pipeline.run_inference(data)
print(box)
print(len(box[0]),len(box[0][1]))


# create open3d tensor
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data['full_point'][:, :3])
pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pcd.points), 3)))
 

# create open3d window
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='kitti',width=960, height=540)

# set background_color and point_size
vis.get_render_option().background_color = np.asarray([0,0,0]).astype(float)
vis.get_render_option().point_size = 0.5

# add lidar axis
#axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
#vis.add_geometry(axis_pcd)

# add point cloud
vis.add_geometry(pcd)

# set zoom, front, up, and lookat
#vis.get_view_control().set_zoom(0.1)
#vis.get_view_control().set_front([0, 0, 1])
#vis.get_view_control().set_up([1, 0, 0])
#vis.get_view_control().set_lookat([0, 0, 0])

vis.run()
vis.destroy_window()









"""
model.eval()

# If run_inference is called on raw data.
device = 'cuda'
if isinstance(data, dict):
    batcher = ConcatBatcher(device, model.cfg.name)
    data = batcher.collate_fn([{
        'data': data,
        'attr': {
            'split': 'test'
        }
    }])

data.to(device)

with torch.no_grad():
    results = model(data)
    boxes = model.inference_end(results, data)

print(results)
"""

"""
kitti_labels = ml3d.datasets.SemanticKITTI.get_label_to_names()
v = ml3d.vis.Visualizer()
lut = ml3d.vis.LabelLUT()
for val in sorted(kitti_labels.keys()):
    lut.add_label(kitti_labels[val], val)
v.set_lut("labels", lut)
v.set_lut("pred", lut)

# load pretrained weights 
ckpt_path_pp, ckpt_path_pr = get_detection_torch_ckpts()

model = ml3d.models.PointPillars(ckpt_path=ckpt_path_pp)
pipeline_pp = ml3d.pipelines.ObjectDetection(model,device="gpu")
pipeline_pp.load_ckpt(model.cfg.ckpt_path)

model = ml3d.models.PointRCNN(ckpt_path=ckpt_path_pr)
pipeline_pr = ml3d.pipelines.ObjectDetection(model,device="gpu")
pipeline_pr.load_ckpt(model.cfg.ckpt_path)


# load dataset
dataset_path = "/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/SemanticKITTI/data_odometry_velodyne"
dataset = ml3d.datasets.KITTI(dataset_path)


# load train dataset
train_split = dataset.get_split("train")
len = train_split.__len__()
print('train:',len)
train_data = train_split.get_data(0)
print(train_data.keys())
#train_data['feat']=np.squeeze(train_data['feat'])
train_data['feat']=None
#print(train_data['point'].shape,train_data['feat'].shape,train_data['label'].shape)
print(train_data['point'],train_data['feat'],train_data['label'])

# load valid dataset
valid_split = dataset.get_split("val")
len = valid_split.__len__()
print('val:',len)
val_data = valid_split.get_data(0)
print(val_data.keys())
#val_data['feat']=np.squeeze(val_data['feat'])
val_data['feat']=None
#print(val_data['point'].shape,val_data['feat'].shape,val_data['label'].shape)
print(val_data['point'],val_data['feat'],val_data['label'])

# load test dataset
test_split = dataset.get_split("test")
len = test_split.__len__()
print('test:',len)
test_data = test_split.get_data(0)
print(test_data.keys())
#test_data['feat']=np.squeeze(test_data['feat'])
test_data['feat']=None
#print(test_data['point'].shape,test_data['feat'].shape,test_data['label'].shape)
print(test_data['point'],test_data['feat'],test_data['label'])


# load train dataset

for i in range(5):
    train_data = train_split.get_data(i)
    #print(train_data.keys())
    train_data['feat']=None
    #print(train_data['point'].shape,train_data['feat'].shape,train_data['label'].shape)
    #print(train_data['point'],train_data['feat'],train_data['label'])

    pred_custom_data(i, train_data, pipeline_r, pipeline_k)
    
v.visualize(vis_points)
"""