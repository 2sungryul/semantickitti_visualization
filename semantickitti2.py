# Running a pretrained model for semantic segmentation

import os
import sys
from os.path import exists, join, dirname
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import numpy as np

"""
    test_split: ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    training_split: ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    all_split: ['00', '01', '02', '03', '04', '05', '06', '07', '09',
                '08', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    validation_split: ['08']
"""

example_dir = os.path.dirname(os.path.realpath(__file__))

def get_torch_ckpts():
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


kitti_labels = ml3d.datasets.SemanticKITTI.get_label_to_names()
v = ml3d.vis.Visualizer()
lut = ml3d.vis.LabelLUT()
for val in sorted(kitti_labels.keys()):
    lut.add_label(kitti_labels[val], val)
v.set_lut("labels", lut)
v.set_lut("pred", lut)

# load pretrained weights 
ckpt_path_r, ckpt_path_k = get_torch_ckpts()

model = ml3d.models.RandLANet(ckpt_path=ckpt_path_r)
pipeline_r = ml3d.pipelines.SemanticSegmentation(model,device="gpu")
pipeline_r.load_ckpt(model.cfg.ckpt_path)

model = ml3d.models.KPFCNN(ckpt_path=ckpt_path_k)
pipeline_k = ml3d.pipelines.SemanticSegmentation(model,device="gpu")
pipeline_k.load_ckpt(model.cfg.ckpt_path)

# load dataset
dataset_path = "/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/SemanticKITTI/data_odometry_velodyne"
dataset = ml3d.datasets.SemanticKITTI(dataset_path)

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
