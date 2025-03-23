# Running a pretrained model for 3d object detection
# 3d od model : PointPillars

import logging
import open3d.ml as _ml3d

from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from open3d.ml import datasets

import argparse
from tqdm import tqdm

import open3d.ml.torch as ml3d
from open3d.ml.torch.dataloaders import TorchDataloader as Dataloader
#from ml3d.torch.dataloaders import TorchDataloader as Dataloader


framework = 'torch'
device = "cuda"
dataset_type = "KITTI"
dataset_path = "/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/KITTI/data_object_velodyne"
path_ckpt_pointpillars = "/mnt/d/Users/2sungryul/Dropbox/Work/open3d/pointpillars_kitti_202012221652utc.pth"

framework = _ml3d.utils.convert_framework_name(framework)
device = _ml3d.utils.convert_device_name(device, ['0'])

classname = getattr(datasets, dataset_type)
dataset = classname(dataset_path)

ObjectDetection = _ml3d.utils.get_module("pipeline", "ObjectDetection", framework)
PointPillars = _ml3d.utils.get_module("model", "PointPillars", framework)
cfg = _ml3d.utils.Config.load_from_file("./pointpillars_" + dataset_type.lower() + ".yml")

#model = PointPillars(device=device, **cfg.model)
model = PointPillars(**cfg.model)

pipeline = ObjectDetection(model, dataset, device=device)

# load the parameters.
pipeline.load_ckpt(ckpt_path=path_ckpt_pointpillars)

test_split = Dataloader(dataset=dataset.get_split('training'),
                        preprocess=model.preprocess,
                        transform=None,
                        use_cache=False,
                        shuffle=False)
print(test_split.__class__)
print(test_split[0].shape)

# run inference on a single example.
data = test_split[5]['data']
print(data.__class__)
print(data.keys())

'''
result = pipeline.run_inference(data)[0]
print(result.__class__)
print(len(result))
print(result[0].__class__)
print(result[0].shape,result[1].shape,result[2].shape)
print(result[0].to_kitti_format())
'''

#boxes = data['bbox_objs']
#boxes.extend(result)
"""
vis = Visualizer()

lut = LabelLUT()
for val in sorted(dataset.label_to_names.keys()):
    lut.add_label(val, val)

# Uncommenting this assigns bbox color according to lut
# for key, val in sorted(dataset.label_to_names.items()):
#     lut.add_label(key, val)

vis.visualize([{
    "name": dataset_type,
    'points': data['point']
}],
                lut,
                bounding_boxes=boxes)

# run inference on a multiple examples
vis = Visualizer()
lut = LabelLUT()
for val in sorted(dataset.label_to_names.keys()):
    lut.add_label(val, val)

boxes = []
data_list = []
for idx in tqdm(range(100)):
    data = test_split[idx]['data']

    result = pipeline.run_inference(data)[0]

    boxes = data['bbox_objs']
    boxes.extend(result)

    data_list.append({
        "name": dataset_type + '_' + str(idx),
        'points': data['point'],
        'bounding_boxes': boxes
    })

vis.visualize(data_list, lut, bounding_boxes=None)"
"""