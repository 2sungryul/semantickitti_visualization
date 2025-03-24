# Running a pretrained model for 3d object detection
# 3d od model : PointPillars

import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

cfg_file = "./pointpillars_kitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model)
cfg.dataset['dataset_path'] = "/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/KITTI/data_object_velodyne"
dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)

# download the weights.
ckpt_folder = "./"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "pointpillars_kitti_202012221652utc.pth"
pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_kitti_202012221652utc.pth"
if not os.path.exists(ckpt_path):
    cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path)
    os.system(cmd)

# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)

test_split = dataset.get_split("training")
data = test_split.get_data(1)
print('data')
print(data.__class__)
print(data.keys())
print(data['bounding_boxes'].__class__)

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
result = pipeline.run_inference(data)[0]

#print(result.__class__)
print('result:',type(result))
print(len(result))
print(result)
print('result[0]:',type(result[0]))
print(result[0].center)
print(result[0].size)
print(result[0].yaw)
print(result[0].label_class)
print(result[0].confidence)
#print(result[0].world_cam)
#print(result[0].cam_img)

# evaluate performance on the test set; this will write logs to './logs'.
#pipeline.run_test()