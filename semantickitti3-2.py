# Running a pretrained model for 3d object detection
# 3d od model : PointRCNN

import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

cfg_file = "./pointrcnn_kitti.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointRCNN(**cfg.model)
cfg.dataset['dataset_path'] = "/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/KITTI/data_object_velodyne"
dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)

# download the weights.
ckpt_folder = "./"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "pointrcnn_kitti_202105071146utc.pth"
pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointrcnn_kitti_202105071146utc.pth"
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
#print(data['bounding_boxes'].__class__)
print(data['point'].__class__)
print(data['point'].shape)
data['point'] = data['point'][:, :3]
# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
result = pipeline.run_inference(data)
#print(result)

print('result')
print(result.__class__)
print(len(result))
print('result[0]')
print(result[0].__class__)
print(len(result[0]))
print('result[0][0]')
#print(len(result[0][0]))
print(result[0][0].__class__)
print(result[0][0].shape)

print('result[0][1]')
#print(len(result[0][0]))
print(result[0][1].__class__)
print(result[0][1].shape)

print('result[0][2]')
#print(len(result[0][0]))
print(result[0][2].__class__)
print(result[0][2].shape)

print('result[1]')
print(len(result[1]))
print(len(result[1][0]))
print(len(result[1][0][0]))
#print(result[1])

# evaluate performance on the test set; this will write logs to './logs'.
#pipeline.run_test()