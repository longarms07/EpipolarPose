import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import lib.models as models
from lib.core.config import config
from lib.core.config import update_config
from lib.core.integral_loss import get_joint_location_result
from lib.utils.img_utils import convert_cvimg_to_tensor
from lib.utils.vis import drawskeleton, show3Dpose
from collections import OrderedDict


cfg_file = 'experiments/h36m/valid.yaml'
update_config(cfg_file)

image_size = config.MODEL.IMAGE_SIZE[0]

model = models.pose3d_resnet.get_pose_net(config, is_train=False)
print('Created model...')

checkpoint = torch.load(config.MODEL.RESUME, map_location=torch.device("cpu"))
new_dict = OrderedDict()
for k,v in checkpoint.items():
    new_dict[k[7:]] = v
model.load_state_dict(new_dict)
model.eval()
print('Loaded pretrained weights...')

print("Enter the filepath for the image you want to detect a pose in.")
img_path = input()
while ((not ".jpg" in img_path) and (not ".png" in img_path)):
	print("Invalid input! Image needs to be a .jpg or .png")
	img_path = input()
image = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
image = cv2.resize(image, (image_size, image_size))

img_height, img_width, img_channels = image.shape
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_patch = convert_cvimg_to_tensor(image)

mean = np.array([123.675, 116.280, 103.530])
std = np.array([58.395, 57.120, 57.375])
    
# apply normalization
for n_c in range(img_channels):
    if mean is not None and std is not None:
        img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
img_patch = torch.from_numpy(img_patch)

preds = model(img_patch[None, ...])
preds = get_joint_location_result(image_size, image_size, preds, useCuda=False)[0,:,:3]

fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot('121')
drawskeleton(image, preds, thickness=2)
ax.imshow(image)
ax = fig.add_subplot('122', projection='3d', aspect="auto")
show3Dpose(preds, ax, radius=128)
ax.view_init(-75, -90)
plt.show()

