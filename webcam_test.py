import cv2
import torch
import numpy as np
import lib.models as models
from lib.core.config import config
from lib.core.config import update_config
from lib.core.integral_loss import get_joint_location_result
from lib.utils.img_utils import convert_cvimg_to_tensor
from lib.utils.vis import drawskeleton, show3Dpose
from collections import OrderedDict


class Epipolar_Webcam:
	def __init__(self):
		#Configure Epipolar Pose and the testing model
		print('Starting Epipolar Pose...')
		cfg_file = 'experiments/h36m/valid.yaml'
		update_config(cfg_file)

		self.image_size = config.MODEL.IMAGE_SIZE[0]

		self.model = models.pose3d_resnet.get_pose_net(config, is_train=False)
		print('Created model...')

		checkpoint = torch.load(config.MODEL.RESUME, map_location=torch.device("cpu"))
		new_dict = OrderedDict()
		for k,v in checkpoint.items():
		    new_dict[k[7:]] = v
		self.model.load_state_dict(new_dict)
		self.model.eval()
		print('Loaded pretrained weights...')

		#Start the webcam and openCV
		self.capture = cv2.VideoCapture(0)
		
	def run(self):
		while(True):
			#frame = cv2.UMat(self.capture.read(), ...)
			ret, frame = self.capture.read()
			frame = self.process_frame(frame)
			cv2.imshow("Pose Detection", frame)
			if cv2.waitKey(1) & 0xFF == ord('x'):
				break;

		capture.release()
		cv2.destroyAllWindows()




	#Tests each frame against the model to find the location and draws it on the frame
	def process_frame(self, frame):
		frame = cv2.resize(frame, (self.image_size, self.image_size))
		image_height, image_width, image_channels = frame.shape
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image_patch = convert_cvimg_to_tensor(frame)

		mean = np.array([123.675, 116.280, 103.530])
		std = np.array([58.395, 57.120, 57.375])

		# apply normalization
		for n_c in range(image_channels):
		    if mean is not None and std is not None:
		        image_patch[n_c, :, :] = (image_patch[n_c, :, :] - mean[n_c]) / std[n_c]
		image_patch = torch.from_numpy(image_patch)

		preds = self.model(image_patch[None, ...])
		preds = get_joint_location_result(self.image_size, self.image_size, preds, useCuda=False)[0,:,:3]

		drawskeleton(frame, preds, thickness=2)

		return frame

test = Epipolar_Webcam()
test.run()