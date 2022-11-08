import cv2
import numpy as np
from rknn.api import RKNN
from rknn_server_class import rknn_server
from timeit import default_timer as timer
import os
import time

model = './upsample_Epoch_98_cpu_QUA_bs100.rknn'


IMG_SAVEDIR = './imgs_noth/'
if not os.path.exists(IMG_SAVEDIR):
	os.makedirs(IMG_SAVEDIR)


def mask_post_process(output):
	# output dim, tensor OR numpy
	# output is list

	# print(output[0].shape)
	# for i in range(output[0].shape[1]):
	# 	if np.squeeze(output[0])[i] > 0.5:
	# 		print(i, output[0][:,i])

	IMG_SIZE = 112

	np_output = np.squeeze(np.array(output[0])).reshape(IMG_SIZE, IMG_SIZE)

	# np_output_debug = cv2.resize(np_output, (IMG_SIZE * 2, IMG_SIZE * 2))

	# time_str = time.strftime('%Y%m%d_%H%M%S')
	# cv2.imwrite(os.path.join(IMG_SAVEDIR, '%s.jpg'%time_str), np.uint8(np_output_debug > 0.5) * 255)
	
	np_output[:,:2] = 0
	# np_output = cv2.resize(np_output, (IMG_SIZE * 2, IMG_SIZE * 2))

	print(np.max(np_output), np.min(np_output))
	print('output shape')
	print(np_output.shape)

	output = np.uint8(np_output > 0.5)

	retval, labels, stats, centroids = cv2.connectedComponentsWithStats(output, connectivity=8)  # connectivity default 8
	stats = stats[stats[:, 4].argsort()]
	bboxs = stats[:-1][:, :-1]
	classes = np.array([0] * len(bboxs))
	scores = np.array([0.9] * len(bboxs))

	print(bboxs)
	print(classes)
	print(scores)
	return bboxs, classes, scores

if __name__ == '__main__':
	rknn = rknn_server(8002)
	rknn.service(model, mask_post_process)