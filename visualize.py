import matplotlib.pyplot as plt
import numpy as np
import cv2

def crop(image):
    img = image[60:136,0:image.shape[1],:]
    return cv2.resize(img, (64, 64), cv2.INTER_AREA)

def flip_img(image, steering):
	""" randomly flip image to gain right turn data (track1 is biaed in left turn) 
		source: https://github.com/windowsub0406/Behavior-Cloning/blob/master/model.py#L89"""
	flip_image = image.copy()
	flip_steering = steering
	num = np.random.randint(2)
	if num == 0:
	    flip_image, flip_steering = cv2.flip(image, 1), -steering
	return flip_image, flip_steering

def brightness_img(image):
	"""
	randomly change brightness by converting Y value
	source: https://github.com/windowsub0406/Behavior-Cloning/blob/master/model.py
	"""
	br_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	coin = np.random.randint(2)
	if coin == 0:
	    random_bright = 0.2 + np.random.uniform(0.2, 0.6)
	    br_img[:, :, 2] = br_img[:, :, 2] * random_bright
	br_img = cv2.cvtColor(br_img, cv2.COLOR_HSV2RGB)
	return br_img

def shift_img(image, steer):
	""" shift image randomly
		source: https://github.com/windowsub0406/Behavior-Cloning/blob/master/model.py """
	max_shift = 55
	max_ang = 0.14  # ang_per_pixel = 0.0025

	rows, cols, _ = image.shape

	random_x = np.random.randint(-max_shift, max_shift + 1)
	dst_steer = steer + (random_x / max_shift) * max_ang
	if abs(dst_steer) > 1:
	    dst_steer = -1 if (dst_steer < 0) else 1

	mat = np.float32([[1, 0, random_x], [0, 1, 0]])
	dst_img = cv2.warpAffine(image, mat, (cols, rows))
	return dst_img, dst_steer


img = plt.imread("center_2017_02_25_18_35_56_238.jpg")
angle =0.0
print("Original")
#plt.imshow(img)
#plt.show()


print("After shift")
image, angle = shift_img(img, angle)
plt.imshow(image)
plt.show()


print("After flip")
image, angle = flip_img(image, angle)
plt.imshow(image)
plt.show()


print("After brightness")
image = brightness_img(image)
plt.imshow(image)
plt.show()

image = crop(image)
print("After crop and resize")
plt.imshow(image)
plt.show()


