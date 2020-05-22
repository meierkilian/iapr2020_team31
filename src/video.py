# conda install av -c conda-forge

import av
import numpy as np
import cv2
from cv2 import cvtColor
import matplotlib.pyplot as plt
from PIL import Image
import os

# Global variable containing programme input parameters
# args.input : input video
# args.output : output video
# args.verbose : bool 
args = None

def video_init(_args) :
	global args
	args = _args


def video_load(path):
	v = av.open(path)

	imgs = []

	for packet in v.demux():
		for frame in packet.decode():
			#if frame.type == 'video':
			img = frame.to_image()  # PIL/Pillow image
			arr = np.asarray(img)  # numpy array
			imgs.append(arr)

	return imgs

def video_showFrame(img):
	plt.imshow(img)
	plt.axis('off')
	plt.show()
	
	
'''
	function name: video_export
	description: this function outputs a video from a numpy array
	
	input:
	- output [string]: path and name of the exported video
	- imgs [np.array]: array of the modified images
	- savePng [bool]: if true, keeps individual frame as png in a folder next to output
	
	output:
	- png pictures of each frame to given png_path
	- final exported video to given video_path
'''
def video_export(output, imgs, savePng):

	# Parsing output file path and name
	outputSplit = os.path.split(output)
	video_name = outputSplit[1]
	video_path = outputSplit[0]
	png_path = os.path.join(video_path, 'png_' + video_name)
	
	# png folder exists emties it, else create folder
	if os.path.exists(png_path) :
		for file in os.scandir(png_path) :
			if file.is_file :
				os.remove(file.path)
	else :
		os.mkdir(png_path)



	#Boucle pour sauver les png.
	i = 0;
	for im in imgs:
		new_im = Image.fromarray(im)
		new_im.save(os.path.join(png_path,"frame%.3d.png" % i))
		i = i+1
		
	#Boucle pour sauver la vidéo
	freq = 2

	listFrame = os.listdir(png_path)
	listFrame.sort()
	images = [img for img in listFrame if img.endswith(".png")]
	frame = cv2.imread(os.path.join(png_path, images[0]))
	height, width, layers = frame.shape

	video = cv2.VideoWriter(output, 0, freq, (width,height))

	for image in images:
		video.write(cv2.imread(os.path.join(png_path, image)))

	cv2.destroyAllWindows()
	video.release()

	# Emptying and removing png folder
	if not savePng :
		for file in os.scandir(png_path) :
			if file.is_file :
				os.remove(file.path)
		os.rmdir(png_path)