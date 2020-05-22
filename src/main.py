#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from video import video_init, video_load, video_showFrame, video_export
from segment import Segment
from Equation import Equation
from overlay import draw_traj_on_pic, write_text_on_pic

import sys
import pickle
import argparse


# Global variable containing programme input parameters
# args.input : input video
# args.output : output video
# args.verbose : bool 
args = None

def parsInput() :
	global args
	parser = argparse.ArgumentParser(description='Deliverable for IAPR 2020 project.')

	defaultInput = os.path.join('..','data','robot_parcours_1.avi')
	defaultOuput = os.path.join('..','data','out','robot_parcours_1_out.avi')
	parser.add_argument('-i','--input', default=defaultInput, help='Input video clip, should be .avi')
	parser.add_argument('-o','--output', default=defaultOuput, help='output video clip (path and name), should be .avi')
	parser.add_argument('-v','--verbose', action='store_true', default=False, help='Makes processing verbose and displays intermediate figures (execution stops when a figure is open)')

	args = parser.parse_args()


def main():
	parsInput()

	global args
	imgsIn = video_load(args.input)
	args.imgShape = imgsIn[0].shape
	video_init(args) # Video module is initialized after first use since video_load is needed to get shape
	print("INFO : Loading video done")

	seg = Segment(args)
	listObj = seg.getObj(imgsIn[0])
	print("INFO : Digit and operator segmentation done")

	eq = Equation(listObj, args)

	listPos = seg.getArrow(imgsIn)
	print("INFO : Arrow segmentation done")

	imgsOut = []
	for i,im in enumerate(imgsIn) :
		eqTemp = eq.newRobPos(listPos[i])
		imgTmp = draw_traj_on_pic(im, listPos[0:i+1])
		imgTmp = write_text_on_pic(imgTmp, text = "Frame {:02d} : {}".format(i,eqTemp), text_size = 150, text_pos = (10,10), text_color = 'red', 
                      background_color = 'white', background_dim = [(0,0), (200,30)])
		imgsOut.append(imgTmp)
	print("INFO : Image overlay done")

	video_export(args.output, imgsOut, False)
	print("INFO : Video export done")
	print("INFO : DONE")

if __name__ == '__main__':
	main()