#======================================================================
#                           LIBRARIES
#======================================================================

# GENERAL : 
import numpy as np
import matplotlib.pyplot as plt
import pickle    #to load files
# 	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #to make prediction

# OPEN CV FUNCTIONS : 
import cv2 # tested using version 4.1.2

# SKIMAGE FUNCTIONS : 
from skimage.filters import median
from skimage.morphology import disk  
from skimage.morphology import area_opening
from skimage.measure import label     
from skimage.measure import perimeter 
from skimage.segmentation import active_contour

# KERAS LIB :
import keras #maybe not necessary, not sure how to extract the "model.predic()" method. 
from keras.models import load_model

#======================================================================
#            MAIN FUNCTION 1) : DIGIT CLASSIFIER
#======================================================================

def classify_getDigit(pic, model_path = 'CNN_model_V1.h5'):
  """
  Given a picture which represents a digit, return its digit.

  => Input :    °)  pic :  - a 2D numpy binary picture of shape (28 x 28)
                           - True = an element of the digit
                           - False = background
                °)  model_path :  - the path where the model file is stored
                                  - Note : the path must also include the name of the model file itself

  => Output :   °)  return a string caracter, either '0', '1', '2', '3', '4', '5', '6', '7' or '8'.
  """
  model = load_model(model_path) # 1) load the CNN model 
  pic = pic.reshape((1,28,28,1)) # 2) reshape as a suitable form for the CNN model
  the_pred = model.predict(pic)  # 3) make the prediction, which is of the form [ [0, 0, 0, 1, 0, 0, 0, 0, 0] ] for instance.
  the_idx = np.where(the_pred[0] == np.amax(the_pred[0])) # 4) convert to a digit, for instance [ [3] ]
  return str(int(the_idx[0]))    # 5) convert to a string caracter

#======================================================================
#            MAIN FUNCTION 2) : SYMBOL CLASSIFIER
#======================================================================

def classify_getSymbol( pic, path_classifier='symbol_classifier_V2.pk' ):
	"""
	Given a picture which represents a symbol, return its symbol.
	
	=> Input :    °)  pic :  - a 2D numpy binary picture of shape (28 x 28)
							 - True = an element of the symbol
							 - False = background
				  °)  path_classifier :  - the path where the classifier file is stored
										 - Note : the path must also include the name of the classifier file
	
	=> Output :   °)  return a string caracter, either '+', '-', '*', '=', ':'
	"""
	lda_model = pickle.load(open(path_classifier,'rb'))
	features = np.zeros((1,3))
	bend_the_knee_to_the_chosen_fourier_descriptor_feature = 5 #Heretics will burn.
	features[0,0:2] = compute_nb_objects(pic)
	features[0,2],_,_ = compute_feat_with_fourier_descriptor(pic, feat_n = bend_the_knee_to_the_chosen_fourier_descriptor_feature)
	features[0,2]*=300 #lazy wait to not normalize the data
	pred = lda_model.predict(features[:,1:3])

	if features[0,0] == 3:   # 3 points in object => Divide symbol
		return '/'
	elif features[0,0] ==2:  # 2 points in object => '=' symbol
		return '='
	elif pred[0] == 1:       # 1 point in object => use lda classifier to decide between + , - ,* symbols
		return '*'
	elif pred[0] == 2:
		return '+'
	elif pred[0] == 3:
		return '-'
	else:
		return 'STH WENT WRONG'

#======================================================================
#                 Functions : To compute the different features
#======================================================================

def get_contour(im, methode = "AC") :
	"""
	This function finds a contour in image im using either Active Contour method
	or the OpenCV native findContours function.

	INPUT :
	  im : binary image to search the contour in
	  methode : specifies methode to be used to search contour, "AC" for active
				contours, "CV2" for OpenCV's findContours(). Default is "AC"
	OUTPUT :
	  im : filtered image used for contour search
	  snake : order point list of the contour
	"""
	# Filtering 
	im = median(im, disk(2)) #maybe disk(1) is enough ??!!
	im = area_opening(im, 40)

	if methode == "AC" :      # --- Active Contours
		# Create a circle used as a seed for active contours
		s = np.linspace(0, 2*np.pi, 300)
		r = 14 + 10*np.sin(s) 
		c = 14 + 10*np.cos(s)
		init = np.array([r, c]).T  #initial snake
		snake = active_contour(im, init, alpha=4, beta=1, gamma=1,
						   w_line = 0, w_edge = 2, coordinates='rc')   
	elif methode == "CV2" : # --- Contour using OpenCV
		if np.max(im) <= 1 :
			imtmp = np.trunc(im*255)
			imtmp = imtmp.astype('uint8')
		else :
			imtmp = im
		contours, hierarchy = cv2.findContours(imtmp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		c = np.array(contours[0])
		snake = np.array([c[:,0,1],c[:,0,0]]).T  
	else :
		raise Exception("Unkown methode : {}".format(methode))
	return im, snake

def extract_fourier_descriptor(im, methode = "AC"):
	"""
	Computes fourier descriptors given a greysclale picture.

	INPUT :
	  im : grayscale image to search to contour in
	  methode : specifies methode to be used to search contour, see getContour()
				for details
	OUPUT :
	  fd : Fourier descriptor coefficients
	  snake : order point list of the contour 
	  im : filtered image used for contour search
	"""
	if type(im[0,0]) == np.bool_: #if picture is binary, convert it to CV_8UC1 type
		im = np.copy(im.astype(np.uint8))
	im, snake = get_contour(im, methode)
	z = (snake[:,0] + 1j*snake[:,1]) #put the 2D points of the snake in a complex representation
	fd = np.fft.fft(z)
	return fd, snake, im 

def compute_feat_with_fourier_descriptor( pic, feat_n = 1 ):
	"""
	Take a CV_8UC1 picture in input and return the n th feature of its Fourier Descriptor which is :
	- Translation invariant (as soon as feat_n is NOT 0)
	- Rotation invariant
	- Sclaled invariant
	"""
	descriptor, snake, im= extract_fourier_descriptor(pic, methode = 'CV2')
	module = np.absolute(descriptor) #compute module : rotation invariant
	module/=module[0] #make it sclale invariant
	return module[feat_n],snake, im

def compute_nb_objects(pic):
	"""
	It returns the number of objects in the picture pic (background does NOT count as an object)
	It also returns the total area of the objects (i.e. the sum of the area of each object, i.e. the area which is NOT background)
	
	input : The input picture must be a 2D binary numpy array ! 
	"""
	if pic.shape[-1] == 4 or pic.shape[-1] == 3:
		print("\n Error : This picture is RGB or RBBA, hence NOT binary. Begone.")
	elif len(pic.shape) ==2: 
		pass
	else:
		print("\n Error : This picture is NOT binary. Begone.")
		
	labels = label(pic)
	return np.amax(labels), np.sum(labels==True)