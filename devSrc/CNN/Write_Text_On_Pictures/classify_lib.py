import tarfile
import os
import skimage.io

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image #to write text on pictures

#method 1 :
import imageio #to load png pictures
from skimage import img_as_ubyte
from skimage import filters
from skimage.color import rgb2gray
from skimage.color import rgba2rgb
#from skimage import filters
from skimage.filters import rank
from skimage.filters import median
from skimage.filters import gaussian
from skimage.filters import sobel
from skimage.morphology import disk
from skimage.morphology import erosion
from skimage.morphology import closing
from skimage.morphology import opening
from skimage.morphology import rectangle
from skimage.morphology import area_closing
from skimage.morphology import area_opening

from skimage.draw import rectangle
#from skimage.draw import circle
from skimage.measure import label


from skimage.segmentation import watershed
from skimage.segmentation import active_contour

#method 2 :
# from skimage.util import img_as_ubyte
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


import warnings 
import cv2 # tested using version 4.1.2
#if cv2.__version__ != "4.1.2" :
#  warnings.warn("OpenCV currently running version {}, developement version is 4.1.2".format(cv2.__version__))

#======================================================================
# To compute Fourier Descriptor
#======================================================================
ONE_OBJECT_SIGNS = (PLUS, MINUS, TIMES) = 0,2,4  #the index of the + - and * symbols

#======================================================================
# To compute Fourier Descriptor
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
    im = median(im, disk(2))
    im = area_opening(im, 40)

    # Active Contours
    if methode == "AC" :
        # Create a circle used as a seed for active contours
        s = np.linspace(0, 2*np.pi, 300)
        r = 14 + 10*np.sin(s) 
        c = 14 + 10*np.cos(s)
        init = np.array([r, c]).T  #initial snake

        snake = active_contour(im, init, alpha=4, beta=1, gamma=1,
                           w_line = 0, w_edge = 2, coordinates='rc')
        
    # Contour using OpenCV
    elif methode == "CV2" :
        contours, hierarchy = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
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

#======================================================================
# To compute nb_objects per picture and total area of picture
#======================================================================

def compute_nb_objects(pic, thresh = 0.9, interval = 0.2):
    """
    return nb_object in the picture (background does NOT count as an object)
    also return the total area of the objects
    
    input : the picture is composed of pixel with values between 0 and 1 !!!
    
    """
    if pic.shape[-1] == 4:
        im_gray = np.copy(rgb2gray(rgba2rgb(im)))
    elif pic.shape[-1] == 3:
        im_gray = np.copy(rgb2gray(pic))  # convert to grayscale
    elif len(pic.shape) ==2: 
        im_gray = np.copy(pic)
    else:
        "\n This picture is neither RBGA nor RGB nor greyscale"
        
    denoised = rank.median(img_as_ubyte(im_gray), disk(1))  #denoise image
    # define markers for wateshed : one small rectagle to define background
    # and markers where there is no background. The background is defined as an interval around the mean color of the image
    markers = np.zeros_like(im_gray)
    
                    #!!! maybe use Kmeans to classify the pixels in two classes : background from symbols ? 
    #thresh = 0.9    #!!! np.average(im_gray)
    #interval = 0.2  #!!! to adjust probably !!!
    markers[im_gray<thresh-interval] = 1
    markers[im_gray>thresh+interval] = 1
    area = np.sum(markers==1)
    rr, cc = rectangle((0,0), extent=(20,20), shape=markers.shape) #maybe to adjust the shape too
    markers[rr,cc] = 1    
    markers = label(markers)

    edge_sobel = filters.sobel(denoised)     # sobel edges
    labels = watershed(edge_sobel, markers)  # perform the watershed

    nb_obj = np.max(labels)-1
    return nb_obj, area