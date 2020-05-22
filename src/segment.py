import matplotlib.pyplot as plt

from skimage import morphology
from skimage.segmentation import watershed, random_walker
from skimage.draw import rectangle
from skimage.draw import circle
from skimage.measure import label
from skimage import color 
from skimage import filters
from skimage import feature
from skimage.util import crop
from skimage.util import img_as_ubyte
from skimage.transform import resize
from skimage import exposure

import numpy as np
import cv2
from cv2 import cvtColor

originalShape = (480,720)

class Segment(object):
    """Class performing object segmentation (arrow, operator and number)"""
    def __init__(self, args):
        # Global variable containing programme input parameters
        # args.input : input video
        # args.output : output video
        # args.verbose : bool 
        # args.imgShape : shape of the image to be worked with
        self.args = args 
        paramCorr = int(self.args.imgShape[0]/originalShape[0])
        # 
        # Max distance objects can be appart to be considered as belonging to the same symbol
        self.maxDist = 40*paramCorr
        # 
        # Max/Min size an object is allowed to be to be considered as a symbol
        self.maxSize = 300*paramCorr
        self.minSize = 40*paramCorr
        #
        # Black border to add around a symbol, if object has size r then extracted image has size int(r*border)
        self.border = 1.4
        #
        # Shape of desired output images in pixels
        self.outputSize = (28,28)
        #
        # If True all the symbols are rescaled to match outputSize, if False the images are cropped directly to outputSize,
        # this may result in different scale or symbol beeing outside of the image.
        self.adjustSize = True
        #
        # Radius or red arrow (TODO estimate this param)
        self.arrowRad = 70*paramCorr

    def getObjCenter(self, img) :
        x = []
        y = []
        for i in range(img.shape[0]) :
            for j in range(img.shape[1]) :
                if img[i,j] :
                    x.append(i)
                    y.append(j)

        meanX = np.round(np.average(x))
        meanY = np.round(np.average(y))
        return (meanX, meanY)


    def getObjRadius(self, img, center) :
        dmax = 0
        for i in range(0,img.shape[0]) :
            for j in range(0, img.shape[1]) :
                if img[i,j] :
                    d = np.linalg.norm(np.subtract((i,j), center))
                    if d > dmax :
                        dmax = d
        return dmax


    def cropCenter(self, img, center, radius) :
        cropLim = (np.clip((center[0] - radius, img.shape[0] - center[0] - radius), 0, img.shape[0]), \
                    np.clip((center[1] - radius, img.shape[1] - center[1] - radius), 0, img.shape[1]))
        return crop(img, cropLim, copy = True)


    def getObj(self,img) :

        # convert to grayscale
        im_gray = img_as_ubyte(color.rgb2gray(img))
        
        # Equalization
        img_gray = filters.rank.equalize(im_gray, selem=morphology.disk(30))
        
        # First marker label approximation 
        thresh = filters.threshold_otsu(im_gray)
        markers = im_bin = im_gray < thresh
        markers = morphology.opening(im_bin*255, selem=morphology.disk(1)) # removes some false positives, in particular table edge
        # markers = morphology.binary_erosion(markers, selem=morphology.disk(1))
        markers = label(markers)

        
        # Merge labels which are close to each other 
        objMeanPos = {}
        for i in range(1,np.max(markers)+1) :
            objMeanPos[i] = self.getObjCenter(markers == i)

        prevMarkers = np.array(markers)

        for i in range(1, np.max(np.max(markers))+1) :
            for j in range(i, np.max(np.max(markers))+1) :
                if np.linalg.norm(np.subtract(objMeanPos[i], objMeanPos[j])) < self.maxDist :
                    markers[markers == j] = i

        while (prevMarkers != markers).any() :
            prevMarkers = np.array(markers)
            for i in range(1, np.max(np.max(markers))+1) :
                for j in range(i, np.max(np.max(markers))+1) :
                    if np.linalg.norm(np.subtract(objMeanPos[i], objMeanPos[j])) < self.maxDist :
                        markers[markers == j] = i        


        # Removes ojects that are to big or to small
        objSize = {}
        for i in range(1,np.max(markers)+1) :
            objSize[i] = np.sum(markers==i)
            if objSize[i] > self.maxSize or objSize[i] < self.minSize:
                markers[markers == i] = 0

        # Removes arrow related objects
        arrowPos = self.getArrow([img])[0].astype(int)
        for i in range(-self.arrowRad, self.arrowRad) :
            for j in range(-self.arrowRad, self.arrowRad) :
                x = np.clip(arrowPos[0] + i, 0, img.shape[0]-1)
                y = np.clip(arrowPos[1] + j, 0, img.shape[1]-1)
                markers[x,y] = 0


        if self.args.verbose :
            # display results
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 12),
                                     sharex=True, sharey=True)
            ax = axes.ravel()

            ax[0].imshow(img)
            ax[0].set_title("Original")

            ax[1].imshow(im_gray, cmap=plt.cm.gray, alpha=1)
            ax[1].set_title("Gray and Equalized")

            ax[2].imshow(markers, cmap=plt.cm.nipy_spectral, alpha=1)
            ax[2].set_title("Object markers")
            
            ax[3].imshow(img, cmap=plt.cm.nipy_spectral, alpha=1)
            ax[3].set_title("Other")
            

            for a in ax:
                a.axis('off')

            fig.tight_layout()
            plt.show()

        # Extract symbol list from original image
        listObj = []
        for l in np.unique(markers) :
            if l == 0:
                continue ## Ignoring background

            obj = {}
            obj["pos"] = self.getObjCenter(markers == l)
        
            if self.adjustSize :
                r = int(np.round(self.getObjRadius(markers == l, obj["pos"]))*self.border)
                imgResized = resize(self.cropCenter(1 - im_gray, obj["pos"], r), self.outputSize, preserve_range = True)
            else :
                imgResized = self.cropCenter(1 - im_gray, obj["pos"], int(self.outputSize[0]/2))

            # Thresholding only the subimage around the object
            thresh = filters.threshold_otsu(imgResized)
            obj["img"] = imgResized > thresh
            listObj.append(obj)

        
        if self.args.verbose :
            fig, axes = plt.subplots(nrows=3, ncols=int(np.ceil(len(listObj)/3)), figsize=self.outputSize, \
            sharex=True, sharey=True)

            axes = axes.ravel()
            
            for obj, ax in zip(listObj, axes) :
                ax.imshow(obj["img"], cmap=plt.cm.gray)
                ax.set_title("Center\n {}".format(obj["pos"]))

            for a in axes:
                a.axis('off')

            plt.show()

        return listObj


    def getArrow(self, video):
        
        arrow_pos=np.zeros((len(video),2))
        i = 0
        thr_min_2 = 0;
        thr_min_1 = 0;
        for im in video:
            #ranges of y, u, v:
            # Y: 0 - 255
            # U: -128 - 127
            # V: -128 - 127
            img_yuv = cvtColor(im, cv2.COLOR_BGR2YUV);
            img_yuv = img_yuv[:,:,1];
            img_bin = np.zeros(video[0][:,:,0].shape);
            
            #selection of threshold with redundancy
            if i==0:
                thr_min_2 = thr_min_1 = thresh = filters.threshold_otsu(img_yuv);
            else:
                thr_min_2 = thr_min_1;
                thr_min_1 = thresh;
                thr_mean = np.mean([thr_min_2, thr_min_1]);
                thresh = filters.threshold_otsu(img_yuv);
                if abs(thresh-thr_mean)>5: #5 = empirically chosen threshold
                    thresh = thr_mean;
            
            img_bin[np.where(img_yuv>thresh)] = 1

            M = cv2.moments(img_bin);
            cY = int(M["m10"] / M["m00"])
            cX = int(M["m01"] / M["m00"])
            arrow_pos[i,:] = [cX,cY];
            i = i+1

        if False and self.args.verbose :
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.set_xlim(0,720)
            ax.set_ylim(480,0)
            plt.imshow(video[0])
            plt.scatter(arrow_pos[:,1], arrow_pos[:,0])
            plt.title("Arrow position")
            plt.show()
            

        return arrow_pos

