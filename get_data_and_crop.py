# This program is used to download the image files and crop them according to the
# requirement. The uncropped images are stored in directory "uncropped". Cropped 
# images are stored in directory "cropped". Pleased these two folders exists in
# the directory before running the program.

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib


# helper function for converting to gray scale.
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return gray/255.
    
    
act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'];


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None


        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default


    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


testfile = urllib.URLopener()   


# The uncropped images are stored in uncropped directory. Cropped images are 
# stored in cropped directory.
def get_data_and_crop():
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open("facescrub_actors.txt"):
            if a in line:
                
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                bound_box = line.split()[5].split(',')
                
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                    
                # Retrieve coordinate of the boundary box.
                x1 = int(bound_box[0])
                y1 = int(bound_box[1])
                x2 = int(bound_box[2])
                y2 = int(bound_box[3])
                print(x1, x2, y1, y2)
                
                # Some uncropped images are broken and cannot be processed.
                try:
                    img = imread('uncropped/'+filename)
                    img = img[y1:y2,x1:x2]
                    img = rgb2gray(img)
                    img = imresize(img, (32, 32))
                    imsave('cropped/' + filename, img, cmap = cm.gray)
                    print(filename)
                except:
                    print("Couldn't process the file: "+filename)
                
                # Makes the number in the name of cropped image match
                # with the number in the name of  uncropped images. So i
                # is still incremented by 1 incase the uncropped image 
                # could not be processed.
                i = i + 1
                
if not os.path.exists("cropped/"):
    os.makedirs("cropped/")
    
if not os.path.exists("uncropped/"):
    os.makedirs("uncropped/")
    
get_data_and_crop()