from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave # new
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

import pandas as pd
from numpy import *
from numpy.linalg import norm

# Cost function
def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

# Deriative of the cost function
def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)

# gradient descent to find best theta.
def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 500 == 0:
            print "Iter", iter
            print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t
    
    
dat = pd.read_csv("galaxy.data.txt")


x1 = dat.loc[:,"east.west"].as_matrix()
x2 = dat.loc[:, "north.south"].as_matrix()

x = vstack((x1, x2))
print(x.shape)

y = dat.loc[:, "velocity"].as_matrix()
print(y.shape)

theta0 = array([0., 0., 0.])
print(theta0.shape)
theta = grad_descent(f, df, x, y, theta0, 0.0000010)
#print(x.shape)
#print(x1.shape);
#print(y.shape);

#theta0 = array([0., 0., 0.])
#print(theta0.shape);

#a = array([0, 0, 0])
#print(a.shape)
# print("I am running")
# im = imread('bieber.jpg')
# show(imshow(im))
# imsave('newbieber.jpg', im)
# imshow(im)
# im.shape


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
    

# im = imread('drescher0.jpg')
# im = im[1116:2874, 1111:2869] # focus on the face
# im = rgb2gray(im) # convert to gray scale showed as heatmap.
# im = imresize(im, (32, 32)) # resize the image to 32 x 32
# imshow(im[:,:], cmap=cm.gray)
# imsave('specialtest.jpg', im)  

# act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'];
# 
# def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
#     '''From:
#     http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
#     import threading
#     class InterruptableThread(threading.Thread):
#         def __init__(self):
#             threading.Thread.__init__(self)
#             self.result = None
# 
#         def run(self):
#             try:
#                 self.result = func(*args, **kwargs)
#             except:
#                 self.result = default
# 
#     it = InterruptableThread()
#     it.start()
#     it.join(timeout_duration)
#     if it.isAlive():
#         return False
#     else:
#         return it.result
# 
# testfile = urllib.URLopener()   
# 
# for a in act:
#     name = a.split()[1].lower()
#     i = 0
#     for line in open("facescrub_actresses.txt"):
#         if a in line:
#             
#             filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
#             bound_box = line.split()[5].split(',')
#             
#             timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
#             if not os.path.isfile("uncropped/"+filename):
#                 continue
#                 
#             # Retrieve coordinate of the boundary box.
#             x1 = int(bound_box[0])
#             y1 = int(bound_box[1])
#             x2 = int(bound_box[2])
#             y2 = int(bound_box[3])
#             print(x1, x2, y1, y2)
#             
#             # Catch broken image file from uncropped folder.
#             try:
#                 img = imread("uncropped/"+filename)
#                 img = img[y1:y2,x1:x2]
#                 img = rgb2gray(img)
#                 img = imresize(img, (32, 32))
#                 imshow(img[:,:], cmap=cm.gray)
#                 imsave('cropped/' + filename, img)
#                 print(filename)
#                 i += 1
#             except:
#                 print("Couldn't read the file: "+filename)

