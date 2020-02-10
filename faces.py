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
from auto_pick import *
from auto_pick_part5 import *
import shutil

act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'];

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



# -----------------------------------------------------------------------------
# PART 2
# -----------------------------------------------------------------------------
def reproduce_part_2():
    test_dir = "part2_data/test_data"
    train_dir = "part2_data/train_data"
    valid_dir = "part2_data/valid_data"
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(valid_dir):
        shutil.rmtree(valid_dir)
    
    # Create folders for part 2.
    os.makedirs(test_dir)
    os.makedirs(train_dir)
    os.makedirs(valid_dir)
    
    # Non-overllaping sets.
    pickrandom(act, "cropped/", train_dir, valid_dir, test_dir);
    
#reproduce_part_2()
    

# -----------------------------------------------------------------------------
# PART 3
# -----------------------------------------------------------------------------
# Cost function
def f(x, y, theta, train_size):
    x = vstack( (ones((1, x.shape[1])), x))

    return (1/float(2*train_size)) * sum( (y - dot(theta.T,x)) ** 2)

# Deriative of the cost function
def df(x, y, theta, train_size):
    x = vstack( (ones((1, x.shape[1])), x))
    print ((y-dot(theta.T, x))*x).shape
    return float(-1/train_size) *sum((y-dot(theta.T, x))*x, 1)

# gradient descent to find best theta.
def grad_descent(f, df, x, y, init_t, alpha, train_size):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t, train_size)
        if iter % 500 == 0:
            print "Iter", iter
            print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t, train_size)) 
            print "Gradient: ", df(x, y, t, train_size), "\n"
        iter += 1
    return t
    
# Part 3 classifier.
# train_dir: relative path to the directory containing images for training.
# test_dir: relative path to directory containing images for evaluating.
def classifier_part3(train_dir, test_dir):
    #------------Building the classifier from training data-----------------#
    train_data = os.listdir(train_dir)
    x = array([])
    y = zeros((200,), dtype=np.int) # 200 = size of the training set
    i = 0;
    train_size = 0;
    for name in train_data: 
        if "carell" in name or "hader" in name:
            train_size = train_size + 1;
            img = imread(train_dir + name)
            img = rgb2gray(img)
            img = img.flatten()
            if x.size == 0:
                x = vstack([img])
            elif x.size > 0:
                x = vstack([x, img])
            if "hader" in name: # hader is 1, carell is 0
                y[i] = 1
            
            i = i + 1;
    x = x.T
    theta0 = np.zeros(1025)
    theta = grad_descent(f, df, x, y, theta0, 0.000001, train_size)
    
    
    #--------------Running the classifier to obtain the result---------------#
    test_data = os.listdir(test_dir);
    count_hader = 0; # Number of correct classification for hader.
    count_carell = 0; # Number of correct classification for carell.
    test_size = 0;
    for name in test_data:
        if "carell" in name or "hader" in name:
            test_size = test_size + 1;
            imgtest = imread(test_dir + name)
            imgtest = rgb2gray(imgtest)
            imgtest = imgtest.flatten()
            result = dot(theta[1:], imgtest) + theta[0] # Hypotheis result
            # print("Hypothesis result: ", result)
            if result > 0.5:
                print('The image {} is classified to be {}'.format(name,'Bill Hader'))
                if "hader" in name: # classification result matches the label
                    count_hader = count_hader + 1;
            elif result <= 0.5:
                print('The image {} is classified to be {}'.format(name,'Steve Carell'))
                if "carell" in name: # classification result matches the label
                    count_carell = count_carell + 1;
    print((float(count_hader + count_carell)/test_size)*100); # Percentage result
    
    
def reproduce_part_3():
    test = "part3_data/test"
    train = "part3_data/train"
    validation = "part3_data/validation"
    
    if os.path.exists(test):
        shutil.rmtree(test)
    
    if os.path.exists(train):
        shutil.rmtree(train)
    
    if os.path.exists(validation):
        shutil.rmtree(validation)
    
    
    os.makedirs(test);
    os.makedirs(train);
    os.makedirs(validation);
    
    # Non-overllaping sets.
    pickrandom(act, "cropped/", "part3_data/train", "part3_data/validation", "part3_data/test");
    
    train_with = "part3_data/train/"; 
    test_on1 = "part3_data/train/";
    test_on2 = "part3_data/validation/"
    test_on3 = "part3_data/test/"
    print("Performance on training set")
    classifier_part3(train_with, test_on1)
    print("Performance on validation set")
    classifier_part3(train_with, test_on2)
    print("Performance on test set")
    classifier_part3(train_with, test_on3)
    
#reproduce_part_3()



# -----------------------------------------------------------------------------
# PART 4
# -----------------------------------------------------------------------------
# train_dir: relative path to the directory containing images for training.
# train_size: size of the training set.
def display_theta_part4(train_dir, train_size):
    #------------Building the classifier from training data-----------------#
    train_data = os.listdir(train_dir) # folder stores set of images
    x = array([])
    y = zeros((train_size,), dtype=np.int) # Either 4 or 200 = size of the training set.
    i = 0;
    train_size = 0;
    for name in train_data: 
        if "carell" in name or "hader" in name:
            train_size = train_size + 1;
            img = imread(train_dir + name) # folder stores set of images
            img = rgb2gray(img)
            img = img.flatten()
            if x.size == 0:
                x = vstack([img])
            elif x.size > 0:
                x = vstack([x, img])
            if "hader" in name:
                y[i] = 1
            
            i = i + 1;
    x = x.T
    theta0 = np.zeros(1025)
    theta = grad_descent(f, df, x, y, theta0, 0.000001, train_size)
    
    # Display the theta image.
    img_theta = theta[1:].reshape((32,32))
    show(imshow(img_theta[:,:]))


def reproduce_part_4():
    act_part4 = ['Bill Hader', 'Steve Carell'];
    if not os.path.exists("part4_data/train_4"):
        os.makedirs("part4_data/train_4")
        pickrandom_p5(act_part4, "cropped/", "part4_data/train_4/", "", "", 2, 0, 0);
    if not os.path.exists("part4_data/train_200"):
        os.makedirs("part4_data/train_200")
        pickrandom_p5(act_part4, "cropped/", "part4_data/train_200/", "", "", 100, 0, 0);

    train_with1 = "part4_data/train_4/"
    train_with2 = "part4_data/train_200/"
    # display_theta_part4(train_with1, 4);
    display_theta_part4(train_with2, 200);

# reproduce_part_4()


# -----------------------------------------------------------------------------
# PART 5
# -----------------------------------------------------------------------------
# gradient descent to find best theta.
def grad_descent_p5(f, df, x, y, init_t, alpha, train_size):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t, train_size)
        iter += 1
    return t


# Part 5.1 - gender classifier for actors in act.
# train_dir: relative path to the directory containing images for training.
# test_dir: relative path to directory containing images for evaluating.
# train_size: size of the training set.
def male_female_classifier(train_dir, test_dir, train_size):
    #------------Building the classifier from training data-----------------#
    train_data = os.listdir(train_dir)
    x = array([])
    y = zeros((train_size,), dtype=np.int) # the parameter = size of the training set.
    i = 0;
    train_size = 0;
    for name in train_data: 
        # Only read files with these names.
        if (("carell" in name) or ("hader" in name) or ("baldwin" in name) or 
            ("drescher" in name) or ("ferrera") in name or ("chenoweth") in name):
            train_size = train_size + 1;
            img = imread(train_dir + name)
            img = rgb2gray(img)
            img = img.flatten()
            if x.size == 0:
                x = vstack([img])
            elif x.size > 0:
                x = vstack([x, img])
            # "hader", "carell", and "baldwin" are male actors, and have their
            # y values set to 1. The female actors have their y = 0.
            if "hader" in name or "carell" in name or "baldwin" in name: 
                y[i] = 1
            i = i + 1;
    
    x = x.T
    theta0 = np.zeros(1025)
    theta = grad_descent_p5(f, df, x, y, theta0, 0.000001, train_size)
    
    
    #--------------Running the classifier to obtain the result---------------#
    test_data = os.listdir(test_dir);
    count_male = 0; # Number of correct classification for Male actors
    count_female = 0; # Number of correct classification for 
    test_size = 0;
    # name is the name of the files, containing name of the actor.
    for name in test_data:
        # Only read files with these names.
        if (("carell" in name) or ("hader" in name) or ("baldwin" in name) or 
            ("drescher" in name) or ("ferrera") in name or ("chenoweth") in name):
            test_size = test_size + 1;
            imgtest = imread(test_dir + name)
            imgtest = rgb2gray(imgtest)
            imgtest = imgtest.flatten()
            result = dot(theta[1:], imgtest) + theta[0] # Hypotheis result
            # print("Hypothesis result: ", result)
            if result > 0.5: # Male
                # print('The image {} is classified to be {}'.format(name,'Male'))
                if ("carell" in name) or ("hader" in name) or ("baldwin" in name): # classification result matches the label
                    count_male = count_male + 1;
            elif result <= 0.5: # Female
                # print('The image {} is classified to be {}'.format(name,'Female'))
                if  ("drescher" in name) or ("ferrera") in name or ("chenoweth") in name: # classification result matches the label
                    count_female = count_female + 1;
    # print((float(count_male + count_female)/test_size)*100); # Percentage result
    return (float(count_male + count_female)/test_size)*100
 
# Part 5.2 - gender classifier for actors in act_test
# train_dir: relative path to the directory containing images for training.
# test_dir: relative path to directory containing images for evaluating.
# train_size: size of the training set.
def male_female_classifier2(train_dir, test_dir, train_size):
    #------------Building the classifier from training data-----------------#
    train_data = os.listdir(train_dir)
    x = array([])
    y = zeros((train_size,), dtype=np.int) # the parameter = size of the training set.
    i = 0;
    train_size = 0;
    for name in train_data: 
        # Only read files with these names.
        if (("carell" in name) or ("hader" in name) or ("baldwin" in name) or 
            ("drescher" in name) or ("ferrera") in name or ("chenoweth") in name):
            train_size = train_size + 1;
            img = imread(train_dir + name)
            img = rgb2gray(img)
            img = img.flatten()
            if x.size == 0:
                x = vstack([img])
            elif x.size > 0:
                x = vstack([x, img])
            # "hader", "carell", and "baldwin" are male actors, and have their
            # y values set to 1. The female actors have their y = 0.
            if "hader" in name or "carell" in name or "baldwin" in name: 
                y[i] = 1
            i = i + 1;
    
    x = x.T
    theta0 = np.zeros(1025)
    theta = grad_descent_p5(f, df, x, y, theta0, 0.000001, train_size)
    
    
    #--------------Running the classifier to obtain the result---------------#
    test_data = os.listdir(test_dir);
    count_male = 0; # Number of correct classification for Male actors
    count_female = 0; # Number of correct classification for 
    test_size = 0;
    # name is the name of the files, containing name of the actor.
    for name in test_data:
        # Only read files with these names.
        if (("butler" in name) or ("radcliffe" in name) or ("vartan" in name) or 
            ("bracco" in name) or ("gilpin") in name or ("harmon") in name):
            test_size = test_size + 1;
            imgtest = imread(test_dir + name)
            imgtest = rgb2gray(imgtest)
            imgtest = imgtest.flatten()
            result = dot(theta[1:], imgtest) + theta[0] # Hypotheis result
            # print("Hypothesis result: ", result)
            if result > 0.5: # Male
                print('The image {} is classified to be {}'.format(name,'Male'))
                if ("butler" in name) or ("radcliffe" in name) or ("vartan" in name): # classification result matches the label
                    count_male = count_male + 1;
            elif result <= 0.5: # Female
                print('The image {} is classified to be {}'.format(name,'Female'))
                if  ("bracco" in name) or ("gilpin") in name or ("harmon") in name: # classification result matches the label
                    count_female = count_female + 1;
    print((float(count_male + count_female)/test_size)*100); # Percentage result
    


def reproduce_part_5():
    print("-----------------------Reproduce part 5.1-------------------------")
    
    if os.path.exists("part5_data"):
        shutil.rmtree("part5_data")
    
    os.makedirs("part5_data/train_data_2")
    os.makedirs("part5_data/valid_data_for_2")
    pickrandom_p5(act, "cropped/", "part5_data/train_data_2/", "part5_data/valid_data_for_2", "",
                    2, 10, 0);
                    

    os.makedirs("part5_data/train_data_5")
    os.makedirs("part5_data/valid_data_for_5")
    pickrandom_p5(act, "cropped/", "part5_data/train_data_5/", "part5_data/valid_data_for_5", "",
                    5, 10, 0);

    i = 10
    while i <= 80:

        os.makedirs("part5_data/train_data_" + str(i))
        os.makedirs("part5_data/valid_data_for_" + str(i))
        # Randomly pick i images for each actor in act.
        pickrandom_p5(act, "cropped/", "part5_data/train_data_" + str(i), "part5_data/valid_data_for_" + str(i), "",
                    i, 10, 0);
        i = i + 10

    
    valid_dir = "part5_data/valid_data_for_2/";
    train_dir = "part5_data/train_data_2/";
    result_1 = male_female_classifier(train_dir, train_dir, 2 * 6)
    result_2 = male_female_classifier(train_dir, valid_dir, 2 * 6)
    print("Train size: {} images per actor: training set: {} %, validation set {} %".format( 2, result_1, result_2))
    
    
    train_dir = "part5_data/train_data_5/";
    valid_dir = "part5_data/valid_data_for_5/";
    result_1 = male_female_classifier(train_dir, train_dir, 5 * 6)
    result_2 = male_female_classifier(train_dir, valid_dir, 5 * 6)
    print("Train size: {} images per actor: training set: {} %, validation set {} %".format( 5, result_1, result_2))
    
    i = 10
    while i <= 80:
        train_dir = "part5_data/train_data_" + str(i) + "/";
        valid_dir = "part5_data/valid_data_for_" + str(i) + "/"
        result_1 = male_female_classifier(train_dir, train_dir, i * 6)
        result_2 = male_female_classifier(train_dir, valid_dir, i * 6)
        print("Train size: {} images per actor: training set: {} %, validation set {} %".format( i, result_1, result_2))
        i = i + 10
    
    print("Use that data to plot graph on Microsoft Excel, professor accepted this method according to Piazza@276.");
    print("-----------------------Reproduce part 5.2-------------------------")
    test_on = "act_test_cropped";
    # act_test_cropped contained cropped faces of the actors in act_test list.
    # Please run get_data_and_crop_part5 to get that folder, if you haven't done so.
    male_female_classifier2("part5_data/train_data_70/", "act_test_cropped/", 70 * 6)
    
# reproduce_part_5()




# -----------------------------------------------------------------------------
# PART 6
# -----------------------------------------------------------------------------
# part 6(c) codes

# Cost function
def cf_part6(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    print("shape thetaT:{}".format(theta.T.shape))
    print("shape x :{}".format(x.shape))
    return sum((dot(theta.T,x) - y) ** 2)

# Vectorized gradient function
def vectorized_gradient(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    return dot(dot(2,x), (dot(theta.T, x) - y).T)
    
    
# part 6(d) codes
# Computing the component (p, q) of the gradient matrix. 
def finite_difference_gradient(x, y, theta, p, q):
    x = vstack((ones((1, x.shape[1])), x))
    sum = 0;
    for i in range(x.shape[1]):
        sum = sum + x[p][i] * (dot(theta.T[q],x[:,i]) - y[q][i]);
    return 2 * sum;


def reproduce_part_6():
    # say each images has 5 pixels (not include 1)
    # 2 images [1,1, 1, 1,5] ,[2, 1, 1, 1, 1]
    # 3 possible labels [1, 0, 0], [0, 1, 0], [0, 0, 1]
    # x = (5 + 1) x 2 matrix
    # y = 3 x 2 matrix
    # theta.T = 3 x (5 + 1) matrix
    
    # Form an x
    img1 = array([1,1, 1, 1,5])
    img2 = array([2, 1, 1, 1, 1])
    img = vstack([img1, img2])
    x = img.T # x is in n * m, where n is number of pixels and m is number of samples.
    print("------------- x is (without 1 on top yet) ---------------")
    print(x)
    
    # Form a theta
    theta0 = zeros((6, 3))
    print("----------- theta zero is -------------")
    print(theta0) # theta is in n * k, where n is number of pixels and k is number of labels..
    
    # Form a y
    y = array([[1, 0, 0], [0, 0, 1]]);
    y = y.T; # y is in k * m, where k is number of labels and m is the sample size.
    print("---------------- y is -----------------")
    print(y)
    

    print("---------vectorized gradient-----------")
    print(vectorized_gradient(x, y, theta0))
    
    print("------gradient computed using finite diffeence------")
    g = zeros_like(vectorized_gradient(x, y, theta0))
    print("hulo", cf_part6(x, y, theta0))
    # Compute every componento f the gradient using fitenite difference method.
    for p in range(x.shape[0] + 1):
        for q in range(y.shape[0]):
            g[p, q] = finite_difference_gradient(x, y, theta0, p, q)
    print(g)

#reproduce_part_6()



# -----------------------------------------------------------------------------
# PART 7
# -----------------------------------------------------------------------------
def vectorized_gradient(x, y, theta):
    x = vstack((ones((1, x.shape[1])), x))
    return dot(dot(2,x), (dot(theta.T, x) - y).T)
    
# Part 7: gradient descent to find best theta.
def grad_descent_p7(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 50000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha * vectorized_gradient(x, y, t)
        if iter % 500 == 0:
            print "Iter", iter
            #print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            #print "Gradient: ", df(x, y, t, train_size), "\n"
        iter += 1
    return t
    
# Part 7 classifier, using one hot ecoding.
# train_dir: relative path to the directory containing images for training.
# test_dir: relative path to directory containing images for evaluating.
def classifier_6_actor_p7(train_dir, test_dir):
    k = 6 # number of possible labels
    m = 600 # sample size
    n = 1025 # number of pixels.
    # train_dir = "part7_data/train_data/";
    
    train_data = os.listdir(train_dir);
    y = zeros((k, m));
    x = array([])
    i = 0;
    
    act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'];
    for name in train_data:
        if any([actor.split()[1].lower() in name for actor in act]):
            img = imread(train_dir + name)
            img = rgb2gray(img)
            img = img.flatten()
            
            if x.size == 0:
                x = vstack([img])
            elif x.size > 0:
                x = vstack([x, img])
            
            # assign y based on label of the trainning images.
            if "drescher" in name:
                y[0][i] = 1
            elif "ferrera" in name:
                y[1][i] = 1
            elif "chenoweth" in name:
                y[2][i] = 1
            elif "baldwin" in name:
                y[3][i] = 1
            elif "hader" in name:
                y[4][i] = 1
            elif "carell" in name:
                y[5][i] = 1
            
            i = i + 1;
    
    x = x.T;
    theta0 = zeros((n, k));
    alpha = 0.000001;
    # print("x.size:{}".format(x.shape));
    theta = grad_descent_p7(cf_part6, vectorized_gradient, x, y, theta0, alpha);
    
    # print("theta");
    # print(theta);
    
    #--------------Running the classifier to obtain the result------------------
    test_data = os.listdir(test_dir);
    
    # Number of correct guess for each actor.
    count_drescher = 0;
    count_ferrera = 0;
    count_chenoweth = 0;
    count_baldwin = 0;
    count_hader = 0;
    count_carell = 0;
    test_size = 0;
    
    # note that name of the image file contains name of the actor.
    for name in test_data:
        if any([actor.split()[1].lower() in name for actor in act]):
            test_size = test_size + 1;
            imgtest = imread(test_dir + name)
            imgtest = rgb2gray(imgtest)
            imgtest = imgtest.flatten()
            result = dot(theta[1:].T, imgtest) + theta[0].T
            greatest_ind = 0
            for i in range(len(result)):
                if result[i] > result[greatest_ind]:
                    greatest_ind = i
            
            print('The image {} is classified to be {}'.format(name,act[greatest_ind]))
            
            # Check if the classification result matches the label.
            if greatest_ind == 0 and "drescher" in name: 
                count_drescher += 1
            elif greatest_ind == 1 and "ferrera" in name:
                count_ferrera += 1
            elif greatest_ind == 2 and "chenoweth" in name:
                count_chenoweth += 1
            elif greatest_ind == 3 and "baldwin" in name:
                count_baldwin += 1
            elif greatest_ind == 4 and "hader" in name:
                count_hader += 1
            elif greatest_ind == 5 and "carell" in name:
                count_carell += 1
        
    # report the result.
    total_correct = count_drescher + count_ferrera + count_chenoweth + count_baldwin + count_hader + count_carell
    return [total_correct, (float(total_correct)/test_size)*100]
    # print("Number of correct: {} ({} %)".format(total_correct, (float(total_correct)/test_size)*100)); # Percentage result
        
def reproduce_part_7():
    train_dir = "part7_data/train_data/"
    valid_dir = "part7_data/valid_data/"
    
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    if os.path.exists(valid_dir): 
        shutil.rmtree(valid_dir)
    
    os.makedirs(train_dir)
    os.makedirs(valid_dir)
    # 100 training images and 100 validating images per actor.
    pickrandom_p5(act, "cropped/", train_dir, valid_dir, "", 100, 10, 0)
    
    # Performance on training set.
    result = classifier_6_actor_p7(train_dir, train_dir);
    print("-----------------Performance on training set----------------------")
    print("Number of correct: {} ({} %)".format(result[0], result[1])); # Percentage result
    
    # Performance on validating set.
    result = classifier_6_actor_p7(train_dir, valid_dir)
    print("-----------------Performance on validating set----------------------")
    print("Number of correct: {} ({} %)".format(result[0], result[1])) # Percentage result

# reproduce_part_7()

# -----------------------------------------------------------------------------
# PART 8
# -----------------------------------------------------------------------------

# Part 7: gradient descent to find best theta.
def grad_descent_p8(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 1500 # 7500 works well, 8500, 9500 goods
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha * vectorized_gradient(x, y, t)
        if iter % 500 == 0:
            print "Iter", iter
            #print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            #print "Gradient: ", df(x, y, t, train_size), "\n"
        iter += 1
    return t

# display theta.
def display_theta_part8(train_dir):
    k = 6 # number of possible labels
    m = 600 # sample size
    n = 1025 # number of pixels.
    train_dir = train_dir;
    
    train_data = os.listdir(train_dir);
    y = zeros((k, m));
    x = array([])
    i = 0;
    
    act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'];
    for name in train_data:
        if any([actor.split()[1].lower() in name for actor in act]):
            img = imread(train_dir + name)
            img = rgb2gray(img)
            img = img.flatten()
            
            if x.size == 0:
                x = vstack([img])
            elif x.size > 0:
                x = vstack([x, img])
            
            if "drescher" in name:
                y[0][i] = 1
            elif "ferrera" in name:
                y[1][i] = 1
            elif "chenoweth" in name:
                y[2][i] = 1
            elif "baldwin" in name:
                y[3][i] = 1
            elif "hader" in name:
                y[4][i] = 1
            elif "carell" in name:
                y[5][i] = 1
            
            i = i + 1;
    
    x = x.T;
    theta0 = zeros((n, k));
    alpha = 0.000001;
    print("x.size:{}".format(x.shape));
    theta = grad_descent_p8(cf_part6, vectorized_gradient, x, y, theta0, alpha);
    
    # turns theta from n * k into k * (n - 1) matrix (i.e. gets rid of the extra 1)
    theta = theta[1:].T;
    img_theta_drescher = theta[0].reshape((32, 32)); # 0th row
    img_theta_ferrera = theta[1].reshape((32, 32)); # 1st row
    img_theta_cheno = theta[2].reshape((32, 32)); # 2nd row
    img_theta_baldwin = theta[3].reshape((32, 32)); # 3rd row
    img_theta_hader = theta[4].reshape((32, 32)); # 4th row
    img_theta_carell = theta[5].reshape((32, 32)); # 5th row
    
    # Either show or print the image.
    #show(imshow(img_theta_drescher))
    #show(imshow(img_theta_ferrera))
    #show(imshow(img_theta_cheno))
    #show(imshow(img_theta_baldwin))
    #show(imshow(img_theta_hader))
    #show(imshow(img_theta_carell)) # 14,500 iterations.
        
    imsave("part8_drescher", img_theta_drescher)
    imsave("part8_ferrera", img_theta_ferrera)
    imsave("part8_chenoweth", img_theta_cheno)
    imsave("part8_baldwin", img_theta_baldwin)
    imsave("part8_hader", img_theta_hader)
    imsave("part8_carell", img_theta_carell)

def reproduce_part_8():
    # Reuse data from part 7.
    train_dir = "part7_data/train_data/"
    valid_dir = "part7_data/valid_data/"
    
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    if os.path.exists(valid_dir): 
        shutil.rmtree(valid_dir)
    
    os.makedirs(train_dir)
    os.makedirs(valid_dir)
    # 100 training images and 100 validating images per actor.
    pickrandom_p5(act, "cropped/", train_dir, valid_dir, "", 100, 10, 0)
    
    display_theta_part8(train_dir)
    
# reproduce_part_8()