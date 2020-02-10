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
import shutil

act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'];


# Randomly pick non-overllaping training set, test set, 
# validation set from the set of cropped faces of the actor.
# Parameters:
#       [1] actor is a list of actors to be picked from
#       [2] source_dir is the source directory of the images to be pick from
#       [3][4][5]training_dir, valid_dir, test_dir are the destination for the training,
#       validation and test sets respectively.
#       [6][7][8] train_size, valid_size and test_size are size of the training set, validation set and test_size
#       respectively.
def pickrandom_p5(actor, source_dir, training_dir, valid_dir, test_dir,
                    train_size, valid_size, test_size):
    
    # Looping through the list of actors
    for i in range(len(actor)):
        random.seed(i)
        
        # List contains random incides of the image to be picked.
        to_be_picked = random.sample(range(1, 151), train_size + valid_size + test_size)
        
        # Iterate through the list to be picked and picked one by one.
        for k in range(len(to_be_picked)):
            count = 0
            
            # Iterate through the list of files in target directory.
            for fname in os.listdir(source_dir):
                
                
                act_last_name = actor[i].split()[1].lower()
                #print(act_last_name)
                
                # Check if last name is in the file name
                if act_last_name in fname:
                    
                    # count is the oder of the file.
                    count = count + 1;
                    
                    # test data, to_be_picked[k] stores the random indicies
                    if count == to_be_picked[k] and k < train_size:
                        shutil.copy(source_dir + fname, training_dir)
                        
                    # validation data
                    elif count == to_be_picked[k] and k >= train_size and k < train_size + valid_size:
                        shutil.copy(source_dir + fname, valid_dir)
                        
                    # test data
                    elif count == to_be_picked[k] and k >= train_size + valid_size and k < train_size + valid_size + test_size:
                        shutil.copy(source_dir + fname, test_dir)
                    
                    
                
# pickrandom_p5(act, "cropped/", "part5_data/train_data_10", "part5_data/valid_data", "part5_data/test_data", 10, 0, 0);
# pickrandom_p5(act, "cropped/", "part5_data/train_data_10", "", "", 10, 0, 0);
# pickrandom_p5(act, "cropped/", "part7_data/train_data", "part7_data/valid_data", "", 120, 10, 0);