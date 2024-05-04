import os
import numpy as np
import random

from sys import platform as sys_pf
if sys_pf == 'darwin':
	matplotlib.use("TkAgg")

import matplotlib
from matplotlib import pyplot as plt


# FOR REFERENCE:
# https://github.com/brendenlake/omniglot
# ---
# Demo for how to load image and stroke data for a character
# ---

# -----------------------------------
# overview
# -----------------------------------

# The Omniglot data set is designed for developing more human-like learning algorithms.
# It contains 1623 different handwritten characters from 50 different alphabets. 
# Each of the 1623 characters was drawn online via Amazon's Mechanical Turk by 20 different people. 
# Each image is paired with stroke data, a sequences of [x,y,t] coordinates with time (t) in milliseconds.

# The Omniglot data set contains 50 alphabets.
# We split these into a background set of 30 alphabets and an evaluation set of 20 alphabets.
#   - images_background:  30 alphabets
#   - images_evaluation:  20 alphabets

# To compare with the results in our paper, only the background set should be
# used to learn general knowledge about characters
#  (e.g., feature learning, meta-learning, or hyperparameter inference). 
# One-shot learning results are reported using alphabets from the evaluation set.

# -----------------------------------
# background small 1 and 2
# -----------------------------------
# Two more challenging "minimal" splits contain only five background alphabets, 
# denoted as "background small 1" and "background small 2". 
# This is a closer approximation to the experience that a human adult might have for characters in general. 
# For the goal of building human-level AI systems with minimal training, 
# given a rough estimate of what "minimal" means for people, 
# there is a need to explore settings with fewer training examples per class 
# and fewer background classes for learning to learn.

# -----------------------------------
# character:  START, BREAK
# -----------------------------------
# A character is a series of pen coordinates (x,y,time) beginning with "START". 
# Breaks between pen strokes are denoted as "BREAK" (indicating a pen up action). 
# The stroke data is raw. It has non-uniform spatial and temporal sampling intervals, 
# as the data was collected from many different web browsers and computers. 
# For most applications, you will want to interpolate to get uniform spatial or temporal sampling intervals.


##########################################################################################################
# --------------------------------------------------------------------------------------------------------
# plot functions
# --------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------
# plot
#   - motor trajectory over an image
# --------------------------------------------------------------------------------------------------------

# Map from motor space to image space (or vice versa)
#
# Input
#   pt: [n x 2] points (rows) in motor coordinates
#
# Output
#  new_pt: [n x 2] points (rows) in image coordinates

def space_motor_to_img(pt):
	pt[:,1] = -pt[:,1]
	return pt

def space_img_to_motor(pt):
	pt[:,1] = -pt[:,1]
	return

# Input
#  I [105 x 105 nump] grayscale image
#  drawings: [ns list] of strokes (numpy arrays) in motor space
#  lw : line width

def plot_motor_to_image(I,drawing,lw=2):
	drawing = [d[:,0:2] for d in drawing] # strip off the timing data (third column)
	drawing = [space_motor_to_img(d) for d in drawing] # convert to image space
	plt.imshow(I,cmap='gray')
	ns = len(drawing)
	for sid in range(ns): # for each stroke
		plot_traj(drawing[sid],get_color(sid),lw)
	plt.xticks([])
	plt.yticks([])


# --------------------------------------------------------------------------------------------------------
# plot
#   - plot individual stroke
# --------------------------------------------------------------------------------------------------------

# Input
#  stk: [n x 2] individual stroke
#  color: stroke color
#  lw: line width
def plot_traj(stk,color,lw):
	n = stk.shape[0]
	if n > 1:
		plt.plot(stk[:,0],stk[:,1],color=color,linewidth=lw)
	else:
		plt.plot(stk[0,0],stk[0,1],color=color,linewidth=lw,marker='.')


# --------------------------------------------------------------------------------------------------------
# plot
#   - load stroke data for a character from text file
# --------------------------------------------------------------------------------------------------------

# Input
#   fn : filename
#
# Output
#   motor : list of strokes (each is a [n x 3] numpy array)
#      first two columns are coordinates
#	   the last column is the timing data (in milliseconds)

def load_motor(fn):
	motor = []
	with open(fn,'r') as fid:
		lines = fid.readlines()
	lines = [l.strip() for l in lines]
	for myline in lines:
		if myline =='START': # beginning of character
			stk = []
		elif myline =='BREAK': # break between strokes
			stk = np.array(stk)
			motor.append(stk) # add to list of strokes
			stk = [] 
		else:
			arr = np.fromstring(myline,dtype=float,sep=',')
			stk.append(arr)
	return motor


##########################################################################################################
# --------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------

# Color map for the stroke of index k
def get_color(k):	
    scol = ['r','g','b','m','c']
    ncol = len(scol)
    if k < ncol:
       out = scol[k]
    else:
       out = scol[-1]
    return out


# convert to str and add leading zero to single digit numbers
def num2str(idx):
	if idx < 10:
		return '0'+str(idx)
	return str(idx)


# Load binary image for a character
#
# fn : filename
def load_img(fn):
	I = plt.imread(fn)
	I = np.array(I,dtype=bool)
	return I


##########################################################################################################
# --------------------------------------------------------------------------------------------------------
# base setting
# --------------------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/omniglot/one-shot'

data_folder = '/media/kswada/MyFiles/dataset/omniglot'



# --------------------------------------------------------------------------------------------------------
# plot images
# --------------------------------------------------------------------------------------------------------

img_dir = os.path.join(data_folder, 'images_background')

stroke_dir = os.path.join(data_folder, 'strokes_background')

print(os.path.exists(img_dir))
print(os.path.exists(stroke_dir))


# ----------
nreps = 20 # number of renditions for each character

nalpha = 5 # number of alphabets to show

# get folder names
alphabet_names = [a for a in os.listdir(img_dir) if a[0] != '.']

# random sample
alphabet_names = random.sample(alphabet_names,nalpha)

print(alphabet_names)


# ----------
for a in range(nalpha): # for each alphabet
	print('generating figure ' + str(a+1) + ' of ' + str(nalpha))
	alpha_name = alphabet_names[a]
	
	# choose a random character from the alphabet
	character_id = random.randint(1,len(os.listdir(os.path.join(img_dir,alpha_name))))

	# get image and stroke directories for this character
	img_char_dir = os.path.join(img_dir,alpha_name,'character'+num2str(character_id))
	stroke_char_dir = os.path.join(stroke_dir,alpha_name,'character'+num2str(character_id))

	# get base file name for this character
	fn_example = os.listdir(img_char_dir)[0]
	fn_base = fn_example[:fn_example.find('_')] 

	plt.figure(a,figsize=(10,8))
	plt.clf()
	for r in range(1,nreps+1): # for each rendition
		plt.subplot(4,5,r)
		fn_stk = stroke_char_dir + '/' + fn_base + '_' + num2str(r) + '.txt'
		fn_img = img_char_dir + '/' + fn_base + '_' + num2str(r) + '.png'			
		motor = load_motor(fn_stk)
		I = load_img(fn_img)
		plot_motor_to_image(I,motor)
		if r==1:
			plt.title(alpha_name[:15] + '\n character ' + str(character_id))
	plt.tight_layout()

plt.show()


# -->
# different colors indicating different strokes
