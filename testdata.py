import scipy.linalg as sp
import numpy as np
import os
from PIL import Image
import pickle

imagePaths = os.listdir("test")

happy = []
sad = []
X = []
for (i, imagePath) in enumerate(imagePaths):
	# extract the person emotion from the image path
	emotion = imagePath.split(".")[1]
	print(emotion)
	image = Image.open("test/%s"%(imagePath),"r")

	pix_val = list(image.getdata())
	X.append(pix_val)
	if (emotion == "sad"):
		sad.append(pix_val)
	else:
		happy.append(pix_val)

#dumping happy faces(test) data into a pickle file
f = open("happy_test.pkl","wb")
pickle.dump(happy,f)
f.close()

#dumping sad faces data(test) into a pickle file
f = open("sad_test.pkl","wb")
pickle.dump(sad,f)
f.close()
