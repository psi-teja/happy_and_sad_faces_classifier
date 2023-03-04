import scipy.linalg as sp
import numpy as np
import os
from PIL import Image
import pickle

imagePaths = os.listdir("train")

happy = []
sad = []
X = []
for (i, imagePath) in enumerate(imagePaths):
	# extract the person emotion from the image path
	emotion = imagePath.split(".")[1]
	print(emotion)
	image = Image.open("train/%s"%(imagePath),"r")

	pix_val = list(image.getdata())
	X.append(pix_val)
	if (emotion == "sad"):
		sad.append(pix_val)
	else:
		happy.append(pix_val)

#array having happy faces data
happy = np.array(happy)
#array having sad faces data
sad = np.array(sad)
#array having all the faces
X = np.array(X)

#finding mean of input data
mean_X = np.mean(X,axis=0)
X = X - mean_X

#covariance matrix
C = np.dot(X.T,X)

#finding eigenvalues and eigenvectors
eigenvalues, eigenvectors = sp.eigh(C)
print("eigenvalues:",eigenvalues)
print("eigenvectors:",eigenvectors)

#storing eigenvalues into a pickle file
f = open("eigenvalues.pkl","wb")
pickle.dump(eigenvalues,f)
f.close()

#stroring 20 most signifiant eigen vectors into a pickle file
vectors = eigenvectors[-20:]
f = open("eigenvectors.pkl","wb")
pickle.dump(vectors,f)
f.close()

#dumping input array X to a pickle file
f = open("input.pkl","wb")
pickle.dump(X,f)
f.close()

#dumping happy faces data into a pickle file 
f = open("happy.pkl","wb")
pickle.dump(happy,f)
f.close()

#dumping sad faces data into a pickle file 
f = open("sad.pkl","wb")
pickle.dump(sad,f)
f.close()