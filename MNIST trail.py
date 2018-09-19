# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 15:58:49 2018

@author: ponce
"""

import numpy as np
import matplotlib.pyplot as plt
from ImportDataset import DataUtils

from sklearn.neural_network import MLPClassifier
import pickle

DataDir="D:/Binxu Docs"#"/Users/binxu/Documents/2017大四下学期/Big Data Algorithm"
trainfile_X = DataDir + "/DataSet/MNIST/train-images.idx3-ubyte"
trainfile_y = DataDir + "/DataSet/MNIST/train-labels.idx1-ubyte"
testfile_X = DataDir + "/DataSet/MNIST/t10k-images.idx3-ubyte"
testfile_y = DataDir + "/DataSet/MNIST/t10k-labels.idx1-ubyte"

train_X = DataUtils(filename=trainfile_X).getImage()
train_y = DataUtils(filename=trainfile_y).getLabel()
test_X = DataUtils(testfile_X).getImage()
test_y = DataUtils(testfile_y).getLabel()
#traineo_y=np.zeros((len(train_y),2))
#evenmsk=(train_y%2==0)
#traineo_y[evenmsk,1]=1
#traineo_y[~evenmsk,0]=1
Ntrain=len(train_y)

# %%

mlpmodel=MLPClassifier(hidden_layer_sizes=(784,200,100,10))
mlpmodel.fit(train_X,train_y);
#%%
pickle.dump(mlpmodel,open( 'mlpmodel_rec.p' , "wb" ))
#%%
def mlpfunc(X,entry=1,model=mlpmodel):
	if X.shape is not (1,784):
		X=X.reshape(1,-1)	
	probvec=model.predict_proba(X);
	if entry is None:
		return probvec
	else:
		try:
			return probvec[0,entry]
		except:
			print("invalid input 'entry'")
			return probvec


# %%
initX=0+np.random.poisson(lam=30, size=(1,784))
pred=mlpmodel.predict(initX)
print("predicted num: %d" % pred)
plt.figure(1);
plt.clf()
plt.imshow(initX.reshape(28,28),cmap=plt.cm.gray)
plt.colorbar()
plt.title("predicted num: %d" % pred)
plt.show()

mlpfunc(initX,entry=None)
#%%
from cmaes import cmaes
#%%
XOut=cmaes(fitnessfunc = mlpfunc, N=784, initX=initX, maximize=True
		   , save=True, savemrk='mlp4-out1', stopsigma=1e-5);
# After see the data, I found the `sigma ` keeps going up and diverge to 1E19 after the first few trials 
# so the data is not good on this sense
#%%
XOut2=cmaes(fitnessfunc = mlpfunc, N=784, initX=initX, maximize=True
		   , save=True, savemrk='mlp4-out1', stopsigma=1e-5);

   
#%%
expid='mlp4-out1'
cnt=200
(xmean_cache, C, sigma, pc, ps)=pickle.load( open( 'CMAES_%s_optm_rec%d.p' % (expid,cnt), "rb" ) )
   
 #%%  
def ShowImg(iteri,expid='mlp4-out1',cnt=200):
	(xmean_cache, C, sigma, pc, ps)=pickle.load( open( 'CMAES_%s_optm_rec%d.p' % (expid,cnt), "rb" ) )
	genNo=cnt*xmean_cache.shape[1] +iteri
	print("%d generations, sigma:%.2E "%(genNo, sigma)		)		
	# , CovMat Spectrum range: (%.2E,%.2E)			    
	imgcol=xmean_cache[:,iteri-1:iteri]
	plt.figure(1);
	plt.clf()
	plt.imshow(imgcol.reshape(28,28),cmap=plt.cm.gray)
	plt.colorbar()
	plt.show()
	
		   
# mlpfunc(initX)