import keras
from keras import applications
import cv2
import sys
from os import listdir, mkdir
from os.path import isfile, join
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, accuracy_score
import time
from sklearn.utils import class_weight
import os
from keras.models import Sequential
from keras.models import Model,Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

def cnorm(cm1):
    cm1=cm1.astype(np.float)
    for i in range(len(cm1)):
        s=sum(cm1[i])
        for j in range(len(cm1)):
            cm1[i][j]=str(round(cm1[i][j]/s,5))
    return cm1

def getlatest(direc,dtype,experimentNumber,validationFold):
    f=os.listdir(direc)
    maxnum=0
    for i in f:
        if 'x'+dtype+'-'+str(experimentNumber)+'-'+str(validationFold) in i:
            num=int(i.split('-')[-1][:-4])
            if num>maxnum:
                maxnum=num
    return maxnum+1

def validate(direc,model,vals,experimentNumber):
#     class 0 is hemangiomas
    posclass=0
    
    batch_num=0
    xval=np.array([])
    yval=np.array([])
    
    preds=[]
    gt=[]
    probs=[]
    
    vals=getlatest(direc,'val',experimentNumber,validationFold)
    for batch_counter in range(len(vals)):
        batch_num=vals[batch_counter]
        
        fname='bottleneck-val-'+str(experimentNumber)+'-'+str(validationFold)+'-'+str(batch_num)+'.npy'
        fname2='yval-'+str(experimentNumber)+'-'+str(validationFold)+'-'+str(batch_num)+'.npy'
        print (str(batch_counter)+'/'+str(len(vals)))
        
        xval=np.load(fname)
        yval=np.load(fname2)
        
        pred=model.predict(xval)
        gt.extend([1 if np.argmax(i)==posclass else 0 for i in yval])
        preds.extend([1 if np.argmax(i)==posclass else 0 for i in pred])
        probs.extend([i[posclass] for i in pred])
        
        acc=accuracy_score(gt,preds)
    
    gt=np.asarray(gt)
    preds=np.asarray(preds)
    probs=np.asarray(probs)
    
    fpr, sensi, threshs = roc_curve(gt,probs)
    speci=1-fpr
    print ("AUC ",roc_auc_score(gt, probs))
    
#     sns.set()
#     plt.figure()
#     plt.plot(sensi,speci)
#     plt.show()
    
    conf=confusion_matrix(gt,preds)
    normedconf=cnorm(conf)
    print (conf)
    print (normedconf)
    return xval,yval

def onehot2normal(yval):
    labels=[]

    for i in yval:
        labels.append(np.argmax(i))
    return labels

# def getclassweights(experimentNumber,trains):
#     y=[]
#     for batch_num in trains:
#         fname='ytrain-'+str(experimentNumber)+'-'+str(batch_num)+'.npy'
#         y.extend(onehot2normal(np.load(fname)))
#     
#     class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(y),y)
#     return class_weights

def getbnecks(experimentNumber,validationFold):
    ftrain=[f for f in listdir(direc) if isfile(join(direc, f)) and ('bottleneck-train-'+str(experimentNumber)+'-'+str(validationFold) in f)]
    fval=[f for f in listdir(direc) if isfile(join(direc, f)) and ('bottleneck-val-'+str(experimentNumber)+'-'+str(validationFold) in f)]
    ftest=[f for f in listdir(direc) if isfile(join(direc, f)) and ('bottleneck-test-'+str(experimentNumber)+'-'+str(validationFold) in f)]
    
    return ftrain,fval,ftest

def deleteold(direc,experimentNumber,validationFold):
    datf=[f for f in listdir(direc) if isfile(join(direc, f))]
    fnames=[direc+'traintop-'+str(experimentNumber)+'-'+str(validationFold)]
    
    print ('deleting in 5 seconds')
    time.sleep(5)
    print ('deleting')
    for fname in fnames:
        for i in datf:
            if i.startswith(fname):
                os.remove(i)

def traintop(direc,experimentNumber,validationFold):
#     deleteold(direc,experimentNumber,validationFold)
    xtrain = np.load(direc+'bottleneck-train-'+str(experimentNumber)+'-'+str(validationFold)+'-0.npy')
    ytrain = np.load(direc+'ytrain-'+str(experimentNumber)+'-'+str(validationFold)+'-0.npy')
    
    i = Input(shape=xtrain.shape[1:])
    a = Flatten(name='a1')(i)
    a= Dense(256, activation='relu',name='a2')(a)
    a = Dropout(0.6,name='a3')(a)
    o = Dense(ytrain.shape[1], activation='softmax',name='a4')(a)
    model = Model(inputs=i,outputs=o)
    
    opt=keras.optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(optimizer=opt,
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])
        
    print (model.summary())
    
#     trains,vals=getbnecks(experimentNumber)
#     cweights=getclassweights(experimentNumber,trains)

    trains=getlatest(direc,'train',experimentNumber,validationFold)
    epochs=1000
    for e in range(epochs):
        for batch_counter in range(trains):
            batch_num=batch_counter
            fname=direc+'bottleneck-train-'+str(experimentNumber)+'-'+str(validationFold)+'-'+str(batch_num)+'.npy'
            fname2=direc+'ytrain-'+str(experimentNumber)+'-'+str(validationFold)+'-'+str(batch_num)+'.npy'

            xtrain = np.load(fname)
            ytrain = np.load(fname2)
            print (str(batch_counter)+'/'+str(trains-1))
            print (xtrain.shape,ytrain.shape)
            model.fit(xtrain, ytrain,batch_size=128)
            
#             xtest=np.expand_dims(xtrain[0], axis=0)
#             ytest=np.expand_dims(ytrain[0], axis=0)
            
#             pred=model.predict(xtest)
            
#         validate(model,vals,experimentNumber)
        if e%10==0:
            model.save(direc+'traintop-'+str(experimentNumber)+'-'+str(validationFold)+'-'+str(e)+'.h5')

if __name__ == "__main__":
    direc='./'
#     direc=sys.argv[1]
    validationFold=1
    experimentNumber=4
    traintop(direc,experimentNumber,validationFold)












