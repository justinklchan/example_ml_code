from keras import applications
# from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from os import listdir, mkdir
from os.path import isfile, join
import time
import numpy as np
import os

def deleteold(dtype,experimentNumber,validationFold=''):
    direc='./'  
    datf=[f for f in listdir(direc) if isfile(join(direc, f))]
    fnames=['bottleneck-'+dtype+'-'+str(experimentNumber)+'-'+str(validationFold)]
    
    print ('deleting in 5 seconds')
    time.sleep(5)
    print ('deleting')
    for fname in fnames:
        for i in datf:
            if i.startswith(fname):
                os.remove(i)

def getlatesttrain(dtype,experimentNumber,validationFold=''):
    f=os.listdir('./')
    maxnum=0
    for i in f:
        if 'x'+dtype+'-'+str(experimentNumber)+'-'+str(validationFold) in i:
            num=int(i.split('-')[3][:-4])
            if num>maxnum:
                maxnum=num
    return maxnum

def getlatesttest(dtype,experimentNumber):
    f=os.listdir('./')
    maxnum=0
    for i in f:
        if 'x'+dtype+'-'+str(experimentNumber)+'-' in i:
            num=int(i.split('-')[2][:-4])
            if num>maxnum:
                maxnum=num
    return maxnum

def bottleneck(experimentNumber,validationFold):
#     base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299,299,3))
    base_model = InceptionV3(include_top=False,weights='imagenet',input_shape=(299,299,3))
    bottleneckhelpertrain(experimentNumber,base_model,validationFold)
    bottleneckhelperval(experimentNumber,base_model,validationFold)
#     bottleneckhelpertest(experimentNumber,base_model)

def bottleneckhelpertrain(experimentNumber,base_model,validationFold):
    print ('bottleneck helper train')
#     deleteold('train',experimentNumber,validationFold)
    batch_num=0
    
    nums=getlatesttrain('train',experimentNumber,validationFold)
    start=time.time()
    for batch_num in range(nums+1):
        fname='train-'+str(experimentNumber)+'-'+str(validationFold)+'-'+str(batch_num)+'.npy'
        xslice=np.load('x'+fname)
#         xslice2=preprocess_input(xslice)
#         xslice3=preprocess_input(xslice)
        print(xslice.shape)
        
        preds=np.asarray(base_model.predict(xslice))
        np.save('bottleneck-train-'+str(experimentNumber)+'-'+str(validationFold)+'-'+str(batch_num)+'.npy',preds)
        print ('batch num '+str(batch_num)+'/'+str(nums+1))
        print ('time '+str(time.time()-start))
        start=time.time()
        
def bottleneckhelperval(experimentNumber,base_model,validationFold):
    print ('bottleneck helper val')
#     deleteold('val',experimentNumber,validationFold)
    batch_num=0
    
    nums=getlatesttrain('val',experimentNumber,validationFold)
    
    start=time.time()
    for batch_num in range(nums+1):
        fname='val-'+str(experimentNumber)+'-'+str(validationFold)+'-'+str(batch_num)+'.npy'
        xslice=np.load('x'+fname)
        xslice=preprocess_input(xslice)
        print(xslice.shape)
        
        preds=np.asarray(base_model.predict(xslice))
        np.save('bottleneck-val-'+str(experimentNumber)+'-'+str(validationFold)+'-'+str(batch_num)+'.npy',preds)
        print ('batch num '+str(batch_num)+'/'+str(nums+1))
        print ('time '+str(time.time()-start))
        start=time.time()
        
def bottleneckhelpertest(experimentNumber,base_model):
    print ('bottleneck helper test')
#     deleteold('test',experimentNumber)
    batch_num=0
    
    nums=getlatesttest('test',experimentNumber)
    
    start=time.time()
    for batch_num in range(nums+1):
        fname='test-'+str(experimentNumber)+'-'+str(batch_num)+'.npy'
        xslice=np.load('x'+fname)
        xslice=preprocess_input(xslice)
        print(xslice.shape)
        
        preds=np.asarray(base_model.predict(xslice))
        np.save('bottleneck-test-'+str(experimentNumber)+'-'+str(batch_num)+'.npy',preds)
        print ('batch num '+str(batch_num)+'/'+str(nums+1))
        print ('time '+str(time.time()-start))
        start=time.time()
        
if __name__ == "__main__":
    experimentNumber=4
    validationFold=1
    bottleneck(experimentNumber,validationFold)

