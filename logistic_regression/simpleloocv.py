from datetime import datetime

import numpy as np
import sklearn as sklearnImport

from sklearn import linear_model  # https://scikit-learn.org/stable/modules/linear_model.html
from sklearn.metrics import roc_curve  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
from sklearn.metrics import accuracy_score  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
from sklearn.metrics import confusion_matrix  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
from sklearn.model_selection import LeaveOneOut  # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html
from sklearn.metrics import roc_auc_score  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
import sys

ground_truth={}

c=open("conf.txt").read().split("\n")
for line in c:
    elts=line.split("\t")
    pid=elts[0]
    ear=elts[1]
    label=elts[2]
    ecpyn=elts[3]
    match=elts[4]
    readout=elts[5]
    myring=elts[6]
    gender=elts[7]
    height=elts[8]
    weight=elts[9]
    age=elts[10]
    wax=elts[11]
    fluidtype=elts[12]
    
    k=pid+"-"+ear
    ground_truth[k]=label  # Ground truth - whatever surgeon said


def cnorm(cm1):
    """
    This function TODO
    :param cm1:
    :return:
    """
    cm1=cm1.astype(np.float)
    for i in range(len(cm1)):
        s=sum(cm1[i])
        for j in range(len(cm1)):
            cm1[i][j]=str(round(cm1[i][j]/s,5))
    return cm1


def parseFileName(fname):
    """
    This function TODO
    :param fname:
    :return:
    """
    elts=fname.split('-')
    return elts[1],elts[2][0]


def getLabelTwoClass(fname):
    """
    This function TODO
    :param fname:
    :return:
    """
    pid,cls=parseFileName(fname)
    k=pid+"-"+cls
    if ground_truth[k] == 'n':
        return 0
    elif ground_truth[k] == 'y':
        return 1
    else:
        return 2

    
def getLabelsTwoClass(fnames):
    """
    This function TODO
    :param fnames:
    :return:
    """
    labels=[]
    for i in range(len(fnames)):
        l=getLabelTwoClass(fnames[i])
        labels.append(l)
    return np.asarray(labels)


d0 = ""  # this is the file path of the input files.  An empty path means use the path where this code is running.

clf1=linear_model.LogisticRegression(class_weight={0:1,1:1})

x_file_path = d0 + 'x-iphone5s-dip.npy'
xall=np.load(x_file_path) # like: 'x-iphone5s-dip.npy'
fnames=np.load(d0+'fnames-iphone5s-dip.npy', allow_pickle=True)
yall=getLabelsTwoClass(fnames)

rocthresh=.211
allpreds=[]
allprobs=[]
ysplits=[]
loo = LeaveOneOut()

for train_index, test_index in (loo.split(xall)):
    xtrain, xtest = xall[train_index], xall[test_index]
    ytrain, ytest = yall[train_index], yall[test_index]
    fnametrain, fnametest = fnames[train_index], fnames[test_index]
    clf1.fit(xtrain,ytrain)
    
    ysplits.append(ytest[0])

    probs=clf1.predict_proba(xtest)
    probs=probs[:,1]
    if probs>=rocthresh:
        allpreds.append(1)
    else:
        allpreds.append(0)
    allprobs.append(probs)

print(datetime.now())
print("python version      ", sys.version)
print("numpy.__version__   ", np.__version__)
print("sklearn.__version__ ", sklearnImport.__version__)

print ("CLF ",accuracy_score(ysplits,allpreds))
print ("AUC ",roc_auc_score(ysplits, allprobs))


# The confusion matrix is a table with two rows and two columns, reports the number of false positives, false negatives, true positives, and true negatives.
# The sum of all 4 numbers should equal the number of samples in the training set - initially 98 ears.
conf = confusion_matrix(ysplits, allpreds)
print("Confusion_matrix ", conf);  # https://en.wikipedia.org/wiki/Confusion_matrix
normedconf=cnorm(confusion_matrix(ysplits,allpreds))
print ("cnorm(confusion_matrix) ", normedconf)

fora=1-(float(conf[0][0])/(conf[1][0]+conf[0][0]))
ppv=(float(conf[1][1])/(conf[0][1]+conf[1][1]))

recall=normedconf[1][1]
precision=ppv
f1score=2/((1/recall)+(1/precision))

print ("")
print ("fora, ppv, (fora+ppv)/2 :")
print (fora, ppv, (fora+ppv)/2)

fpr, sensi, threshs = roc_curve(ysplits,allprobs)
speci=1-fpr

print ("")
print("i, sensi[i], speci[i], threshs[i] :")
for i in range(len(threshs)):
    print("%5.0f %.2f %.2f %.5f" % (i, sensi[i], speci[i], threshs[i]))

maxthresh=0
maxmet=0
for i in range(len(threshs)):
    if sensi[i]+speci[i]>maxmet:
        maxmet=sensi[i]+speci[i]
        maxthresh=threshs[i]

print("")
print("fnames of %5.0f allpreds :" % len(allpreds))
for i in range(len(allpreds)):
    if allpreds[i] == 1:
        sys.stdout.write('"'+fnames[i]+'",')
print("")

np.savetxt ('./weights_loocv.txt', clf1.coef_, delimiter='\n')
np.savetxt ('./intercept_loocv.txt', clf1.intercept_)
np.savetxt ('./thresh_loocv.txt', np.asarray([maxthresh]))

print("intercept_loocv.txt %.18E" % clf1.intercept_[0])
print("thresh_loocv.txt %.18E" % np.asarray([maxthresh]))