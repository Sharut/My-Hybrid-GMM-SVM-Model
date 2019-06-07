#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:59:47 2019

@author: uiet_mac1
"""

from GMM_nDim3 import read_data
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from sklearn.metrics import confusion_matrix


labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
X = np.loadtxt("mu.txt")
file_name = "iris.data"
data, ref_clusters = read_data(file_name)
y_test = label_binarize(ref_clusters, classes=[0, 1, 2])
n_classes = y_test.shape[1]
y = np.array([0, 1, 2])
clf = svm.SVC(C=1.0,kernel='rbf',tol = 0.001, decision_function_shape = 'ovr',gamma='auto')
clf.fit(X, y) 
y_score = clf.decision_function(data)
#print(y_score)
prediction = []
for i in range(150):
    print(clf.predict([list(data[i])]), end=" ")
    prediction.append(int(clf.predict([list(data[i])])))

#Accuracy
favour=0
total=0
for i in range(len(ref_clusters)):
    total+=1
    if(ref_clusters[i]==prediction[i]):
        favour+=1
print()
print("Accuracy by sklearn  is : " + str(clf.score(data,ref_clusters)*100))
print("Favourable are " + str(favour) + " Total are "+ str(total) )
acc = (float)(favour/total) * 100
print("Accuracy is "+ str(acc) +"%")


#LinearSVC minimizes the squared hinge loss while SVC minimizes the regular hinge loss.
#LinearSVC uses the One-vs-All (also known as One-vs-Rest) multiclass reduction while SVC uses the One-vs-One multiclass reduction.

data = np.concatenate((data,X),axis = 0)

X_embedded = TSNE(n_components=2).fit_transform(np.array(data))
X = X_embedded[0:150,:]

y = ref_clusters
train_X = X_embedded[150:153,:]
train_y = np.array([0, 1, 2])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(X))
print(scaler.mean_)
X = scaler.transform(X)
#h = .02


C = 1.0  # SVM regularization parameter
#Trained on means
svc = svm.SVC(kernel='linear', C=C).fit(train_X, train_y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(train_X, train_y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(train_X, train_y)
lin_svc = svm.LinearSVC(C=C).fit(train_X, train_y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# title for the plots
titles = ['Hybrid GMM-SVC with linear kernel',
          'Hybrid GMM-LinearSVC (linear kernel)',
          'Hybrid GMM-SVC with RBF kernel',
          'Hybrid GMM-SVC with polynomial (degree 3) kernel']

i=0
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.figure(figsize=(20,10))
    plt.subplot(2, 2, i + 1)
   
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.summer, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.seismic)
    #plt.xlabel('Sepal length')
    #plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()



#Evaluation Metric
from sklearn.metrics import precision_score
y_true = ref_clusters
y_pred = prediction
print("EVALUATION METRIC")
print("Macro precision is",precision_score(y_true, y_pred, average='macro')*100,"%") 
print("Micro precision is",precision_score(y_true, y_pred, average='micro')*100,"%")
print("Weighted precision is",precision_score(y_true, y_pred, average='weighted')*100,"%")
print("Average precision is",precision_score(y_true, y_pred, average=None)*100)

from sklearn.metrics import recall_score
print("TPR(True Positive Rate) / Recall /Sensitivity")
print("Macro recall is",recall_score(y_true, y_pred, average='macro')*100,"%") 
print("Micro recall is",recall_score(y_true, y_pred, average='micro')*100,"%")
print("Weighted recall is",recall_score(y_true, y_pred, average='weighted')*100,"%") 
print("Average recall is",recall_score(y_true, y_pred, average=None)*100)

from sklearn.metrics import precision_recall_fscore_support
#print("Macro precision_recall_fscore is",precision_recall_fscore_support(y_true, y_pred, average='macro')*100,"%") 
#print("Micro precision_recall_fscore is",precision_recall_fscore_support(y_true, y_pred, average='micro')*100,"%")
#print("Weighted precision_recall_fscore is",precision_recall_fscore_support(y_true, y_pred, average='weighted')*100,"%")

from sklearn.metrics import multilabel_confusion_matrix
print("Confusion metric is ")
cm=confusion_matrix(y_true, y_pred)
print(multilabel_confusion_matrix(y_true, y_pred))
#fig = plt.figure(figsize=(6, 4), dpi=75)
fig=plt.figure()
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.xlabel("Predicted Species")
plt.ylabel("True Species")
fig.savefig('./outputs/cm.png', bbox_inches='tight')




# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()