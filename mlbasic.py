from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
import pydotplus
import threading
import time
import matplotlib.pyplot as plt

iris=load_iris()

print "The feature names are", iris.feature_names
time.sleep(2)
print "The target names are", iris.target_names
time.sleep(2)
print "The first index", iris.target[0]
time.sleep(2)
print "The first data in the index", iris.data[0]
time.sleep(2)
print "The third data (row-wise) in the dataset", iris.data[2]
time.sleep(4)


test_idx=[0,50,100]

#for i in range(len(iris.target)):
 #   print "Example %d: label %s, features %s" %(i, iris.target[i], iris.data[i])

#training data
train_target=np.delete(iris.target, test_idx)
train_data=np.delete(iris.data, test_idx, axis=0)  #along the rows so axis=0


#testing data
test_target=iris.target[test_idx]
test_data=iris.data[test_idx]

clf=tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)


print "Our prediction is",test_target
time.sleep(2)
print "What the tree predicts", clf.predict(test_data)
time.sleep(2)

#viz code

from sklearn.externals.six import StringIO
import pydot

dot_data=StringIO()
tree.export_graphviz(clf,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)




import pydotplus

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")                                    
open("/home/samir/lectures/iris.pdf")

print "The data after testing is", test_data[0], "The index of the first data after testing is", test_target[0]
time.sleep(2)
print "The original data in first row is", iris.data[0], "The first index is", iris.target[0]
time.sleep(2)
print "The feature and species names are", iris.feature_names,  iris.target_names
time.sleep(2)
print "hence the results are same"

plt.plot(iris.data[0], 'r--', iris.data[1], 'bs', iris.data[2], 'g^')
plt.show()












