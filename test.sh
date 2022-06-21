#!/bin/bash

weka -main weka.Run .MexicanHat -n 1000 -N 0.1 > MHReg.arff
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner RandomForestRegressor -t $(pwd)/MHReg.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner LinearRegression -t $(pwd)/MHReg.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner MBSGDRegressor -t $(pwd)/MHReg.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner SVR -t $(pwd)/MHReg.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner KNeighborsRegressor -t $(pwd)/MHReg.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLDaskClassifier -split-percentage 80 -v -learner KNeighborsRegressor -t $(pwd)/MHReg.arff -py-command python
# The following two test would fall as cuml raise exception for not implementing but their API doc has the two algorithms
#weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLDaskClassifier -split-percentage 80 -v -learner LinearRegression -t $(pwd)/MHReg.arff -py-command python
#weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLDaskClassifier -split-percentage 80 -v -learner Ridge -t $(pwd)/MHReg.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLDaskClassifier -split-percentage 80 -v -learner Lasso -t $(pwd)/MHReg.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLDaskClassifier -split-percentage 80 -v -learner ElasticNet -t $(pwd)/MHReg.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLDaskClassifier -split-percentage 80 -v -learner CD -t $(pwd)/MHReg.arff -py-command python


weka -main weka.Run .RandomRBF -n 50 -a 1000 > RBFa50n1k.arff
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner LogisticRegression -t $(pwd)/RBFa50n1k.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner MBSGDClassifier -t $(pwd)/RBFa50n1k.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner MultinomialNB -t $(pwd)/RBFa50n1k.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner BernoulliNB -t $(pwd)/RBFa50n1k.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner GaussianNB -t $(pwd)/RBFa50n1k.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner RandomForestClassifier -t $(pwd)/RBFa50n1k.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner SVC -t $(pwd)/RBFa50n1k.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner LinearSVC -t $(pwd)/RBFa50n1k.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLClassifier -split-percentage 80 -v -learner KNeighborsClassifier -t $(pwd)/RBFa50n1k.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLDaskClassifier -split-percentage 80 -v -learner KNeighborsClassifier -t $(pwd)/RBFa50n1k.arff -py-command python
weka -memory 48g -main weka.Run weka.classifiers.rapids.CuMLDaskClassifier -split-percentage 80 -v -learner MultinomialNB -t $(pwd)/RBFa50n1k.arff -py-command python

rm MHReg.arff RBFa50n1k.arff
