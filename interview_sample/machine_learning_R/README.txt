———————————
Instruction 
———————————
This is R version assignments of the GT OMSCS Machine Learning class.
Five machine learning algorithms are implemented on two datasets: Breast Cancer Diagnosis and White Wine Quality. Two datasets are normalized.

==========================
     Data Information 
==========================

1. breast cancer diagnosis data
There are 7 attributes in breast cancer diagnosis(BCD)file:


1)ID
2)Mean.Concave.Points
3)Worst.Concave.Points
4)Worst.Area
5)Worst.Radius
6)Worst.Perimeter
7)Diagnosis (Benign/Malignant)

(Benign and Malignant columns are only used in neural network)

There are two BCD files: cancerNor.csv and 10cancer.csv, 
*10cancer.csv file only used in plotting boosting iterations 
*cancerNor.csv is used in all algorithm

2. white wine quality data
There are 8 attributes in white wine quality file:

1)ID
2)alcohol
3)pH
4)citric acid
5)free sulfur dioxide
6)total sulfur dioxide
7)residual sugar
8)quality (Very Bad, Bad, Fair, Good, Nice, Very Good, Fantastic)


There are two files: wineNor.csv and wineNormal.csv, 
wineNormal.csv file only used in neural network and plotting boosting iterations. In the wineNormal.csv file, quality is changed into numeric vector to avoid compiling error.

======================
        Steps
======================
all the data can be accessed via R working directory 
1. load data and split data into training data and testing data
2. decision tree
3. support vector machine
4. boosting
5. kNN
6. neural network (ann)
7. plotting training error and testing error

=====================
       Package
=====================

caret : training model, 10-folds validation, tuning parameters according to model
C50: C5.0 algorithm to build decision tree in boosting
raprt: regression and classification tree to build decision tree model
gmodels: plot confusion matrix table
kernlab: build support vector machine
neural network: build neural network model
rsnns: use to plot iteration errors in neural network
gym: build boosting model 
rapt.plot: plot part tree


