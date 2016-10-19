Synoposis
---------

In this project, we use the data from personal activity to quantify how
well the participant do a particular activity. The "classe" is the
variable that is going to be predicted. Firstly, we extract relevant
predictors. Then, we implement two relevant models for the multi-class
classification problem, i.e., the decision tree and random forest models
to check which model perform better. Finally, we use the chosen model to
predict 20 different test cases. It turns out that the random forest
model perform better in terms of the accuracy, therefore we have enough
reasons to implement the random forest model to predict the test cases.

    library(caret)

    ## Warning: package 'caret' was built under R version 3.2.5

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library(knitr)

    ## Warning: package 'knitr' was built under R version 3.2.5

    library(rpart)
    library(rpart.plot)

    ## Warning: package 'rpart.plot' was built under R version 3.2.5

    library(markdown)
    library(randomForest)

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

Download and read the data sets
-------------------------------

This R code supports download training and testing datasets directly
from the website. We read the data sets after having download them.

    setwd("/Users/yuyu/Desktop/DataScientist/MachineLearning/PracticalMachineLearningDataScienceTrackCoursera/week4/FinalProject")
    trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(trainURL, destfile = "./traindat")
    download.file(testURL, destfile="./testdat")
    traindat <- read.csv("traindat")
    testdat <- read.csv("testdat")

Preprocessing data
------------------

We split the "traindat" into training (75 percent) and testing (25
percent) data sets. We do the following procesured to pre-process the
data. Firstly, we remove near zero-variance predictors; then, we remove
predictors with more than 60 percent missing values; finally, we remove
intuitively bad predictors (e.g., ID number, user names and recording
times etc). Note that in the steps of splitting data, removing "near
zero-variance", predictors with more than 80 percent missing values and
intuitively bad predcitors, we check the dimensions of training and
testing data sets after the corresponding steps are implemented.

    # split the traindat into training and testing data sets
    inTrain <- createDataPartition(traindat$classe, p = 0.75, list=FALSE)
    mytraining <- traindat[inTrain,]
    mytesting <- traindat[-inTrain,]
    dim(mytraining); dim(mytesting)

    ## [1] 14718   160

    ## [1] 4904  160

    #remove "near zero-variance" predictors
    nzv <- nearZeroVar(mytraining, saveMetrics = TRUE)
    mytraining <- mytraining[,nzv$nzv==FALSE]
    mytesting <- mytesting[,nzv$nzv==FALSE]
    dim(mytraining); dim(mytesting)

    ## [1] 14718   105

    ## [1] 4904  105

    #remove predictors with more than 80 percent missing values
    mytraining <- mytraining[, -which(colMeans(is.na(mytraining)) > 0.8)]; mytesting <- mytesting[,names(mytraining)]; 
    dim(mytraining); dim(mytesting);

    ## [1] 14718    59

    ## [1] 4904   59

    #remove the predictors that are not intuitively relevant
    mytraining <- mytraining[, -c(1:5)];
    mytesting <- mytesting[,-c(1:5)]; 
    dim(mytraining); dim(mytesting);

    ## [1] 14718    54

    ## [1] 4904   54

From the above results, we can see that after removing some predictors,
we end up with predictors 53 in the end (note that "classe" is the
variable we are going to predict).

Prediction with Decision Trees
------------------------------

As this is a multi-class classification problem, we can try to implement
the decision tree model.

    set.seed(12345)
    dtm <- rpart(classe ~., data=mytraining, method = "class")
    rpart.plot(dtm, type = 4, extra = 101, main = "Decision Tree Model")

![](courseproject_files/figure-markdown_strict/unnamed-chunk-4-1.png)

    #using the decision tree model to predict
    p <-predict(dtm,mytesting, type="class")
    confusionMatrix(p,mytesting$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1252  206   33   81   57
    ##          B   31  539   60   26   80
    ##          C   17   52  683  109   88
    ##          D   75  123   53  546  111
    ##          E   20   29   26   42  565
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.731           
    ##                  95% CI : (0.7184, 0.7434)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6582          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8975   0.5680   0.7988   0.6791   0.6271
    ## Specificity            0.8926   0.9502   0.9343   0.9117   0.9708
    ## Pos Pred Value         0.7686   0.7323   0.7197   0.6013   0.8284
    ## Neg Pred Value         0.9563   0.9016   0.9565   0.9354   0.9204
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2553   0.1099   0.1393   0.1113   0.1152
    ## Detection Prevalence   0.3322   0.1501   0.1935   0.1852   0.1391
    ## Balanced Accuracy      0.8950   0.7591   0.8666   0.7954   0.7989

Prediction with Random Forests
------------------------------

We also try the random forests model to do the classification and check
whether there are some improvements in the accuracy.

    rfm <-randomForest(classe~., data=mytraining)
    p<-predict(rfm, mytesting, type="class")
    confusionMatrix(p,mytesting$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1395    0    0    0    0
    ##          B    0  948    9    0    0
    ##          C    0    1  846    7    0
    ##          D    0    0    0  796    1
    ##          E    0    0    0    1  900
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9961         
    ##                  95% CI : (0.994, 0.9977)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9951         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9989   0.9895   0.9900   0.9989
    ## Specificity            1.0000   0.9977   0.9980   0.9998   0.9998
    ## Pos Pred Value         1.0000   0.9906   0.9906   0.9987   0.9989
    ## Neg Pred Value         1.0000   0.9997   0.9978   0.9981   0.9998
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2845   0.1933   0.1725   0.1623   0.1835
    ## Detection Prevalence   0.2845   0.1951   0.1741   0.1625   0.1837
    ## Balanced Accuracy      1.0000   0.9983   0.9937   0.9949   0.9993

Indeed, we see there is a big improvement in the accuracy compared with
the decision tree model. As the accuracy for the testing data set is
0.9969, the out of sample error is (1-0.9969)=0.0031. Therefore, we are
in favor of the random forest model over the decision tree model.

Prediction for the test cases
-----------------------------

Now we use the random forest model to predict 20 different test cases.

    predict(rfm, testdat, type = "class")

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
