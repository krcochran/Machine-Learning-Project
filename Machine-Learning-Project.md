Practical Machine Learning Project
================

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. The goal of your project is to predict the
manner in which they did the exercise. This is the “classe” variable in
the training set. You may use any of the other variables to predict
with.

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     combine

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
setwd("~/Documents/Data Science Specialization/Practical Machine Learning")
train <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")
```

First we must partition the training dataset to test our model. I’ve
changed the classe variable from “character” to “factor” as it will be
important once we build our model.

``` r
set.seed(1223)
trainingdata <- createDataPartition(train$classe, p = .8, list = FALSE)
Training <- train[trainingdata, ]
Validation <- train[-trainingdata, ]
Training$classe <- as.factor(Training$classe)
Validation$classe <- as.factor(Validation$classe)
```

Now we clean the data to get rid of unhelpful columns (NA columns,
columns with mostly spaces etc.) and descriptive columns.

``` r
Training <- Training[, colSums(is.na(Training)) == 0]
Training <- Training[, colSums(Training == "") == 0]
Training <- Training[, -c(1:7)]
```

Next, we build the model using a random forest algorithm.

``` r
model <- randomForest(classe ~ ., data = Training, importance = TRUE, ntrees = 15)
```

Then we test it against the training set and use the validation
partition to cross validate.

``` r
self_train <- predict(model, Training)
confusionMatrix(self_train, Training$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 4464    0    0    0    0
    ##          B    0 3038    0    0    0
    ##          C    0    0 2738    0    0
    ##          D    0    0    0 2573    0
    ##          E    0    0    0    0 2886
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9998, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##                                      
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

``` r
validation <- predict(model, Validation)
confusionMatrix(validation, Validation$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1116    1    0    0    0
    ##          B    0  757    4    0    0
    ##          C    0    1  679    8    0
    ##          D    0    0    1  635    1
    ##          E    0    0    0    0  720
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9959          
    ##                  95% CI : (0.9934, 0.9977)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9948          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9974   0.9927   0.9876   0.9986
    ## Specificity            0.9996   0.9987   0.9972   0.9994   1.0000
    ## Pos Pred Value         0.9991   0.9947   0.9869   0.9969   1.0000
    ## Neg Pred Value         1.0000   0.9994   0.9985   0.9976   0.9997
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1930   0.1731   0.1619   0.1835
    ## Detection Prevalence   0.2847   0.1940   0.1754   0.1624   0.1835
    ## Balanced Accuracy      0.9998   0.9981   0.9950   0.9935   0.9993

The accuracy is at 0.995 so our out of sample error should be at 0.005.
Since that error rate is very low, we can say the model performs
considerably well.

Finally, we use our model on the test data set.

``` r
test_final <- predict(model, test)
test_final
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
