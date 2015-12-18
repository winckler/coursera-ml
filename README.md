# Practical Machine Learning Report
Gabriel A. von Winckler  
December 18, 2015  

# Summary

This is a report on the course project from the Practical Machine Learning course from Johns Hopkins University Data Science specialization on Coursera. This exercise uses the "Weight Lifting Exercise Dataset" from PUC-Rio. The goal is to create and train a machine learn model and evaluate it's performance with a test sub-sample.

# Setup

This work uses the _caret_ package from R. The package _doMC_ allow parallel computation using multiple threads and was also used to allow speed-up the model training using all the cores in the machine (4 cores).

A seed was fixed for reproducibility purpose.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(doMC)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
registerDoMC(cores = 4)

set.seed(999)
```

# Model creation

## Cleaning

The data was loaded from the CSV file. Than it was splitted in training and test subsets. The training subset has 75% of the original samples.


```r
all <- read.csv("pml-training.csv")
inTrain <- createDataPartition(y=all$classe, p=0.75, list=F)
training <- all[inTrain,]
test <- all[-inTrain,]
```

Than the dataset (training subset only) was cleaned to remove:

* obvious unclassifiable columns (X, user_name, *_timestamp, *_window)
* near zero covariates
* column with almost all values _NA_


```r
# remove X, user_name, *_timestamp, *_window
training <- training[,8:160]

# remove near zero covariates
nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[,!nzv[,4]]

# remove columns with NA
training <- training[, colSums(is.na(training)) == 0]
```

## Training

Some different models were tested, such as _Random Forests_, _Classification and regression trees_ and _Classification and regression trees_. Due to the accuracy and processing time, the _Random Forests_ was selected for this model.

Also, to speed-up the training, the training control was tweaked, using _8-fold cross-validation_. (8 was chosen as a multiple of the number of cores for better performance)


```r
model <- train(classe ~ ., data=training, method="rf", 
               trControl=trainControl(method="cv", number=8))
```

```
## Loading required package: randomForest
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

# Performance

## In-sample

The in-sample performance was evaluated using the _confusionMatrix_ function.


```r
pred <- predict(model, training)
cm <- confusionMatrix(pred, training$classe)
print(cm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

As you can see, the performance is excellent.

## Out-of-sample

Regarding the out-of-sample performance, the same approach was used:


```r
pred_test <- predict(model, test)
cm_test <- confusionMatrix(pred_test, test$classe)
print(cm_test)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    5    0    0    0
##          B    0  943    3    0    0
##          C    0    1  851   13    0
##          D    0    0    1  790    1
##          E    0    0    0    1  900
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9925, 0.9967)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9937   0.9953   0.9826   0.9989
## Specificity            0.9986   0.9992   0.9965   0.9995   0.9998
## Pos Pred Value         0.9964   0.9968   0.9838   0.9975   0.9989
## Neg Pred Value         1.0000   0.9985   0.9990   0.9966   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1923   0.1735   0.1611   0.1835
## Detection Prevalence   0.2855   0.1929   0.1764   0.1615   0.1837
## Balanced Accuracy      0.9993   0.9965   0.9959   0.9910   0.9993
```

The result is impressive, with an accuracy of 0.9949021 yielding an confidence interval (CI) of (0.9924837,0.9966983)


# References
```
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
```
