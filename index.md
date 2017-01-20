# MLearning
Leandro Meili  
1/19/2017  



## Machine Learning Final Project

In this report we are going to check the activity type from some wearables sensors. The goal is to predict the correct exercise type (A,B,C,D,E). The train data has ~ 20k observations.

## Choosing the Variables


Let`s take a look at our train data

```r
test <- read.csv("pml-testing.csv",header=TRUE)
train <- read.csv("pml-training.csv",header=TRUE)
dim(train)
```

```
## [1] 19622   160
```

```r
summary(train$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

So, first, we will partition this raw data in a training (70%) and testing set (30%).

```r
set.seed(100)

inBuild <- createDataPartition(y=train$classe,p=0.7,list=FALSE)
training <- train[inBuild,]
testing <- train[-inBuild,]
```

## Model Testing

Reading the original paper from the brazilian researches, looks like that roll, pitch and accel are the most important variables. So, for each sensor (there are 4), we will pcik the roll_,pitch_ and total_accel, giving us 12 variables for the prediction.

methods:
  mod1: gbm
  mod2: random forest
  

```r
mod1 <- train(classe ~roll_belt + pitch_belt + total_accel_belt
              +roll_arm + pitch_arm + total_accel_arm +
                +roll_dumbbell+pitch_dumbbell+total_accel_dumbbell
              +roll_forearm+pitch_forearm+total_accel_forearm,method = "gbm", 
              data=training,verbose=FALSE)
mod2 <- train(classe ~roll_belt + pitch_belt + total_accel_belt
              +roll_arm + pitch_arm + total_accel_arm +
                +roll_dumbbell+pitch_dumbbell+total_accel_dumbbell
              +roll_forearm+pitch_forearm+total_accel_forearm,method = "rf", 
              data=training)

pred1 <- predict(mod1,testing)
pred2 <- predict(mod2,testing)

m1 <- confusionMatrix(pred1,testing$classe)
m2 <- confusionMatrix(pred2,testing$classe)
```
## Conclusions

Checking the accuracy of both models, we see that the randomForest gave us a very nice prediction ~98%, against 90% from the gbm method

```r
m1$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.051827e-01   8.801291e-01   8.974106e-01   9.125503e-01   2.844520e-01 
## AccuracyPValue  McnemarPValue 
##   0.000000e+00   2.300809e-16
```

```r
m2$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9823280      0.9776447      0.9786275      0.9855384      0.2844520 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```
It will be clear when we check the confusion matrix for the random forest

```r
m2$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1664   19    2    0    0
##          B    3 1098    5    0    5
##          C    4   22 1005   17    4
##          D    2    0   14  946    5
##          E    1    0    0    1 1068
```
So, with this good prediction, the method of choice is the RandomForest.


