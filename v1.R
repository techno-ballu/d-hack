# clear the memory!
rm(list=ls(all=TRUE))
gc()

library(caret)
library(readr)
library(xgboost)
library(plyr)

set.seed(786)

# set the work directory
setwd("~/Backup/d-hack")

cat("reading the train and test data\n")
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

outliersTolerance <- 10
varianceTolerance <- 0.009

feature.names <- names(train)[2:ncol(train)-1]

cat("assuming text variables are categorical & replacing them with numeric ids\n")
c <- 0
for (f in feature.names) {
    cat(f, class(train[[f]]), '\n')
    # levels
    levelsZ <- unique(c(train[[f]], test[[f]]))
    levelsX <- unique(train[[f]])
    anyNA <- any(is.na(levelsX))
    
    if (length(levelsX) == 1 | (length(levelsX) == 2 & anyNA == TRUE)) {
        print('dropped')
        train[[f]] <- NULL
        test[[f]] <- NULL
        c <- c + 1
    }
    
    # numerically label categories for now
    # TODO deal properly later...
    if (class(train[[f]])=="character") {
        # check for zero variance
        train[[f]] <- as.integer(factor(train[[f]], levels=levelsZ))
        test[[f]]  <- as.integer(factor(test[[f]],  levels=levelsZ))
    } 
#     # remove outliers from numerical
#     else if (class(train[[f]])=="integer") {
#         x <- train[[f]]
#         y <- test[[f]]
#         z <- c(x, y)
#         varZ <- var(z,na.rm = T)
#         
#         if (is.na(varZ) | varZ < varianceTolerance) {
#             print('dropped')
#             train[[f]] <- NULL
#             test[[f]] <- NULL
#             c <- c + 1
#         } else {
#             meanZ <- mean(z,na.rm = T)
#             stdZ <- sd(z,na.rm = T)
#             if (any(x > (meanZ + outliersTolerance*stdZ) | x < (meanZ - outliersTolerance*stdZ), na.rm = T)) {
#                 cat(f, 'train ')
#                 train[[f]] <- replace(x, x > (meanZ + outliersTolerance*stdZ), (meanZ + outliersTolerance*stdZ)) # 
#                 train[[f]] <- replace(x, x < (meanZ - outliersTolerance*stdZ), (meanZ - outliersTolerance*stdZ)) # 
#             } 
#             if (any(y > (meanZ + outliersTolerance*stdZ) | y < (meanZ - outliersTolerance*stdZ), na.rm = T)) {
#                 cat(f, 'test\n')
#                 test[[f]] <- replace(y, y > (meanZ + outliersTolerance*stdZ), (meanZ + outliersTolerance*stdZ)) # 
#                 test[[f]] <- replace(y, y < (meanZ - outliersTolerance*stdZ), (meanZ - outliersTolerance*stdZ)) # 
#             }
#         }
#     }
    
    if ( any(is.na(train[[f]])) | any(is.na(test[[f]]))) {
        x <- train[[f]]
        y <- test[[f]]
        medianX <- median(x,na.rm = T)
        train[[f]] <- replace(x, is.na(x), medianX)
        test[[f]] <- replace(y, is.na(y), medianX)
        # median !
    }
#     levelsX <- unique(train[[f]])
#     if (length(levelsX) == 1) {
#         print(f + ' after')
#         train[[f]] <- NULL
#         test[[f]] <- NULL
#         c <- c + 1
#     }
}
cat('\ndropped ', c, ' columns!')

feature.names <- names(train)[2:ncol(train)-1]

# folds <- createFolds(factor(train$Happy), k = 2, list = TRUE, returnTrain = TRUE)
# train <- train[sample(nrow(train), 40000),]

f <- factor(train$Happy, levels=unique(train$Happy), labels = c(1, 0, 2))
train$Happy <- as.numeric(levels(f))[f]

cat("Making train and validation matrices\n")
inTraining <- createDataPartition(factor(train$Happy), p = 0.80, list = FALSE)
validation  <- train[-inTraining,]
train <- train[ inTraining,]

gc()

# check stratification...
table(train$Happy)
table(validation$Happy)

# as DMatrix
dtrain <- xgb.DMatrix(data.matrix(train[, feature.names]), 
                      label=train$Happy) # , missing = NaN
dval <- xgb.DMatrix(data.matrix(validation[,feature.names]), 
                    label=validation$Happy) # , missing = NaN

evalerror <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    
    n <- length(preds)
    diff <- (labels - preds) * 5
    diff <- replace(diff, diff==0, 50)
    diff <- replace(diff, diff==5, 100)
    diff <- replace(diff, diff==10, 5)
    diff <- replace(diff, diff==100, 10)
    
    err <- sum(diff) / (n * 50)
    return(list(metric = "error", value = err))
}

watchlist <- list(eval = dval, train = dtrain)

param <- list(  objective           = "multi:softmax", 
                num_class           = 3,
                eta                 = 0.05, # changed from default of 0.001
                max_depth           = 7, # changed from default of 14
#                 subsample           = 0.7, # changed from default of 0.6
#                 colsample_bytree    = 0.7, # changed from default of 0.6
                eval_metric         = evalerror
                #                 scale_pos_weight    = (sum_neg/sum_pos)
                #                 max_delta_step      = 3
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 500, # changed from 50
                    verbose             = 1, 
                    early.stop.round    = 30,
                    watchlist           = watchlist,
                    maximize            = TRUE)

# importance_matrix <- xgb.importance(feature.names, model = clf)
# xgb.plot.importance(importance_matrix, numberOfClusters = 4)

submission <- data.frame(ID=test$ID)
submission$Happy <- predict(clf, data.matrix(test[,feature.names]))
submission$Happy[submission$Happy==0] = "Not Happy"
submission$Happy[submission$Happy==1] = "Pretty Happy"
submission$Happy[submission$Happy==2] = "Very Happy"

# benchmark = 
cat("saving the submission file\n")
write_csv(submission, "xgboost_submission.csv")
