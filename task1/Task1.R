X_inp <- read.csv("~/ETH/HS20/Advanced ML/Project 1/X_train.csv")
Y <- read.csv("~/ETH/HS20/Advanced ML/Project 1/y_train.csv")
X_test_imp <-read.csv("~/ETH/HS20/Advanced ML/Project 1/X_test.csv")
Y <- Y[,2]
un <- array(0,length(X_inp))
for(i in 1:length(X_inp)){un[i] <- length(unique(X_inp[,i]))}
X_inp <- X_inp[ , un>4]
X_test_imp <- X_test_imp[,un>10]
X <- X_inp[,2:length(X_inp)]
X_test <- X_test_imp[,2:length(X_test_imp)]

library(caret)
library(randomForest)
X_train <- X
Y_train <- Y
normParam <- preProcess(X_train)
X_train <- predict(normParam,X_train)
X_test <- predict(normParam,X_test)
X_train[is.na(X_train)] <- 0
X_test[is.na(X_test)] <- 0
featureselection <- randomForest(X_train,Y_train,importance = TRUE)
X_train_feat <- X_train[,(featureselection$importance[,1]>0.15)==TRUE]
X_test_feat <- X_test[,(featureselection$importance[,1]>0.15)==TRUE]

predi <- randomForest(X_train_feat,Y_train,ntree=500)
Y_test <- predict(predi,newdata=X_test)

Y_test <- cbind(X_test_imp[,1],Y_test)
colnames(Y_test) <- c("id","y")
write.csv(Y_test,file="Y_test.csv",row.names = F)
