
install.packages('xgboost')
library(dplyr)
library(tidyverse)
library(data.table)
library(rpart.plot)
library(ggplot2)
library(randomForest)
library(caret) #confusionMatrix
library(reprtree)
library(randomForestExplainer)
library(xgboost) 
library(ROCR) 
library(skimr)
library(stargazer)
library(caret)
library(e1071)
library(corrplot)

diagnoses <- read.csv('/Users/namhyunjin/Byon8/MIMIC/sql/DIAGNOSES_ICD.csv')
D_ICD_DIAGNOSES <-  read.csv('/Users/namhyunjin/Byon8/MIMIC/sql/D_ICD_DIAGNOSES.csv')
D_LABITEMS <- read.csv('/Users/namhyunjin/Byon8/MIMIC/sql/D_LABITEMS.csv')
LABEVENTS <- fread('/Users/namhyunjin/Byon8/MIMIC/sql/LABEVENTS.csv', header = T, sep = ',')
ADMISSIONS <-  read.csv('/Users/namhyunjin/Byon8/MIMIC/sql/ADMISSIONS.csv')
PATIENTS <- read.csv('/Users/namhyunjin/Byon8/MIMIC/sql/PATIENTS.csv')
weight <- read.csv('/Users/namhyunjin/Byon8/MIMIC/sql/weight.csv')

set.seed(1)
diabetes.id <- diagnoses %>%
  filter(ICD9_CODE %in% paste0(250,rep(seq(from=0,to=9),each=4),rep(c(0,2),10)) ) %>%
  dplyr::select(HADM_ID)
notdiabetes.id <- diagnoses %>%
  filter(ICD9_CODE %in% setdiff(diagnoses$ICD9_CODE,paste0(250,rep(seq(from=0,to=9),each=4),rep(c(0,2),10)))) %>%
  dplyr::select(HADM_ID) 

##################################################################
######################## Making Data set #########################
##################################################################

#LABEVENTS
LABEVENTS.new <- LABEVENTS  %>%
  dplyr::select(HADM_ID,ITEMID,VALUE) %>%
  group_by(HADM_ID, ITEMID) %>%
  mutate(VALUE = as.numeric(as.character(VALUE))) %>%
  summarise_at(vars(VALUE), funs(mean(., na.rm=TRUE))) %>%
  ungroup() %>%
  spread("ITEMID","VALUE",fill = NA) 

label.frame <- D_LABITEMS %>%
  filter(ITEMID %in% colnames(LABEVENTS.new)[-1]) %>%
  dplyr::select(ITEMID,LABEL,LOINC_CODE)

new.colnames <- data.frame(ITEMID =  colnames(LABEVENTS.new)[-1]) %>% 
  mutate(ITEMID = as.numeric(as.character(ITEMID)))%>%
  inner_join(label.frame, by='ITEMID')

colnames(LABEVENTS.new) <- c('HADM_ID',paste(as.character(new.colnames$LABEL),'!',new.colnames$ITEMID,'!',new.colnames$LOINC_CODE)) 

#ADMISSIONS
admissions.new <- ADMISSIONS %>%
  filter(HADM_ID %in% LABEVENTS.new$HADM_ID) %>%
  dplyr::select(SUBJECT_ID,HADM_ID,ETHNICITY,ADMITTIME) #ADMITTIME,MARITAL_STATUS,RELIGION

#PATIENTS
admissions.patients.new <- PATIENTS %>%
  dplyr::select(SUBJECT_ID,GENDER,DOB) %>% #DOB
  inner_join(admissions.new, by='SUBJECT_ID') %>%
  mutate(DOB = as.Date(DOB,format='%Y-%m-%d'),
         ADMITTIME = as.Date(ADMITTIME,format='%Y-%m-%d'),
         Age = as.numeric(ADMITTIME - DOB )%/% (365)) %>%
  mutate(Age = ifelse(Age >200 , 0, Age)) %>%
  dplyr::select(-DOB,-ADMITTIME)  

#Weight
BMI = function(height,weight){return((weight/((height/100)^2)))}
weight <- weight %>%
  mutate(BMI = round(BMI(height_first,weight_first),2))

#Merge all the data
data <- admissions.patients.new %>%
  left_join(LABEVENTS.new, by='HADM_ID') %>%
  left_join(weight, by=c('HADM_ID'='hadm_id'))  %>%
  #left_join(complication, by='HADM_ID') %>%
  mutate(Diabetes = ifelse(HADM_ID %in% diabetes.id$HADM_ID , 1, 0)) %>%
  dplyr::select(-height_first,-weight_first)


#Discard if all the valuse are NA
data <-  data %>%
  dplyr::select(names(which(apply(is.na(data),2,sum) !=dim(data)[1]))) 
D.data <- data %>% dplyr::select(-SUBJECT_ID,-HADM_ID) 



##################################################################
#################### Description Statistics ######################
##################################################################
D.data %>% select(Diabetes) %>% table
1-0.2144136
12933/(47385+12933)
#look over specific person
random.sample <- sample(data$HADM_ID, 1)
data %>%
  filter( HADM_ID == random.sample)
diagnoses %>%
  filter( HADM_ID == random.sample)

D.data$ETHNICITY %>% table

D.discriptive <- skim(D.data) 
D.discriptive1 <-  skimr::kable(D.discriptive, format=latex, digits = getOption("digits"),
                                row.names = NA, col.names = NA, align = NULL, caption = NULL,
                                format.args = list(), escape = TRUE)

##################################################################
######################## Desicion Tree ###########################
##################################################################
tree.data <- D.data
tree.data[is.na(tree.data)] <- 0

# Sample
set.seed(1)
sample = sample.int(n = nrow(tree.data), size = floor(.8*nrow(tree.data)), replace = F)
tree.train = tree.data[sample, ]
tree.test = tree.data[-sample, ] 

# Step1: Begin with a small cp.
rpart.fit <- rpart(Diabetes~., data=tree.train, method="class",control = rpart.control(cp = 0, maxdepth=4)) 

printcp(rpart.fit) %>% stargazer(summary=F)
plotcp(rpart.fit)
printcp(rpart.fit)

# Step2: Pick the tree size that minimizes misclassification rate (i.e. prediction error).
# Prediction error rate in training data = Root node error * rel error * 100%
# Prediction error rate in cross-validation = Root node error * xerror * 100%
# Hence we want the cp value (with a simpler tree) that minimizes the xerror.
bestcp <- rpart.fit$cptable[which.min(rpart.fit$cptable[,"xerror"]),"CP"]
rpart.fit$cptable

# Step3: Prune the tree using the best cp.
tree.pruned <- prune(rpart.fit, cp = bestcp)
#printcp(tree.pruned), plotcp(rpart.fit), printcp(rpart.fit)
print(Sys.time())
par(mfrow=c(1,2)) # two plots on one page 
rsq.rpart(rpart.fit) # visualize cross-validation results   
summary(tree.pruned)

# Draw a tree plot
only_count <- function(x, labs, digits, varlen){paste(x$frame$n)}
boxcols <- c("grey", "red")[tree.pruned$frame$yval]

par(mfrow=c(1,1)) 
prp(rpart.fit, faclen = 0, cex = 0.8, node.fun= only_count, box.col = boxcols)
printcp(rpart.fit)

par(mfrow=c(1,1)) 
prp(tree.pruned, faclen = 0, cex = 0.8, node.fun= only_count, box.col = boxcols)
printcp(tree.pruned)

# Prediction error rate in training data = Root node error * rel error * 100%
# Prediction error rate in cross-validation = Root node error * xerror * 100%
printcp(tree.pruned)
tree.train.err <- 0.21459*0.83332
1-tree.train.err
tree.cross.err <- 0.21459*0.84462
1-tree.cross.err

# confusion matrix (test data)
tree.conf.matrix <- table(tree.test[,ncol(tree.test)],predict(tree.pruned,tree.test,type="class"))
tree.y <- tree.test[,ncol(tree.test)]
tree.hat <- predict(tree.pruned,tree.test,type="class")
tree.conf <- confusionMatrix(tree.hat, tree.y %>% as.factor,  positive='1')

#Importance plot
tree.imp.var.names <-  names(tree.pruned$variable.importance)[1:10]
tree.imp.var.names.frame <- str_split(tree.imp.var.names,'!', simplify=T)
tree.imp.table <- data.frame(Feature= paste(tree.imp.var.names.frame[,1],tree.imp.var.names.frame[,3]),
                             Gain =(tree.pruned$variable.importance)[1:10])
xgb.plot.importance(tree.imp.table[1:10,] %>% data.table, rel_to_first = TRUE, xlab = "Relative importance")

#x <-tree.imp.table$Gain
#normalized <- (x-min(x))/(max(x)-min(x))



# mat : is a matrix of data
# ... : further arguments to pass to the native R cor.test function
cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}

rownames(tree.imp.table[1:10,])
# matrix of the p-value of the correlation
cor.tree <- tree.train %>% dplyr::select(rownames(tree.imp.table[1:10,])) 
cor.tree.names <- str_split(rownames(tree.imp.table[1:10,]),'!', simplify=T)
names(cor.tree) <- paste(cor.tree.names[1:10,1],cor.tree.names[1:10,3])
p.mat.tree <- cor.mtest(cor.tree)
M.tree<-cor(cor.tree)

col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(M.tree, method="color", col=col(200),  
         type="upper", #order="hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45,tl.cex=0.7,cl.cex=0.5, number.cex=0.5, #Text label color and rotation
         # Combine with significance
         p.mat = p.mat.tree, sig.level = 0.01, insig = "blank", 
         # hide correlation coefficient on the principal diagonal
         diag=F 
)



##################################################################
######################## Random Forest ###########################
##################################################################
#https://datascienceplus.com/random-forests-in-r/
rf.data <- D.data 
rf.data[is.na(rf.data)] <- 0

names(rf.data) <- make.names(names(rf.data))
rf.data <- rf.data %>%
  mutate( Diabetes = as.factor(Diabetes))

# Sample
rf.train = rf.data[sample, ]
rf.test = rf.data[-sample, ] 
100-16.04

set.seed(1)
#diabetes.rf.fit <- randomForest(Diabetes~. , data=rf.train, ntree=300 , mtry=sqrt(ncol(rf.train)), importance=T)
rf.fit <- randomForest(Diabetes~. , data=rf.train, ntree=300 , mtry=sqrt(ncol(rf.train)), importance=T)
rf.fit1 <- randomForest(Diabetes~. , data=rf.train, ntree=300 , mtry=25, importance=T)

#Plot importance
rf.imp <- importance(rf.fit1) %>% as.data.frame %>%  mutate(var =var.names)%>% arrange(desc(MeanDecreaseGini)) 
rf.imp.var.names <-  (rf.imp$var)[1:10]
rf.imp.var.names.frame <- c("Glucose   2345-7",
                            "Age",
                            "% Hemoglobin A1c   4548-4",
                            "Creatinine    2160-0",
                            "Urea Nitrogen    3094-0",
                            "Glucose   2339-0",
                            "Potassium   2823-3",
                            "MCH   785-6",
                            "BMI   ",
                            "RDW   788-0")
rf.imp.table <- data.frame(Feature= rf.imp.var.names.frame,
                           Gain =diabetes.rf.imp$MeanDecreaseGini[1:10])
xgb.plot.importance(rf.imp.table[1:10,] %>% data.table, rel_to_first = TRUE, xlab = "Relative importance")
#plot dbug d


# matrix of the p-value of the correlation
cor.rf <- rf.train %>% dplyr::select(rf.imp.var.names) 
names(cor.rf) <-rf.imp.var.names.frame
p.mat.rf <- cor.mtest(cor.rf)
M.rf<-cor(cor.rf)

corrplot(M.rf, method="color", col=col(200),  
         type="upper",# order="hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45,tl.cex=0.7,cl.cex=0.5, number.cex=0.5, #Text label color and rotation
         # Combine with significance
         p.mat = p.mat.rf, sig.level = 0.01, insig = "blank", 
         # hide correlation coefficient on the principal diagonal
         diag=FALSE 
)


# confusion matrix (test data)
rf.y <- rf.test[,ncol(rf.test)]
rf.hat <- predict(rf.fit1,rf.test,type="class")
rf.conf <- confusionMatrix(rf.hat, rf.y %>% as.factor,  positive='1')



#Tune

set.seed(1)
rf.tunegrid1 <- expand.grid(.mtry=c(17:26))
rf.tunegrid <- expand.grid(.mtry=sqrt(ncol(rf.train)))
rf_gridsearch <- train(Diabetes~.,
                       data=rf.train, method="rf", metric="Accuracy",
                       trControl = trainControl(method = "oob"),
                       tuneGrid=rf.tunegrid)
rf_gridsearch1 <- train(Diabetes~.,
                        data=rf.train, method="rf", metric="Accuracy",
                        trControl = trainControl(method = "oob"),
                        tuneGrid=rf.tunegrid1)
print(rf_gridsearch)
plot(rf_gridsearch)





##################################################################
########################### XGBoost ##############################
##################################################################
#https://www.kaggle.com/rtatman/machine-learning-with-xgboost-in-r
one.hot <- function(one.hot.data){
  numeric.data <- one.hot.data %>%
    dplyr::select_if(is.numeric)
  one.hot.matrix <- model.matrix(as.formula("~ ETHNICITY+ GENDER"), one.hot.data)[,-1]
  result.data <- cbind(one.hot.matrix,numeric.data)
}

XG.data <- one.hot(D.data)
XG.data[is.na(XG.data)] <- 0


set.seed(1) # Set a random seed so that same sample can be reproduced in future runs
which.D <- (which(colnames(XG.data)== 'Diabetes'))

# Sample
train = XG.data[sample, ] #just the samples
test  = XG.data[-sample, ]  #everything but the samples

train_y = train[,which.D]
train_x = train[, -which.D]

test_y = test[,which.D] 
test_x = test[, -which.D]

dtrain = xgb.DMatrix(data =  as.matrix(train_x), label = train_y )
dtest = xgb.DMatrix(data =  as.matrix(test_x), label = test_y)

negative_cases <- sum(train_y == 0)
postive_cases <- sum(train_y == 1)


#Step1: Tune eta
#Tuning with Caret Packages 
tune_grid <- expand.grid(
  nrounds = 300,
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(4),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 10, # with n folds 
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = TRUE, # FALSE for reproducible results
  seeds=1
  
)

set.seed(1)
xgb_tune <- caret::train(
  x = train_x,
  y = train_y %>% as.factor,
  trControl = tune_control,
  tuneGrid = tune_grid,
  method = "xgbTree",
  verbose = TRUE
)

# helper function for the plots
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$Accuracy, probs = 1), min(x$results$Accuracy))) +
    theme_bw()
}

tuneplot(xgb_tune)

#Step2: Tune min_child_weight
tune_grid2 <- expand.grid(
  nrounds = 300,
  eta = xgb_tune$bestTune$eta,
  max_depth = 4,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3),
  subsample = 1
)

set.seed(1)
xgb_tune2 <- caret::train(
  x = train_x,
  y = train_y %>% as.factor,
  trControl = tune_control,
  tuneGrid = tune_grid2,
  method = "xgbTree",
  verbose = TRUE
)#0.8469347

#Step4: Tune Gamma
tune_grid3 <- expand.grid(
  nrounds = 300,
  eta = 0.3,
  max_depth = 4,
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  colsample_bytree = 1,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = 1
)

xgb_tune3 <- caret::train(
  x = train_x,
  y = train_y %>% as.factor,
  trControl = tune_control,
  tuneGrid = tune_grid3,
  method = "xgbTree",
  verbose = TRUE
)#0.8475771

tuneplot(xgb_tune3)

#Step4: Tune Max depth
tune_grid4 <- expand.grid(
  nrounds = 300,
  eta = 0.3,
  max_depth = c(3,4,5,6),
  gamma = 0.7,
  colsample_bytree = 1,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = 1
)

xgb_tune4 <- caret::train(
  x = train_x,
  y = train_y %>% as.factor,
  trControl = tune_control,
  tuneGrid = tune_grid4,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune4)

tune_grid5 <- expand.grid(
  nrounds = 300,
  eta = 0.3,
  max_depth = c(7,8,9,10),
  gamma = 0.7,
  colsample_bytree = 1,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = 1
)

xgb_tune5 <- caret::train(
  x = train_x,
  y = train_y %>% as.factor,
  trControl = tune_control,
  tuneGrid = tune_grid5,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune5)

xgb_tune5
xgb_tune4.5 <- xgb_tune4
xgb_tune4.5$results <- rbind(xgb_tune4$results, xgb_tune5$results)
tuneplot(xgb_tune4.5)



#Final model
final_grid <- expand.grid(
  nrounds = xgb_tune2$bestTune$nrounds,
  eta = xgb_tune2$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = xgb_tune2$bestTune$gamma,
  colsample_bytree = xgb_tune2$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune2$bestTune$subsample)

train_control <- caret::trainControl(
  method = "none",
  verboseIter = FALSE, # no training log
  allowParallel = TRUE,# FALSE for reproducible results 
  seeds=1
)

set.seed(1)
xgb_model <- caret::train(
  x = train_x,
  y = train_y %>% as.factor,
  trControl = train_control,
  tuneGrid = final_grid,
  method = "xgbTree",
  verbose = TRUE
)


summary(xgb_model)
xgb_predict <- predict(xgb_model, newdata = test_x)
xgb_confusion <- table(xgb_predict,test_y)
sum(diag(xgb_confusion))/sum(xgb_confusion)


xgb_tune3$bestTune


##XGBoost
params <- list(nrounds = 300,
               max_depth = 4,
               eta = 0.3,
               gamma = 0,
               colsample_bytree = 1,
               min_child_weight = 1,
               subsample = 1)

D.xgb.trin <- xgb.train(params = params,
                        nfold=10,
                        nrounds = 300,
                        data = dtrain,
                        objective = "binary:logistic",
                        watchlist = list(val=dtest, train=dtrain),
                        metrics = list("error","auc"),
                        seed=1) #     300  0.154012    0.082459
1-0.154012 #0.845988

D.xgb.cv <- xgb.cv(params = params, 
                   nfold=10,
                   nround = 300,
                   data = dtrain,
                   objective = "binary:logistic",
                   metrics = list("error","auc"),
                   seed=1)
# iter train_error_mean train_error_std train_auc_mean train_auc_std test_error_mean test_error_std test_auc_mean test_auc_std
# 300        0.0801156    0.0005240775      0.9641491  0.0004920359       0.1539768    0.002393683     0.8734740  0.005456315
#0.8460232
#  1                 0.8469347  0.4991396

params1 <- list(nrounds = 300,
                max_depth = 4,
                eta = 0.3,
                gamma = 0.7,
                colsample_bytree = 1,
                min_child_weight = 1,
                subsample = 1)
bbm
D.xgb.trin1 <- xgb.train(params = params1,
                         nfold=10,
                         nrounds = 300,
                         data = dtrain,
                         objective = "binary:logistic",
                         watchlist = list(val=dtest, train=dtrain),
                         metrics = list("error","auc"),
                         seed=1) #     300  0.150862    0.084304
#0.849138


D.xgb.cv1 <- xgb.cv(params = params1, 
                    nfold=10,
                    nround = 300,
                    data = dtrain,
                    objective = "binary:logistic",
                    metrics = list("error","auc"),
                    seed=1)
#    iter train_error_mean train_error_std train_auc_mean train_auc_std test_error_mean test_error_std test_auc_mean test_auc_std
#     300        0.0803250    0.0009053397      0.9637885  0.0006506108       0.1524635    0.004403562     0.8730802  0.003304229
#0.8475365
#0.70   0.8475771  0.5018122


model1 <- xgboost(data = dtrain, # the data      
                  params = params1, 
                  seed=1,
                  nrounds=300,
                  objective = "binary:logistic"# the objective function
) 




###
params2 <- list(nrounds = 300,
                max_depth = 6,
                eta = 0.3,
                gamma = 0.7,
                colsample_bytree = 1,
                min_child_weight = 1,
                subsample = 1)


D.xgb.train2 <- xgb.train(params = params2,
                          nfold=10,
                          nrounds = 300,
                          data = dtrain,
                          objective = "binary:logistic",
                          watchlist = list(val=dtest, train=dtrain),
                          metrics = list("error","auc"),
                          seed=1) #  300  0.154095    0.017429
#0.849138
#0.845905

D.xgb.cv2 <- xgb.cv(params = params2, 
                    nfold=10,
                    nround = 300,
                    data = dtrain,
                    objective = "binary:logistic",
                    metrics = list("error","auc"),
                    seed=1)
#    iter train_error_mean train_error_std train_auc_mean train_auc_std test_error_mean test_error_std test_auc_mean test_auc_std
#     300        0.0803250    0.0009053397      0.9637885  0.0006506108       0.1524635    0.004403562     0.8730802  0.003304229
#     300        0.0132839    0.0009430603      0.9988901  0.0001444808       0.1521536    0.003514881     0.8734645  0.004000607

#0.8478464
#0.70   0.8475771  0.5018122


model2 <- xgboost(data = dtrain, # the data      
                  params = params2, 
                  seed=1,
                  nrounds=300,
                  objective = "binary:logistic"# the objective function
) 



1-0.017429
length(model2.prediction)
model2.pre <- predict(model2, dtest,type="class")
model2.prediction <- as.numeric(model2.pre > 0.5)
xg.conf <- confusionMatrix(model2.prediction%>% as.factor, test_y %>% as.factor,  positive='1')


#Variance Importance
xg.importance_matrix <- xgb.importance(model = model1)

xg.imp.var.names <-  xg.importance_matrix$Feature
xg.imp.var.names.frame <- str_split(xg.imp.var.names,'!', simplify=T)
xg.imp.table <- data.frame(Feature= paste(xg.imp.var.names.frame[,1],xg.imp.var.names.frame[,3]),
                           Gain =(xg.importance_matrix$Gain))
xgb.plot.importance(xg.imp.table[1:10,] %>% data.table, rel_to_first = TRUE, xlab = "Relative importance")
xg.imp.table[1:10,]


# matrix of the p-value of the correlation
cor.xg <- train_x %>% dplyr::select(xg.imp.var.names[1:10]) 
names(cor.xg) <- paste(xg.imp.var.names.frame[1:10,1],xg.imp.var.names.frame[1:10,3])
p.mat.xg <- cor.mtest(cor.xg)
M.xg<-cor(cor.xg)

dim(p.mat.xg)

par(mfrow=c(1,1))
corrplot(M.xg, method="color", col=col(200),  
         type="upper", #order="hclust", 
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45,tl.cex=0.7,cl.cex=0.5, number.cex=0.5, #Text label color and rotation
         # Combine with significance
         p.mat = p.mat.xg, sig.level = 0.01, insig = "blank", 
         # hide correlation coefficient on the principal diagonal
         diag=FALSE 
)

    

#Plot the error rate
plot(data.frame(x=seq(from=1,to=300,by=1), D.xgb.cv1$evaluation_log[,6]), 
     type='l',col=3, ylim = c(0.14,0.26), xlim= c(0,300),
     ylab="Validation Error",xlab="Number of tree")
points(rf.fit$err.rate[,1], type='l')
abline(h=tree.cross.err, col=2)
legend("top", c('Random Forest','Decision Tree', 'Boosting'),
       col=1:4,cex=0.8,fill=1:4,horiz=F)

tuneplot(xgb_tune3)
png("image.png", width = 800, height = 600)



