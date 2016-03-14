##################################################################################################
# setting up working environment
# 1. load all necessary packages
# 2. set working directory
# 3. read training data & partition data into train, validation, test
##################################################################################################

# 1. load all necessary packages
EnsurePackage <- function(packageName)
{
  x <- as.character(packageName)
  if(!require(x, character.only = TRUE))
  {
    install.packages(pkgs = x, 
                     repos = "http://cran.r-project.org")
    require(x, character.only = TRUE)
  }
} 
# EnsurePackage("flexclust") # used for kmeans clustering
# # EnsurePackage("randomForest")
# # needed for stratified random sampling fxn createDataPartition()
# EnsurePackage('caret')
EnsurePackage('data.table')
EnsurePackage("RTextTools")
EnsurePackage("e1071")
EnsurePackage("openxlsx")
EnsurePackage("xlsx")
# text mining package
EnsurePackage("tm")
# caret package needed for confusionMatrix()
EnsurePackage("caret")
EnsurePackage("wordcloud")


# 2. set working directory
getwd()
path = 'C:/Users/ali3/Documents/cisco data science program/cisco DSP 2016/capstone/sample data from kaggle & such/'
setwd(path)
getwd() # confirm that you have changed the working directory

# load the prepared data;
# data already partitioned and prepared 
load("sentiment140 data dot001.RData")
save.image("sentiment140 data dot5.RData")

#################################################################################################
# if you dont use load the data, then u can use the following code:

####################### if you wanna use sentiment140 data ######################################
# sentiment140 data set contains 1M rows of labeled data
# read training data & partition data into train, validation, test
# read the data
# data.all = read.csv("kaggle-train (w label).csv")
data.orig = read.csv("sentiment140 data labeled.csv")


# rid of the last 3 columns & 1st column; select the columns that contain the word "Sentiment"
data.orig = data.orig[, grepl("Sentiment", colnames(data.orig))]
# get rid of the column "sentimentsource" which is the 2nd column
data.orig = data.orig[, -2]
# re-factor the label column
data.orig$Sentiment = factor(data.orig$Sentiment)
# make tweet column into character
data.orig$SentimentText = as.character(data.orig$SentimentText)

####################### if u wanna use kaggle University of Michigan data #######################
data.orig = read.csv("kaggle-train (w label).csv")
# this data set contains about 7.5k rows of data
# kaggle competition website: https://inclass.kaggle.com/c/si650winter11
# kaggle competition title: UMICH SI650 - Sentiment Classification
# problem w this kaggle training set is that.... the tweets are looping/duplicating :/ 
# NOT RECOMMENDED


###################### shrink the data size u have to work with #################################
data.all = data.orig[createDataPartition(y = data.orig$Sentiment,
                                         p = .005,
                                         list = F),]
## sample data in stratified manner 
# 70% training, 15% validation, 15% test
# following method will return index value
data.part = createDataPartition(y = data.all$Sentiment,
                                p = .7,
                                list = F)
data.train = data.all[data.part, ]
temp = data.all[-data.part, ]
data.part = createDataPartition(y = data.all$Sentiment,
                                p = .5,
                                list = F)
data.val = temp[data.part, ]
data.test = temp[-data.part, ]

rm(data.part)
rm(temp)


##################################################################################################
# build the document-term matrix (dtm)
dtm.train = create_matrix(data.train$SentimentText,
                    language = "english",
                    removeStopwords = T,
                    removeNumbers = T,
                    stemWords = T)

# train the model using naive bayes
dtm.train.matrix = as.matrix(dtm.train)


# train naive classifier
NBclassifier = naiveBayes(dtm.train.matrix, data.train$Sentiment)

# # create a corpus that contain the training tweets
# corpus.train = Corpus(VectorSource(data.train$SentimentText))
# 
# # create a term document matrix
# tdm.train = TermDocumentMatrix(
#   corpus.train,
#   control = list(
#     removePunctuation = TRUE,
#     stopwords = stopwords(kind = "en"),
#     removeNumbers = TRUE, 
#     tolower = TRUE)
# )
# 
# # cant use code in line below b/c matrix is sparce and taking up too much computing power (mainly memory)
# matrix.train = as.matrix(tdm.train)
# 
# # build naive bayes classifier
# # > dim(matrix.train)
# # [1] 2739  736
# # so must use transpose of the matrix
# NBclassifier = naiveBayes(t(matrix.train), data.train$Sentiment)

############################# test on validation set ##############################################
# create a corpus that contain the validation tweets
corpus.val = Corpus(VectorSource(data.val$SentimentText))

# create a term document matrix for validation set
tdm.val = TermDocumentMatrix(
  corpus.val,
  control = list(
    removePunctuation = TRUE,
    stopwords = stopwords(kind = "en"),
    removeNumbers = TRUE, 
    tolower = TRUE)
)

# build the document-term matrix (dtm)
dtm.val = create_matrix(data.val$SentimentText,
                        language = "english",
                        removeStopwords = T,
                        removeNumbers = T,
                        stemWords = T)

# must convert the term-doc matrix into an actual matrix
# matrix.val = as.matrix(tdm.val)
dtm.val.matrix = as.matrix(dtm.val)
# predict using naiveBayes built above using training data
# data.val$pred.sentiment = predict(NBclassifier, newdata = t(matrix.val), type = "class")
data.val$pred.sentiment = predict(NBclassifier, newdata = dtm.val.matrix, type = "class")
# confusion table to validate your data set
table(data.val$pred.sentiment, data.val$Sentiment)
#    0  1
# 0 23 29
# 1 45 56





############################# test on test set #######################################################
# create a corpus that contain the test tweets
corpus.test = Corpus(VectorSource(data.test$SentimentText))

# create a term document matrix for test set
tdm.test = TermDocumentMatrix(
  corpus.test,
  control = list(
    removePunctuation = TRUE,
    stopwords = stopwords(kind = "en"),
    removeNumbers = TRUE, 
    tolower = TRUE)
)

# must convert the term-doc matrix into an actual matrix
matrix.test = as.matrix(tdm.test)
# predict using naiveBayes built above using training data
data.test$pred.sentiment = predict(NBclassifier, t(matrix.test), type = "class")
# confusion table to test your data set
table(data.test$pred.sentiment, data.test$Sentiment)
#    0  1
# 0 30 33
# 1 50 48

############################# see the performance #####################################################
pred.eval.val = confusionMatrix(data.val$pred.sentiment, data.val$Sentiment)
names(pred.eval.val$byClass)
# [1] "Sensitivity"          "Specificity"          "Pos Pred Value"       "Neg Pred Value"       "Prevalence"      "Detection Rate"      
# [7] "Detection Prevalence" "Balanced Accuracy" 

perf.val1 = as.data.frame(pred.eval.val$overall)
colnames(perf.val1) = 'value'
perf.val1
#                     value
# Accuracy       0.53974359
# Kappa          0.07645119
# AccuracyLower  0.50403027
# AccuracyUpper  0.57515542
# AccuracyNull   0.53205128
# AccuracyPValue 0.34679370
# McnemarPValue  0.83279993

perf.val2 = as.data.frame(pred.eval.val$byClass)
colnames(perf.val2) = 'value'
perf.val2
#                         value
# Sensitivity          0.3382353
# Specificity          0.6588235
# Pos Pred Value       0.4423077
# Neg Pred Value       0.5544554
# Prevalence           0.4444444
# Detection Rate       0.1503268
# Detection Prevalence 0.3398693
# Balanced Accuracy    0.4985294

pred.eval.val
# Confusion Matrix and Statistics
# 
#           Reference
# Prediction  0  1
#           0 23 29
#           1 45 56
# 
# Accuracy : 0.5163          
# 95% CI : (0.4342, 0.5978)
# No Information Rate : 0.5556          
# P-Value [Acc > NIR] : 0.85480         
# 
# Kappa : -0.003          
# Mcnemar's Test P-Value : 0.08121         
# 
# Sensitivity : 0.3382          
# Specificity : 0.6588          
# Pos Pred Value : 0.4423          
# Neg Pred Value : 0.5545          
# Prevalence : 0.4444          
# Detection Rate : 0.1503          
# Detection Prevalence : 0.3399          
# Balanced Accuracy : 0.4985          
# 
# 'Positive' Class : 0   
# positive:
# an optional character string for the factor level that corresponds to a "positive" result 
# (if that makes sense for your data). If there are only two factor levels, the first level 
# will be used as the "positive" result.

############################# performance test (not working) #######################################
# following method does not work b/c ROCR performance() only work on continuous prediction
# but ur prediction is classification
# however, saving the code for illustration purpose
#####################################################################################################

# load the package to calculate performance
EnsurePackage("ROCR")

pred.val = prediction(data.val$pred.sentiment, data.val$Sentiment)
pred.val = prediction(as.numeric(data.val$pred.sentiment), as.numeric(data.val$Sentiment))

# store the performance
val.perf = performance(pred.val,
                       measure = "prec",
                       x.measure = "rec"
)



#################################### create word cloud #################################









