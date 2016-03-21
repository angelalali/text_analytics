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
EnsurePackage("lsa")
# needed for clusplot()
EnsurePackage("cluster")


# 2. set working directory
getwd()
path = 'C:/Users/ali3/Documents/data science/cisco data science program/cisco DSP 2016/capstone/sample data from kaggle & such/'
setwd(path)

data.orig = read.csv("sentiment140 data subset.csv")

# rid of the last 3 columns & 1st column; select the columns that contain the word "Sentiment"
data.orig = data.orig[, grepl("Sentiment", colnames(data.orig))]
# get rid of the column "sentimentsource" which is the 2nd column
data.orig = data.orig[, -2]
# re-factor the label column
data.orig$Sentiment = factor(data.orig$Sentiment)
# make tweet column into character
data.orig$SentimentText = as.character(data.orig$SentimentText)


###################### shrink the data size u have to work with #################################
data.all = data.orig[complete.cases(data.orig),]

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

# study how the original data look like
table(data.train$Sentiment)

table(data.val$Sentiment)



##################################################################################################
# create a corpus that contain the training tweets
corpus.train = Corpus(VectorSource(data.train$SentimentText))

# create a term document matrix
# this will create a term by document matrix; so terms are rows and docs are columns
tdm.train = TermDocumentMatrix(
  corpus.train,
  control = list(
    removePunctuation = TRUE,
    stopwords = stopwords(kind = "en"),
    removeNumbers = TRUE, 
    tolower = TRUE,
    weighting = weightTfIdf)
)
# somehow i always get the warnings:
# Warning message:
#   In weighting(x) : empty document(s): 177 294

tdm.train.matrix = as.matrix(tdm.train)




#################################### dimension reduction w LSA #################################
lsa.train = lsa(tdm.train, dimcalc_share()) # default: share = 0.5

par(mar=c(5,5,0,2))
plot(lsa.train$sk) # so pick an index value where 75% of the variance is explained
# so by picking at the elbow location, set the k number (manual for now)
k = 60
# hmm ok but wehre to use this k value?...

# project the training data onto a new dimension w the weights
projected.train = fold_in(docvecs = tdm.train.matrix, LSAspace = lsa.train)[1:k, ]
dim(projected.train)
# make this into a matrix; pass into naivebayes & class label
projected.train.matrix = matrix(projected.train, 
                                nrow = dim(projected.train)[1],
                                ncol = dim(projected.train)[2])
dim(projected.train.matrix)
length(data.train$Sentiment)

# # train naive classifier
NBclassifier = naiveBayes(t(projected.train.matrix), data.train$Sentiment)

sum(is.na(tdm.train.matrix))
# returns 0



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
    tolower = TRUE,
    weighting = weightTfIdf)
)

tdm.val.matrix = as.matrix(tdm.val)
dim(tdm.val.matrix)
# lsa.val = lsa(t(dtm.val.matrix), dimcalc_share(share = 0.8))
lsa.val = lsa(tdm.val, dimcalc_share())

projected.val = fold_in(tdm.val.matrix, lsa.val)[1:k,]
dim(projected.val)
# make this into a matrix; pass into naivebayes & clas label
projected.val.matrix = matrix(projected.val,
                              nrow = dim(projected.val)[1],
                              ncol = dim(projected.val)[2])


# predict using naiveBayes built above using training data
# data.val$pred.sentiment = predict(NBclassifier, newdata = t(matrix.val), type = "class")
data.val$pred.sentiment = predict(NBclassifier, 
                                  newdata = t(projected.val.matrix), 
                                  type = "class")
# confusion table to validate your data set
table(data.val$pred.sentiment, data.val$Sentiment)





#################################### cross validation ##################################
# create a container so that you can use it for the X-validation fxn
cv.container = create_container( 
  # matrix = tdm.train,
  # matrix = projected.train,
  matrix = t(projected.train),
  labels = data.train$Sentiment,
  trainSize = floor(.7 * ncol(projected.train)),
  testSize = ceiling(.3 * ncol(projected.train)),
  virgin = F    # F means the test data is also labeled
)
# projected.train has row = term & col = doc; so we need to transpose it so that row = docs/tweets

cv.SVM = cross_validate(container = cv.container, 
                        nfold = 5, 
                        algorithm = "SVM", 
                        verbose = T    # set to T: it'll provide descriptive output about the training process
)
# Error in na.fail.default(y) : missing values in object

cv.Boosting = cross_validate(cv.container, nfold = 5, algorithm = "BOOSTING")

cv.Bagging = cross_validate(cv.container, nfold = 5, algorithm = "BAGGING")

cv.RF = cross_validate(cv.container, nfold = 5, algorithm = "RF")

# all above cross validation technique returns an error that says:
# Error in na.fail.default(y) : missing values in object

cv.MaxEnt = cross_validate(cv.container, nfold = 5, algorithm = "MAXENT")
