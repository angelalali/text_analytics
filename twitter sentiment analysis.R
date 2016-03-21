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
# # needed to use cfs()
# EnsurePackage("FSelector")
# # FSelector dependency
# EnsurePackage('RWekajars')


# 2. set working directory
getwd()
path = 'C:/Users/ali3/Documents/data science/cisco data science program/cisco DSP 2016/capstone/sample data from kaggle & such/'
setwd(path)
getwd() # confirm that you have changed the working directory

# load the prepared data; not partitioned yet
load("sentiment140 data.RData")
save.image("sentiment140 data.RData")

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
                                         p = .002,
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

# study how the original data look like
table(data.train$Sentiment)
# ok, so pretty much 50-50; but sentiment 1 (positive) is always more

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

tdm.train.matrix = as.matrix(tdm.train)

# matrix.train = as.matrix(tdm.train)
# 
# 
# # build the document-term matrix (dtm)
# # create_matrix will create a document by term matrix; so document will be rows, and terms be columns
# dtm.train = create_matrix(data.train$SentimentText,
#                     language = "english",
#                     removeStopwords = T,
#                     removeNumbers = T,
#                     stemWords = T,
#                     removePunctuation = T,
#                     toLower = T,
#                     weighting = weightTfIdf)
# 
# # train the model using naive bayes
# dtm.train.matrix = as.matrix(dtm.train)






#################################### dimension reduction w LSA #################################
# dimension reduction leveraging Latent Semantic Analysis (LSA) using singular value decomposition (SVD) 
# because of following two reasons:
#   1. As there are k dimensions (number of documents/posts/tweets) and n terms (number of unique words), 
#   therefore, it will be difficult to analyze all these at the same time.
#   2. TDM is essentially a very sparse matrix (99% sparseness is very common). So to remove 
#   sparseness, LSA is used.

# LSA using SVD
# weightTfIdf {tm}: Weight by Term Frequency - Inverse Document Frequency
# Weight a term-document matrix by term frequency - inverse document frequency.
# weightTfIdf(m, normalize = TRUE)
# m: A TermDocumentMatrix in term frequency format.
# normalize: A Boolean value indicating whether the term frequencies should be normalized.
# tfidf.train = weightTfIdf(dtm.train)
# since the input m has to be a termdocumentmatrix obj, must input dtm.train, which is directly 
# created by the create_matrix()

# as reminder: 
# dtm.train.matrix = as.matrix(dtm.train), and
# tdm.train.matrix = t(dtm.train.matrix)
# tfidf.train.matrix = as.matrix(tfidf.train)

# lsa {lsa}: Create a vector space with Latent Semantic Analysis (LSA)
# Calculates a latent semantic space from a given document-term matrix.
# lsa( x, dims=dimcalc_share() )
# x: a document-term matrix (recommeded to be of class textmatrix), containing documents in colums, terms in rows and occurrence frequencies in the cells.
# dims: either the number of dimensions or a configuring function.
# lsa.train = lsa(tdm.train, dimcalc_share(share = 0.8))
lsa.train = lsa(tdm.train, dimcalc_share(share = 0.8)) # default: share = 0.5
# lsa.train.tk = as.data.frame(lsa.train$tk)
# lsa.train.dk = as.data.frame(lsa.train$dk)
# lsa.train.sk = as.data.frame(lsa.train$sk)

# mar = c(A,B,C,D);
# A = bottom;  B = left;  C = up;  D = right
par(mar=c(5,5,0,2))
plot(lsa.train$sk) # so pick an index value where 75% of the variance is explained
# so by picking at the elbow location, set the k number (manual for now)
k = 500
# lsa.train.sk.topk = sum(lsa.train.sk[1:k, 1]) / sum(lsa.train.sk[,1])

# project the training data onto a new dimension w the weights
# fold_in {lsa}: 
# Ex-post folding-in of textmatrices into an existing latent semantic space
# LSAspace: a latent semantic space generated by createLSAspace.
# docvecs: a textmatrix.
projected.train = fold_in(docvecs = tdm.train.matrix, LSAspace = lsa.train)[1:k,]
dim(projected.train)
# make this into a matrix; pass into naivebayes & class label
projected.train.matrix = matrix(projected.train, 
                                nrow = dim(projected.train)[1],
                                ncol = dim(projected.train)[2])
dim(projected.train.matrix)
length(data.train$Sentiment)

# train naive classifier
NBclassifier = naiveBayes(t(projected.train.matrix), data.train$Sentiment)

# train a svm
SVMclassifier = svm(x = t(projected.train.matrix), y = data.train$Sentiment)

# tune svm
tuneSVM = tune.svm(x = t(projected.train.matrix), 
                   y = data.train$Sentiment,
                   cost = 2^7,
                   epsilon = 0)

tuneSVM$best.parameters
tuneSVM$best.model

SVM.tuned = tuneSVM$best.model

summary(tuneSVM)



sum(is.na(tdm.train.matrix))



#################################### cross validation ##################################
# create a container so that you can use it for the X-validation fxn
# create_container {RTextTools}: creates a container for training, classifying, and analyzing documents.
# Given a DocumentTermMatrix from the tm package and corresponding document labels, creates a container 
#     of class matrix_container-class that can be used for training and classification 
#     (i.e. train_model, train_models, classify_model, classify_models)
# create_container(matrix, labels, trainSize=NULL, testSize=NULL, virgin)

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

# train SVM model using the container
SVMbyContainer = train_model(container = cv.container, algorithm = "SVM")
# Error in svm.default(x = container@training_matrix, y = container@training_codes,  : 
#   x and y don't match.


cv.SVM = cross_validate(container = cv.container, 
                        nfold = 5, 
                        algorithm = "SVM", 
                        verbose = T    # set to T: it'll provide descriptive output about the training process
                        )

cv.Boosting = cross_validate(cv.container, nfold = 5, algorithm = "BOOSTING")

cv.Bagging = cross_validate(cv.container, nfold = 5, algorithm = "BAGGING")

cv.RF = cross_validate(cv.container, nfold = 5, algorithm = "RF")

# all above cross validation technique returns an error that says:
# Error in na.fail.default(y) : missing values in object

cv.MaxEnt = cross_validate(cv.container, nfold = 5, algorithm = "MAXENT")






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
# 
# # build the document-term matrix (dtm)
# dtm.val = create_matrix(data.val$SentimentText,
#                         language = "english",
#                         removeStopwords = T,
#                         removeNumbers = T,
#                         stemWords = T)

# must convert the term-doc matrix into an actual matrix
# matrix.val = as.matrix(tdm.val)
tdm.val.matrix = as.matrix(tdm.val)
dim(tdm.val.matrix)
# lsa.val = lsa(t(dtm.val.matrix), dimcalc_share(share = 0.8))
lsa.val = lsa(tdm.val, dimcalc_share())

projected.val = fold_in(tdm.val.matrix, lsa.val)[1:k,]
dim(projected.val)
dim(data.val)
# make this into a matrix; pass into naivebayes & clas label
projected.val.matrix = matrix(projected.val,
                              nrow = dim(projected.val)[1],
                              ncol = dim(projected.val)[2])


# predict using naiveBayes built above using training data
# data.val$pred.sentiment = predict(NBclassifier, newdata = t(matrix.val), type = "class")
data.val$pred.sentiment = predict(NBclassifier, 
                                  newdata = t(projected.val.matrix), 
                                  type = "class")

# predict using SVM
data.val$pred.sentiment = predict(SVMclassifier, 
                                  newdata = t(projected.val.matrix), 
                                  type = "class")

# predict using tuned SVM
data.val$pred.sentiment = predict(SVM.tuned, 
                                  newdata = t(projected.val.matrix), 
                                  type = "class",
                                  na.action = na.fail)

# confusion table to validate your data set
table(data.val$pred.sentiment, data.val$Sentiment)
#    0  1
# 0 23 29
# 1 45 56





############################# test on test set #######################################################
# create a corpus that contain the validation tweets
corpus.test = Corpus(VectorSource(data.test$SentimentText))

# create a term document matrix for validation set
tdm.test = TermDocumentMatrix(
  corpus.test,
  control = list(
    removePunctuation = TRUE,
    stopwords = stopwords(kind = "en"),
    removeNumbers = TRUE, 
    tolower = TRUE,
    weighting = weightTfIdf)
)

# must convert the term-doc matrix into an actual matrix
# matrix.val = as.matrix(tdm.val)
tdm.test.matrix = as.matrix(tdm.test)
dim(tdm.test.matrix)
lsa.test = lsa(tdm.test, dimcalc_share())

projected.test = fold_in(tdm.test.matrix, lsa.test)[1:k,]
dim(projected.test)
# make this into a matrix; pass into naivebayes & clas label
projected.test.matrix = matrix(projected.test,
                              nrow = dim(projected.test)[1],
                              ncol = dim(projected.test)[2])


# predict using naiveBayes built above using training data
# data.val$pred.sentiment = predict(NBclassifier, newdata = t(matrix.val), type = "class")
data.test$pred.sentiment = predict(NBclassifier, 
                                  newdata = t(projected.test.matrix), 
                                  type = "class")
# confusion table to validate your data set
table(data.test$pred.sentiment, data.test$Sentiment)






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





#################################### create word cloud #################################
# must transpose the document-term-matrix into term-document-matrix in order for 
# the word cloud to work correctly
tdm.train.matrix = t(dtm.train.matrix)

# sort the matrix by their term frequencies 
tdm.train.sorted = sort(rowSums(tdm.train.matrix), decreasing = T)
df.train = data.frame(words = names(tdm.train.sorted), freq = tdm.train.sorted)
wordcloud(df.train$words, 
          df.train$freq, 
          max.words = 300, 
          colors = brewer.pal(8, "Dark2"),
          scale = c(1, 0.5), 
          # both numbers indicate font size (i think); if its (3,5) or (1,1), it'll be too 
          # cramped and wont fit in a screen nicely as a round circle, but rather squeezed 
          # into a square; so far (1,0.5) has been the best combo
          random.order = F)
# wordcloud(words,freq,scale=c(4,.5),min.freq=3,max.words=Inf,
# random.order=TRUE, random.color=FALSE, rot.per=.1,
# colors="black",ordered.colors=FALSE,use.r.layout=FALSE,
# fixed.asp=TRUE, ...)




# matrix.train.sorted = sort(rowSums(matrix.train), decreasing = T)
# df.train = data.frame(words = names(matrix.train.sorted), freq = matrix.train.sorted)
# wordcloud(df.train$words, 
#           df.train$freq, 
#           max.words = 300, 
#           colors = brewer.pal(10, "Dark2"),
#           scale = c(3, 0.5),
#           random.order = F)













######################## cluster analysisafter lsa w hclust() & kmean() ###################
#randomly creating 150 clusters with k-means
k150.train.tk = kmeans(scale(lsa.train.tk), 
                       centers=150, 
                       nstart=20)
c150.train.tk = aggregate(cbind(V1,V2,V3) ~ k150.train.tk$cluster,
                          data = lsa.train.tk,
                          FUN = mean)

k150.train.dk = kmeans(scale(lsa.train.dk), 
                       centers=50, 
                       nstart=20)
c150.train.dk = aggregate(cbind(V1,V2,V3) ~ k150.train.dk$cluster,
                          data = lsa.train.dk,
                          FUN = mean)

#hierarchical clustering to find optimal # of clusters for c150.train.tk 
dist.train.c150 = dist(scale(c150.train.tk[, -1]))
hclust.train = hclust(d,method='ward.D')
par(oma=c(0,0,2,0))
plot(hclust.train, hang = -1)
# hang: The fraction of the plot height by which labels should hang below the rest of the plot.
#       A negative value will cause the labels to hang down from 0.

# rect.hclust {stats}: Draw Rectangles Around Hierarchical Clusters
# Draws rectangles around the branches of a dendrogram highlighting the corresponding clusters. 
#     First the dendrogram is cut at a certain level, then a rectangle is drawn around selected branches.
# rect.hclust(tree, k = NULL, which = NULL, x = NULL, h = NULL, border = 2, cluster = NULL)
# k, h: Scalar. Cut the dendrogram such that either exactly k clusters are produced or by cutting at height h.
rect.hclust(hclust.train, h=20, border="blue") #2
rect.hclust(hclust.train, h=12, border="cyan") #2
rect.hclust(hclust.train, h=35, border="red") #2
# 2 clusters seem to be the optimal number
k2.train.tk = kmeans(scale(lsa.train.tk), 
                       centers=2, 
                       nstart=20)
c2.train.tk = aggregate(cbind(V1,V2,V3) ~ k150.train.tk$cluster,
                          data = lsa.train.tk,
                          FUN = mean)

#hierarchical clustering to find optimal no of clusters for c150.train.dk 
dist.train.c150 = dist(scale(c150.train.dk[,-1]))
hclust.train = hclust(dist.train.c150, method='ward.D')
plot(hclust.train, hang = -1)
rect.hclust(hclust.train, h=5, border="blue") #7
rect.hclust(hclust.train, h=15, border="red") #2
rect.hclust(hclust.train, h=8, border="green") #4
# so pick 2 again, since thats the most optimal number
k2.train.dk = kmeans(scale(lsa.train.dk), 
                     centers=2, 
                     nstart=20)
c2.train.dk = aggregate(cbind(V1,V2,V3) ~ k150.train.dk$cluster,
                        data = lsa.train.dk,
                        FUN = mean)

# R does not by default allot any space to the outer margins; see par("oma")
# this will solve the problem of plot titles getting cut off
par(mar=c(4,8,2,4))
# par(xpd=NA,oma=c(0,0,2,0))
clusplot(lsa.train.dk, 
         k2.train.dk$cluster, 
         color = T,
         shade = T,
         labels = 2,
         lines = 0)
# clusplot {cluster}: Bivariate Cluster Plot (of a Partitioning Object)
# Draws a 2-dimensional "clusplot" (clustering plot) on the current graphics device. 
#     The generic function has a default and a partition method.
# clusplot(x, ...)
# x: an R object, here, specifically an object of class "partition", 
#   e.g. created by one of the functions pam, clara, or fanny.
## S3 method for class 'partition'
# clusplot(x, main = NULL, dist = NULL, ...)
# main: title for the plot; when NULL (by default), a title is constructed, using x$call.
# dist: when x does not have a diss nor a data component, e.g., for pam(dist(*), keep.diss=FALSE), 
#       dist must specify the dissimilarity for the clusplot.

# Result of clustering on lsa.train.tk
# reminder: tdm.train.sorted = sort(rowSums(tdm.train.matrix), decreasing = T)
# df.train = data.frame(words = names(tdm.train.sorted), freq = tdm.train.sorted)
k2.train.tk1 = df.train[k2.train.tk$cluster == 1,]
k2.train.tk2 = df.train[k2.train.tk$cluster == 2,]
wordcloud(k2.train.tk1$words, 
          k2.train.tk1$freq, 
          max.words = 200, 
          colors = brewer.pal(8, "Dark2"),
          scale = c(1, 0.5), 
          random.order = F)
wordcloud(k2.train.tk2$words, 
          k2.train.tk2$freq, 
          max.words = 200, 
          colors = brewer.pal(8, "Dark2"),
          scale = c(1, 0.5), 
          random.order = F)

# making sure the plot title is not cut off
par(mar=c(0,0,2,0))
clusplot(lsa.train.tk, 
         k2.train.tk$cluster, 
         color = T,
         shade = T,
         labels = 2,
         lines = 0)


# lsa.train.tk
# why only 1:3? (or 1:6? which is originally in the article)
lsa.train.tk3 = data.frame(words = rownames(lsa.train.tk), lsa.train.tk[, 1:3])

# plot the dimension 1 vs dimension 2
plot(lsa.train.tk3$V1, lsa.train.tk3$V2)
# replace the dots/circles w the texts, so u have a better understanding
text(lsa.train.tk3$V1, lsa.train.tk3$V2, label=lsa.train.tk3$words)

plot(lsa.train.tk3$V2, lsa.train.tk3$V3)
text(lsa.train.tk3$V2, lsa.train.tk3$V3, label=lsa.train.tk3$words)

plot(lsa.train.tk3$V1, lsa.train.tk3$V3)
text(lsa.train.tk3$V1, lsa.train.tk3$V3, label=lsa.train.tk3$words)


#Result of clustering on lsa.train.dk
lsa.train.dk = cbind(1:nrow(lsa.train.dk), lsa.train.dk)
k2.train.dk=lsa.train.dk[k2.train.dk$cluster == 1,]
k2.train.dk=lsa.train.dk[k2.train.dk$cluster == 2,]

colnames(lsa.train.dk)[1]="doc"

plot(lsa.train.dk$V1, lsa.train.dk$V2)
text(lsa.train.dk$V1,lsa.train.dk$V2,label = lsa.train.dk$doc)

plot(lsa.train.dk$V2,lsa.train.dk$V3)
text(lsa.train.dk$V2,lsa.train.dk$V3,label=lsa.train.dk$doc)

plot(lsa.train.dk$V1,lsa.train.dk$V3)
text(lsa.train.dk$V1,lsa.train.dk$V3,label=lsa.train.dk$doc)


#################################### lsa subsetting ####################################
#subset
#FSelector
lsa.train.tk2 = cbind(lsa.train.tk,k2.train.tk$cluster)
names(lsa.train.tk2)[51]="cluster_tk"
lsa.train.tk2$cluster_tk=as.factor(lsa.train.tk2$cluster_tk)

# cfs {FSelector}: cfs filter
# The algorithm finds attribute subset using correlation and entropy measures for continous 
#     and discrete data.
# cfs(formula, data)

# prlblems:
# the RWekajars() is no longer available, and that package is a dependency for FSelector 
# package, which is needed to use the cfs() function.
subset.lsa.tk = cfs(cluster_tk ~ ., lsa.train.tk2)
cfs.formula = as.simple.formula(subset.lsa.tk, "cluster_tk")
print(f)

lsa.train.dk2 = cbind(lsa.train.dk, k2.train.dk$cluster)
names(lsa.train.dk2)[53] = "cluster_dk"
lsa.train.dk2$cluster_dk=as.factor(lsa.train.dk2$cluster_dk)

subset.lsa.dk=cfs(cluster_dk~.,lsa.train.dk2)
formula = as.simple.formula(subset.lsa.dk, "cluster")
print(formula)




















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

