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
# matrix.train.sorted = sort(rowSums(matrix.train), decreasing = T)
# df.train = data.frame(words = names(matrix.train.sorted), freq = matrix.train.sorted)
# wordcloud(df.train$words, 
#           df.train$freq, 
#           max.words = 300, 
#           colors = brewer.pal(10, "Dark2"),
#           scale = c(3, 0.5),
#           random.order = F)

# # build naive bayes classifier
# # > dim(matrix.train)
# # [1] 2739  736
# # so must use transpose of the matrix
# NBclassifier = naiveBayes(t(matrix.train), data.train$Sentiment)


################################################################################################
#################################### dimension reduction w LSA #################################
################################################################################################
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
tfidf.train = weightTfIdf(dtm.train)
# since the input m has to be a termdocumentmatrix obj, must input dtm.train, which is directly 
# created by the create_matrix()

# as reminder: 
# dtm.train.matrix = as.matrix(dtm.train), and
# tdm.train.matrix = t(dtm.train.matrix)
tfidf.train.matrix = as.matrix(tfidf.train)

# lsa {lsa}: Create a vector space with Latent Semantic Analysis (LSA)
# Calculates a latent semantic space from a given document-term matrix.
# lsa( x, dims=dimcalc_share() )
# x: a document-term matrix (recommeded to be of class textmatrix), containing documents in colums, terms in rows and occurrence frequencies in the cells.
# dims: either the number of dimensions or a configuring function.
lsa.train = lsa(tdm.train.matrix, dimcalc_share(share = 0.8))
lsa.train.tk = as.data.frame(lsa.train$tk)
lsa.train.dk = as.data.frame(lsa.train$dk)
lsa.train.sk = as.data.frame(lsa.train$sk)


#################################### lsa clustering ####################################
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
# par(oma=c(0,0,2,0))
par(xpd=NA,oma=c(0,0,2,0))
clusplot(lsa.train.dk, 
         k2.train.dk$cluster, 
         color = T,
         shade = T,
         labels = 2,
         lines = 0
         )
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
par(xpd=NA,oma=c(0,0,2,0))
clusplot(lsa.train.tk, 
         k2.train.tk$cluster, 
         color = T,
         shade = T,
         labels = 2,
         lines = 0
)


# lsa.train.tk
# learn the first 3 dimensions only
lsa.train.tk3 = data.frame(words = rownames(lsa.train.tk), lsa.train.tk[, 1:3])

# plot the dimension 1 vs dimension 2
plot(lsa.train.tk3$V1, lsa.train.tk3$V2)
# replace the dots/circles w the texts, so u have a better understanding
text(lsa.train.tk3$V1, lsa.train.tk3$V2, label=lsa.train.tk3$words)

plot(lsa.train.tk3$V2, lsa.train.tk3$V3)
text(lsa.train.tk3$V2, lsa.train.tk3$V3, label=lsa.train.tk3$words)

plot(lsa.train.tk3$V1, lsa.train.tk3$V3)
text(lsa.train.tk3$V1, lsa.train.tk3$V3, label=lsa.train.tk3$words)











# train naive classifier
NBclassifier = naiveBayes(dtm.train.matrix, data.train$Sentiment)

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

