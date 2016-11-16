# ======================================================================
# R code for Assignment #2 in STATS5703W
# Description: (1) Compute PCA and ICA on handwritten digits dataset and
# summarize the findings and (2) cluster analysis of seeds dataset.
# Written by: Philippe Paradis
# Additional credits: Part of the code was taken from Shirley
# Mills' STATS5703W course notes.
# ======================================================================
# The following packages are required (except 'parallel', as long as
# 'run.parallel' is set to FALSE):
#   fastICA, caret, randomForest, parallel, lattice, e1071, xtable,
#   plyr, deldir, combinat
#
# Please set the following global variables according to your system:
# 'work.dir' - Working directory in which this file is located
# 'data.dir' - Data directory containing the handwritten digit datasets
# 'code.dir' - Directory with code from STATS5703W
# 'save.to.pdf'   - Set to FALSE for interactive plotting
# 'run.parallel'  - Set to FALSE to turn off parallel computing 
# 'use.max.cores' - Maximum number of cores to use for parallel computing
#                   or 0 to use all cores available.
switch(Sys.info()[['sysname']],
Windows =
  {
     ######################################
     # EDIT YOUR SETTINGS FOR WINDOWS HERE
     ######################################
     work.dir <- "D:/proj/stat5703w/ass2"
     data.dir <- "D:/proj/stat5703w/data"
     code.dir <- "D:/proj/stat5703w/code"
     save.to.pdf <- TRUE
     run.parallel <- TRUE
     use.max.cores <- 0
  },
Linux =
  {
     ######################################
     # EDIT YOUR SETTINGS FOR LINUX HERE
     ######################################
     work.dir <- "/proj/stat5703w/ass2"
     data.dir <- "/proj/stat5703w/data"
     code.dir <- "/proj/stat5703w/code"
     save.to.pdf <- TRUE
     run.parallel <- FALSE
     use.max.cores <- 0
  },
Darwin =
  {
     ######################################
     # EDIT YOUR SETTINGS FOR MAC HERE
     ######################################
     work.dir <- "/proj/stat5703w/ass2"
     data.dir <- "/proj/stat5703w/data"
     code.dir <- "/proj/stat5703w/code"
     save.to.pdf <- TRUE
     run.parallel <- TRUE
     use.max.cores <- 0
  })

# Misc global variables
global.cl <- NULL

# Set 'work.amount' to "easy", "medium" or "hard" to determine
# the amount of ICA work to do. This will change the size of the
# training set fed into fastICA, which can be quite slow as the
# size of the training set increases. Warning: selecting 'hard'
# could result in hours-long compute time, unless your system has
# is very powerful.
work.ica.amount <- "hard"

# This sets the training/testing dataset sizes for ICA. Note that
# those are sizes *per digit*.
switch(work.ica.amount,
       easy   = {global.ica.train.size <-  80; global.ica.test.size <-  20},
       medium = {global.ica.train.size <- 200; global.ica.test.size <-  70},
       hard   = {global.ica.train.size <- 400; global.ica.test.size <- 140})

# Load required libraries
library(stats)
library(fastICA)
library(xtable)
library(randomForest)
library(class)
library(e1071)
library(caret)
library(plyr)
library(deldir)
library(combinat)

# Move to working directory and create "figures" subdirectory
# into which the plots will be saved
setwd(work.dir)
dir.create("figures", showWarnings = FALSE)

#======================================================================
# FUNCTION DEFINITIONS
#======================================================================

# This function calls 'pdf(...)' if 'save.to.pdf' is TRUE,
# otherwise it just calls 'dev.new()'
prep.out <- function(...)
{
   if (save.to.pdf)
      pdf(...)
   else
      dev.new()
}

# Source: Shirley Mills STAT5703W
plot.digits <- function(digits) {
   if (dev.cur() == 1) {
      x11(width = 6, height = 5)    # Open a graphics window of given size  
   }
   # Create a plot matrix with 144 subplots - plot in row-wise order
   layout(matrix(1:144, 12, 12, byrow = TRUE))
   # No margin (see ?par)
   oldpar <- par(mar = c(0, 0, 0, 0))
   for (i in 1:144) {
      # xaxt = "n", yaxt = "n" - no axes
      image(matrix(digits[i,],16,16)[,16:1], xaxt = "n", yaxt = "n",
            col = gray((255:0)/255))
   }
   par(oldpar)
}

# General images drawing function with variable length. It will adjust
# number of rows and columns automatically. It also accepts images of
# varying dimensions.
plot.images <- function(data, len=10, image_size=c(16,16)) {
   dim_width <- min(len, 10)
   dim_height <- ceiling(len/10)
   if (dev.cur() == 1) {
      x11(width = dim_width*8/10, height = dim_height*8/10)
   }
   # Create a plot matrix with dim_width * dim_height subplots
   layout(matrix(1:(dim_width*dim_height),
                 dim_height, dim_width, byrow = TRUE))
   # No margin (see ?par)
   oldpar <- par(mar = c(0, 0, 0, 0))
   # xaxt = "n", yaxt = "n" - no axes
   for (n in 1:len) {
      image(matrix(data[n,], image_size[1], image_size[2])[,image_size[2]:1],
            xaxt = "n", yaxt = "n",
            col = gray((255:0)/255))
   }
   par(oldpar)
}


#####################################
# Helper function for classification
#####################################
classify <- function(pipeline, train.var, train.labels, test.var, test.labels)
{
   measures <- c()
   for (classifier in pipeline) {
      classifier.name <- classifier[[1]]
      classifier.func <- classifier[[2]]
      
      cat(paste("Fitting model '", classifier.name, "'...  ", sep=""))
      
      if (identical(classifier.func, randomForest)) {
         fit <- randomForest(train.var, train.labels,
                             xtest=test.var, ytest=test.labels,
                             ntree=200, keep.forest=TRUE)
         pred <- predict(fit, test.var)      
      } else if (identical(classifier.func, knn)) {
         fit <- knn(train.var, test.var, train.labels, k=7)
         pred <- fit
      } else {
         fit <- classifier.func(train.var, train.labels)
         pred <- predict(fit, test.var)
      }
      cm <- confusionMatrix(pred, test.labels)
      acc <- cm$overall[[1]]
      measures <- rbind(measures, list(name=classifier.name, accuracy=acc))
      cat(paste(signif(acc, 3), " accuracy\n", sep=""))
   }
   measures
}
##########################################
# Function to classify digits without any
# data dimensionality reduction.
##########################################
perf.measures <- c()

run_classifiers <- function(num_train=300, num_test=100)
{
  num.training.examples <- num_train
  num.testing.examples <- num_test
  train <- c()
  for (d in 0:9) {
    a <- 1
    b <- num.training.examples
    train <- rbind(train, cbind(d.digits[[d+1]][a:b,], d))
  }
  test <- c()
  for (d in 0:9) {
    a <- num.training.examples + 1
    b <- num.training.examples + num.testing.examples
    test <- rbind(test, cbind(d.digits[[d+1]][a:b,], d))
  }
  
  # Shuffle train and test
  set.seed(1990)
  train <- train[sample(nrow(train)),]
  test <- test[sample(nrow(test)),]
  
  # Labels
  train.labels <- as.factor(train[,257])
  train.var <- train[,-257]
  test.labels <- as.factor(test[,257])
  test.var <- test[,-257]
  
  # Fix colnames (having empty colname strings cause errors)
  colnames(train.var) <- NULL
  colnames(test.var) <- NULL
  
  pipeline <- list(c("Random Forest", randomForest),
                   c("k-NN", knn),
                   c("Naive Bayes", naiveBayes),
                   c("SVM", svm))
  classify(pipeline, train.var, train.labels, test.var, test.labels)
}

########################################################
# Function which first applies PCA to reduce the digits
# dataset dimensionality, then classifies digits and
# measures performance.
########################################################

# We will keep only the top 'num.pca.components' PCA components

run_classifiers_pca <- function(num.pca = 50, num_train=300, num_test=100)
{
  num.training.examples <- num_train
  num.testing.examples <- num_test
  
  train.init <- c()
  for (d in 0:9) {
    a <- 1
    b <- num.training.examples
    train.init <- rbind(train.init, cbind(d.digits[[d+1]][a:b,], d))
  }
  
  # Perform PCA on 'train.init'
  pca.train <- prcomp(train.init[, -257], center = FALSE)
  
  # Transform 'train.init'
  train.transf <- cbind(pca.train$x[ , 1:num.pca], train.init[, 257])
  
  # Load test dataset and apply PCA transform to it
  test.transf <- c()
  for (d in 0:9) {
    a <- num.training.examples + 1
    b <- num.training.examples + num.testing.examples
    prod <- d.digits[[d+1]][a:b, ] %*% pca.train$rotation[, 1:num.pca]
    test.transf <- rbind(test.transf, cbind(prod, d))
  }
  
  # Shuffle train and test
  set.seed(1990)
  train <- train.transf[sample(nrow(train.transf)), ]
  test <- test.transf[sample(nrow(test.transf)), ]
  
  # Labels
  train.labels <- as.factor(train[, num.pca+1])
  train.var <- train[, -(num.pca+1)]
  test.labels <- as.factor(test[, num.pca+1])
  test.var <- test[, -(num.pca+1)]
  
  # Fix colnames (having empty colname strings cause errors)
  colnames(train.var) <- NULL
  colnames(test.var) <- NULL
  
  str.pca <- paste("PCA (", num.pca, ")", sep="")
  pipeline <- list(c(paste(str.pca, "+ Random Forest"), randomForest),
                   c(paste(str.pca, "+ k-NN"), knn),
                   c(paste(str.pca, "+ Naive Bayes"), naiveBayes),
                   c(paste(str.pca, "+ SVM"), svm))
  classify(pipeline, train.var, train.labels, test.var, test.labels)
}

########################################################
# Function which first applies ICA to reduce the digits
# dataset dimensionality, then classifies digits and
# measures performance.
########################################################
# We will keep only the top 'num.pca.components' PCA components
run_classifiers_ica <- function(n.comp = 50, num_train=0, num_test=0)
{
  if (num_train > 0) {
    num.training.examples <- num_train
  } else {
    num.training.examples <- global.ica.train.size
  }
  if (num_test > 0) {
    num.testing.examples <- num_test
  } else {
    num.testing.examples <- global.ica.test.size
  }
  total <- num.training.examples + num.testing.examples
  
  # Combined all digits into one training dataset
  combined <- c()
  for (d in 0:9) {
    a <- 1
    b <- total
    combined <- rbind(combined, d.digits[[d+1]][a:b,])
  }
  
  set.seed(0) #for reproducibility
  w.init <- matrix(rnorm(n.comp*n.comp), n.comp, n.comp)
  t1 <- proc.time()
  # Perform ICA on 'combined' (includes all digits, as well
  # as trainin set and testing set)
  ica.train <- fastICA(t(combined),
                       n.comp,
                       alg.typ = "parallel",
                       fun = "logcosh",
                       alpha = 1,
                       method = "R",
                       row.norm = FALSE,
                       maxit = 200,
                       tol = 0.0001,
                       verbose = TRUE,
                       w.init=w.init)
  cat(paste("fastICA (n.comp = ", n.comp, ") elapsed time: ",
            (proc.time()-t1)[["elapsed"]], " seconds\n", sep=""))
  
  train.transf <- c()
  for (d in 0:9) {
    a <- 1 + d*total
    b <- num.training.examples + d*total
    train.transf <- rbind(train.transf, cbind(t(ica.train$A)[a:b,], d))
  }
  
  test.transf <- c()
  for (d in 0:9) {
    a <- num.training.examples + 1 + d*total
    b <- num.training.examples + num.testing.examples + d*total
    test.transf <- rbind(test.transf, cbind(t(ica.train$A)[a:b,], d))
  }
  
  # Shuffle train and test
  set.seed(1990)
  train <- train.transf[sample(nrow(train.transf)), ]
  test <- test.transf[sample(nrow(test.transf)), ]
  
  # Labels
  train.labels <- as.factor(train[, n.comp+1])
  train.var <- train[, -(n.comp+1)]
  test.labels <- as.factor(test[, n.comp+1])
  test.var <- test[, -(n.comp+1)]
  
  # Fix colnames (having empty colname strings cause errors)
  colnames(train.var) <- NULL
  colnames(test.var) <- NULL
  
  str.ica <- paste("ICA (", n.comp, ")", sep="")
  pipeline <- list(c(paste(str.ica, "+ Random Forest"), randomForest),
                   c(paste(str.ica, "+ k-NN"), knn),
                   c(paste(str.ica, "+ Naive Bayes"), naiveBayes),
                   c(paste(str.ica, "+ SVM"), svm))
  classify(pipeline, train.var, train.labels, test.var, test.labels)
}


#################################################################
# Cleanup the parallel computing environment variables. Although
# it's not necessarily to do an explicit cleanup, the parallel
# workers can be shutdown at any timeby calling this function.
#################################################################
clean_parallel_env <- function()
{
   if (!is.null(global.cl))
      tryCatch(stopCluster(global.cl), error=function(e) FALSE)
   global.cl <<- NULL
}

##############################################################
# Setup parallel environment (if 'run.parallel' is TRUE and
# if more than 1 core was detected).
##############################################################
setup_parallel_env <- function()
{
   if (!run.parallel)
      return(NULL)
   
   library(parallel)
   avail.cores <- detectCores()
   cat(paste("Detected ", avail.cores, " cores...\n", sep=""))
   if (avail.cores <= 1) {
      run.parallel <- FALSE
      global.cl <<- NULL
   } else {
      if (use.max.cores > 0) {
         num.cores <- min(avail.cores, use.max.cores)
         cat(paste("Launching ", num.cores,
                   " worker threads in parallel...\n", sep=""))
      } else {
         num.cores <- avail.cores
      }
      clean_parallel_env()
      cl <- makePSOCKcluster(num.cores, outfile="",
                             useXDR=FALSE)
      #setDefaultCluster(cl)
      # Export the necessary variables and functions
      clusterExport(cl, c("d.digits", "combined.d.digits", "randomForest",
                          "global.ica.train.size", "global.ica.test.size",
                          "run_classifiers_pca", "run_classifiers_ica",
                          "knn", "naiveBayes", "svm",
                          "classify", "confusionMatrix", "fastICA"))
      global.cl <<- cl
   }
}

#################################################################
# This function returns FALSE if at least one of the connections
# to the workers was shut down, otherwise it returns TRUE.
#################################################################
check_parallel_connections <- function()
{
   tryCatch(all(sapply(sapply(global.cl, "[")["con",], isOpen)),
            error=function(e) FALSE)
}

##################################################################
# Just call this like 'lapply'. Parallelism will be automatically
# handled by this function, including initialization/cleanup of
# worker threads. The function will proceed sequentially if
# parallelismm is disabled or if the system is single core.
##################################################################
parallel_lapply <- function(X, FUN)
{
   if (run.parallel) {
      # Check if 'global.cl' was defined
      if (is.null(global.cl))
         setup_parallel_env()
      # Check if the connections in 'global.cl' were closed down
      else if (!check_parallel_connections())
         setup_parallel_env()
      if (!is.null(global.cl)) {
         cat(paste("Running ", length(X)," tasks in parallel...\n", sep=""))
         flush.console()
         results <- tryCatch(parLapplyLB(global.cl, X, FUN),
                             error=function(e){clean_parallel_env(); stop(e)})
         return(results)
      }
   }
   return(lapply(X, FUN))
}


##################################################################
#
# QUESTION 1: PCA & ICA for handwritten digits.
#
##################################################################


#######################################################
# Load handwritten digits training datasets
#######################################################
d.file <- {}
d.digits <- c({}, {}, {})
for (i in 0:9) {
   d.file[i+1] <- paste(data.dir, "/train_", i, ".dat", sep = "")
   d.digits[[i+1]] <- matrix(scan(d.file[i+1], sep = ","),
                             ncol = 256, byrow = T)
}

#######################################################
# Perform PCA
#######################################################
# Perform PCA analysis on each training set independently
# i.e. on each digit independently
pc.digits <- {}
prep.out("figures/digits-pca.pdf", height=4)
par(mfrow=c(2,5), mar=c(4.1,2.1, 2.1, 1.1))
for (i in 0:9) {
   # Important: 'center' is set to FALSE. This makes analysis
   # much simpler, especially since our data is already fairly well
   # centered.
   pc.digits[[i+1]] <- prcomp(d.digits[[i+1]], center = FALSE)
   plot(pc.digits[[i+1]], col = heat.colors(10), main = i)
   
   # Uncomment the following to see a summary of the PCA results...
   # print(summary(pc.digits[[i+1]]))
}
dev.off()

#######################################################
# Calculate cumulative proportion of variance explained
#######################################################
# This function computes how many top PCA components are
# necessary to explain a variance of at least 'min.variance'
how.many.pcs.for.variance <- function (min.variance)
{
   results <- matrix(NA, 2, 10)
   for (i in 0:9) {
      for (pc.index in 1:256) {
         cumul <- sum(pc.digits[[i+1]]$sdev[1:pc.index]^2) /
            sum(pc.digits[[i+1]]$sdev^2)
         if (cumul >= min.variance) {
            results[,i+1] <- c(i, pc.index)
            break
         }
      }
   }
   
   for (i in 0:9) {
      print(paste("The first", results[2,i+1],
                  "principal components of the digit",
                  i, "explain", cumul, "% of the total variance."))
   }
   results
}

# Call the above function
(req.pcs <- how.many.pcs.for.variance(0.95))

# Produce LaTeX table describing this result
row.names(req.pcs) <- c("Digit", "# of Components")
xres <- xtable(req.pcs)
print(xres, include.colnames = FALSE)

####################################################
# Examine the PCA components one by one and
# investigate what abstract 'features' they measure
####################################################
num_pcs <- 256
pc <- array(dim = c(num_pcs, 256, 10),
            dimnames = list(c(1:num_pcs),1:256,c(0:9)))
for (j in 1:num_pcs) {
   for (i in 0:9) {
      pc[j,,i+1] <- pc.digits[[i+1]]$rotation[,j]
   }
}

# Source: Shirley Mills STAT5703W
display.mean.pc <- function(pca_comp, digits) {
   mean <- apply(digits, 2, mean)
   for (i in 1:15) {
      image(matrix(mean+(i-8)*pca_comp, 16,16)[,16:1],
            xaxt = "n", yaxt = "n", col = gray((255:0)/255))
   }
}

# Source: Shirley Mills STAT5703W
display.pcs <- function (pcnum) {
   if (dev.cur() == 1) {
      x11(width = 7, height = 5)
   }
   oldpar <- par(mar = c(0, 0, 0, 0))
   layout(matrix(1:150, 10, 15, byrow = TRUE))
   for (i in 0:9) {
      display.mean.pc(pc[pcnum,,i+1], d.digits[[i+1]])
   }
   par(oldpar)
}

prep.out("figures/top-pcs-digit-0.pdf", height=3)
plot.images(t(pc.digits[[0+1]]$rotation), 40)
dev.off()

prep.out("figures/pc-1.pdf", width=7, height=5)
display.pcs(1)
dev.off()

prep.out("figures/pc-2.pdf", width=7, height=5)
display.pcs(2)
dev.off()

#####################################
# Reconstruction of digits with PCA
#####################################
d.digits.pc <- {}
for (i in 0:9) {
   d.digits.pc[[i+1]] <- d.digits[[i+1]]%*%pc.digits[[i+1]]$rotation
}

num.cases <- unlist(lapply(d.digits, dim))[seq(1,20,2)]
num.features <- unlist(lapply(d.digits, dim))[seq(2,21,2)]
num.table <- xtable(rbind(num.cases, num.features))
row.names(num.table) <- c("Training Examples", "Features")
print(num.table)

# Recreate a digit from some subset of its PCA coefficients
recreate <- function(pc.range, digit) {
   tmp <- matrix(0, num.cases[digit+1], 256)
   tmp <- d.digits.pc[[digit+1]][,pc.range] %*%
      t(pc.digits[[digit+1]]$rotation[,pc.range])
   tmp <- tmp/max(abs(range(tmp))) # Scale the data to lie in [-1, 1]
   tmp
}

# Recreate a digit from some subset of its PCA coefficients
# Also, add center and rescale as necessary
recreate.clean <- function(pc.range, digit) {
   tmp <- matrix(0, num.cases[digit+1], 256)
   tmp <- d.digits.pc[[digit+1]][,pc.range] %*% 
      t(pc.digits[[digit+1]]$rotation[,pc.range])
   #Add the center and rescale back data
   if (!identical(pc.digits[[digit+1]]$scale, FALSE)) {
      tmp <- scale(tmp, center = FALSE , scale=1/pc.digits[[digit+1]]$scale)
   }
   if (!identical(pc.digits[[digit+1]]$center, FALSE)) {
      # For presentation purposes, we introduce 'clean.coeff', which
      # takes care of cleaning 
      clean.coeff <- (256-max(pc.range))/256*-1*pc.digits[[digit+1]]$center
      tmp <- scale(tmp, center = clean.coeff, scale=FALSE)
   }
   tmp <- tmp/max(abs(range(tmp))) # Dcale the data to lie in [-1, 1]
   tmp
}

# Recreate training sets digits using the PCs specified by 'pc.range'
plot.recreate.all <- function(pc.range) {
   if (dev.cur() == 1) {
      x11(width = 8, height = 8/12*10) 
   }
   # Create a plot matrix with 144 subplots - plot in row-wise order
   layout(matrix(1:120, 10, 12, byrow = TRUE))
   # No margin (see ?par)
   oldpar <- par(mar = c(0, 0, 0, 0))
   for (d in 0:9) {
      recreated.digits <- recreate(pc.range, d)
      for (i in 1:12) {
         # xaxt = "n", yaxt = "n" - no axes
         image(matrix(recreated.digits[i,],16,16)[,16:1],
               xaxt = "n", yaxt = "n", col = gray((255:0)/255))
      }
   }
   par(oldpar)
}

# Recreate training sets digits using the PCs specified by 'pc.range'
plot.recreate.gradual <- function(digit) {
   if (dev.cur() == 1) {
      x11(width = 9, height = 8/12*10) 
   }
   # Create a plot matrix with 144 subplots - plot in row-wise order
   layout(matrix(1:130, 10, 13, byrow = TRUE))
   # No margin (see ?par)
   oldpar <- par(mar = c(0, 0, 0, 0))
   plot.new()
   text(0.5, 0.5, labels="Origi-\nnal", cex=2)
   for (i in 1:12) {
      image(matrix(d.digits[[digit+1]][i,],16,16)[,16:1],
            xaxt="n", yaxt="n", col = gray((255:0)/255))
   }
   for (num.pcs in c(1,2,3,5,15,30,75,150,256)) {
      pc.range <- 1:num.pcs
      recreated.digits <- recreate.clean(pc.range, digit)
      plot.new()
      text(0.5, 0.5, paste(num.pcs, "\nPCs"), cex=2)
      for (i in 1:12) {
         # xaxt = "n", yaxt = "n" - no axes
         image(matrix(recreated.digits[i,],16,16)[,16:1], 
               xaxt = "n", yaxt = "n", col = gray((255:0)/255))
      }
   }
   par(oldpar)
}

prep.out("figures/recreate-gradual-digit-0.pdf", width=9, height=7)
plot.recreate.gradual(0)
dev.off()

#######################################
# Perform ICA on the combined datasets
#######################################
# Combine the training datasets for digits 0 to 9 into
# a single matrix called 'combined.d.digits'.
# Because the size of the training sets for the various digits is
# not the same, we only take the first 'num.training.examples' rows 
# for each digit, in order to keep the dataset balanced.

num.training.examples <- global.ica.train.size
combined.d.digits <- c()
for (d in 0:9) {
   combined.d.digits <- rbind(combined.d.digits,
                              d.digits[[d+1]][1:num.training.examples,])
}

n.comp <- 40

set.seed(0) #for reproducibility
w.init <- matrix(rnorm(n.comp*n.comp), n.comp, n.comp)

t1 <- proc.time()
combined.ica.digits <- fastICA(t(combined.d.digits),
                               n.comp,
                               alg.typ = "parallel",
                               fun = "logcosh",
                               alpha = 1,
                               method = "R",
                               row.norm = FALSE,
                               maxit = 200,
                               tol = 0.0001,
                               verbose = FALSE,
                               w.init = w.init)
cat(paste("fastICA (n.comp = ", n.comp, ") elapsed time: ",
          (proc.time()-t1)[["elapsed"]], " seconds\n", sep=""))
# Plot all the ICA components
prep.out("figures/ica-components.pdf", width=8, height=4)
plot.images(t(combined.ica.digits$S), n.comp)
dev.off()

########################################################################
# Reconstruct first 10 examples from training sets for each digit
########################################################################
index.first.twelve <- rep(1:10, 10) + rep(seq(0, 9*num.training.examples,
                                              num.training.examples), each=10)

prep.out("figures/ica-reconstruction.pdf", width=8, height=8)
plot.images(t(combined.ica.digits$S %*%
                 combined.ica.digits$A)[index.first.twelve, ],
            length(index.first.twelve))
dev.off()

########################################################################
# Reconstruction of digit 0 for different number of ICA components
########################################################################
fast.reconstruct.zero <- function(n.comp)
{
   set.seed(1990) # for reproducibility
   w.init <- matrix(rnorm(n.comp*n.comp), n.comp, n.comp)
   t1 <- proc.time()
   ica <- fastICA(t(combined.d.digits),
                  n.comp,
                  alg.typ = "parallel",
                  fun = "logcosh",
                  alpha = 1,
                  method = "R",
                  row.norm = FALSE,
                  maxit = 200,
                  tol = 0.0001,
                  verbose = TRUE,
                  w.init = w.init)
   cat(paste("fastICA (n.comp = ", n.comp, ") elapsed time: ",
             (proc.time()-t1)[["elapsed"]], " seconds\n", sep=""))
   ica
}

n.comp.list <- c(1,2,3,5,15,30,75,150,255)
ica.list <- parallel_lapply(n.comp.list, fast.reconstruct.zero)

# Reconstruct the first 12 digits in the training set for each ICA soln
images.reconstructed.list <- c()
for (ica in ica.list) {
   images.reconstructed.list <- rbind(images.reconstructed.list,
                                      list(t(ica$S %*% ica$A)[1:12, ]))
}

prep.out("figures/ica-reconstruction-gradual-digit-zero.pdf", width=9, height=7)
if (dev.cur() == 1) {
   x11(width = 9, height = 8/12*10) 
}
# Create a plot matrix with 144 subplots - plot in row-wise order
layout(matrix(1:130, 10, 13, byrow = TRUE))
# No margin (see ?par)
oldpar <- par(mar = c(0, 0, 0, 0))
plot.new()
text(0.5, 0.5, labels="Origi-\nnal", cex=2)
# Plot original digit 0
for (i in 1:12) {
   image(matrix(d.digits[[0+1]][i,],16,16)[,16:1],
         xaxt="n", yaxt="n", col = gray((255:0)/255))
}
# Here we use apply to simultaneously loop over the two
# lists 'n.comp.list' and 'img.rows.list'
apply(data.frame(n.comp.list, images.reconstructed.list, length(n.comp.list)),
      1,
      function(pair) {
         n.comp <- pair[[1]]
         img <- pair[[2]]
         
         plot.new()
         text(0.5, 0.5, paste(n.comp, "\nICs"), cex=2)
         for (i in 1:12) {
            # xaxt = "n", yaxt = "n" - no axes
            image(matrix(img[i,],16,16)[,16:1], 
                  xaxt = "n", yaxt = "n", col = gray((255:0)/255))
         }
      })
par(oldpar)
dev.off()


prep.out("figures/ica-recreate-gradual-digit-0.pdf", width=9, height=7)
plot.recreate.gradual(0)
dev.off()

###########################################################
# Helper function for running parallel jobs
###########################################################
run_job <- function(job)
{
   type <- job[[1]]
   n.comp <- job[[2]]
   
   if (type == "pca")
      return(run_classifiers_pca(n.comp))
   else if (type == "ica")
      return(run_classifiers_ica(n.comp))
   stop(simpleError("Invalid job type submitted."))
}

###########################################################
# Run classification tasks
###########################################################
m <- run_classifiers()
# Launch the jobs in parallel

A <- c("pca", "ica")
B <- c(10, 25, 50, 100)
# Build the cartesian product A x B
jobs.list <- dlply(expand.grid(A, B), 1:2, c)
# Run the jobs
m_jobs <- parallel_lapply(jobs.list, run_job)

# Show the results :)
perf.measures <- do.call(rbind, c(list(m), m_jobs))
xres <- xtable(perf.measures[order(unlist(perf.measures[,"accuracy"]),
                                   decreasing=TRUE),], digits=3)
print(xres, include.rownames = FALSE)

# Show results with deltas! :)
M <- perf.measures
num.A <- length(A)
num.B <- length(B)
num.C <- 4 # Number of classifiers used
M <- cbind(M, rep(0, num.C))
for (a in 1:num.A) {
   for (b in 1:num.B) {
      s <- seq(num.C+b+(num.C*num.B)*(a-1),
               num.C+num.C*num.B+(num.C*num.B)*(a-1),
               num.C)
      M[s,3] <- unlist(M[s,2]) - rep(M[[b,2]], num.C)
   }
}
xres <- xtable(M, digits=3)
print(xres, include.rownames = FALSE)


############################################################
#
# QUESTION 2: Clustering of seeds dataset
#
############################################################


############################################################
# Load and format seeds dataset
############################################################
seeds <- read.table("seeds_dataset.txt", header=FALSE,
                    colClasses = c(rep("numeric", 7),
                                   "factor"),
                    col.names=c("area", # area A,
                                "perimeter", # perimeter P,
                                "compactness", # C = 4*pi*A/P^2, 
                                "kernel.length", 
                                "kernel.width",
                                "asymmetric.coefficient",
                                "kernel.groove.length",
                                "class" # {karma, rosa, canadian}
                    ))

seeds.class <- as.numeric(seeds$class)

#########################################################################
# This function generates all permutations of the labels and applies
# it to 'cluster'. It compares it with the actual labels and returns the
# cluster with the labels permutation having the lowest error rate.
#########################################################################
find.best.cluster.labels <- function(cluster, actual, num.labels)
{
   best_acc <- 0
   best_cluster <- c()
   permutations <- permn(num.labels)
   for (p in permutations) {
      new_cluster <- cluster
      c <- 1
      for (x in p) {
         new_cluster[cluster==c] <- x
         c <- c+1
      }
      acc <- sum(new_cluster == actual)
      if (acc > best_acc) {
         best_acc <- acc
         best_cluster <- new_cluster
      }
   }
   best_cluster
}

############################################################
# Run hierarchical clustering algorithms on seeds dataset
############################################################
run_hierarchical_clustering <- function(seeds, seeds.class)
{
   # Hierarchical clustering   
   methods = c("ward.D", "ward.D2", "single", "complete", "average",
               "mcquitty", "median", "centroid")
   classes.clust.avg <- c()
   
   seeds.dist <- dist(seeds[,-8])
   
   results <- c()
   for (method in methods) {
      seeds.hclust <- hclust(seeds.dist, method=method)
      seeds.hclust$labels <- seeds$class
      pdf(paste("figures/seeds-hclust-", method, ".pdf", sep=""),
          width=8, height=6)
      oldpar <- par(mar=c(5,4,1,1))
      # Produce hierarchical clustering graph
      plot(seeds.hclust, 
           main=paste("Hierarchical clustering with method '",
                      method, "'", sep=""))
      # Split into 3 clusters and label with blue rectangles
      in.clust <- rect.hclust(seeds.hclust, k=3, border="blue")
      par(oldpar)
      dev.off()
      
      # Compute the clustering accuracy
      classes.clust <- c()
      classes.clust[unname(in.clust[[1]])] <- 1
      classes.clust[unname(in.clust[[2]])] <- 2
      classes.clust[unname(in.clust[[3]])] <- 3
      
      # Find correct labels permutation
      correct.classes.clust <- find.best.cluster.labels(classes.clust,
                                                        seeds.class, 3)
      
      # Add results to 'results' table
      cm <- confusionMatrix(seeds$class, correct.classes.clust)
      acc <- cm$overall[[1]]
      results <- rbind(results, list(method, acc))
      
      # Save the results from method "average" for later
      if (method == "average")
         classes.clust.avg <- classes.clust
   }
   colnames(results) <- c("Method", "Accuracy")
   print(xtable(results, digits=3))
   
   pdf("figures/seeds-hclust-average-versus.pdf", width=8, height=4)
   oldpar <- par(mfrow=c(1,2), mar=c(5,2.5,2,1))
   plot(area ~ asymmetric.coefficient, seeds, col=classes.clust.avg, pch=17,
        main="Hierachical clustering using 'average'")
   plot(area ~ asymmetric.coefficient, seeds, col=seeds.class, pch=17,
        main="Actual labels")
   par(oldpar)
   dev.off()
}

########################################################
# Run vector quantization (k-means) on seeds dataset
########################################################
run_vector_quantization_clustering <- function(seeds, seeds.class)
{
   # Produce colorized pairs-plot
   pdf("figures/seeds-pairs.pdf", width=8, height=8)
   pairs(seeds[,1:7], col = seeds.class+1, cex.labels=1.0)
   dev.off()
   
   # Compute k-means
   set.seed(1990)
   (seeds.kmeans <- kmeans(seeds[,-8], 3, 20, algorithm="Hart"))
   correct.cluster <- find.best.cluster.labels(seeds.kmeans$cluster,
                                               seeds.class, 3)
   cm <- confusionMatrix(correct.cluster, seeds[,8])
   acc <- cm$overall[[1]]
   
   print(cm$table)
   print(paste("Accuracy:", signif(acc, 4)))
   
   pdf("figures/seeds-kmeans.pdf", width=8, height=4)
   oldpar <- par(mfrow=c(1,2), mar=c(5,2.5,2,1))
   plot(area ~ asymmetric.coefficient, seeds, col=correct.cluster, pch=17,
        main="k-means clustering")
   plot(area ~ asymmetric.coefficient, seeds, col=seeds.class, pch=17,
        main="Actual labels")
   par(oldpar)
   dev.off()
}

run_vector_quantization_clustering(seeds, seeds.class)
run_hierarchical_clustering(seeds, seeds.class)

