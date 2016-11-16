######################################################################
# QUESTION 2: Clustering of seeds dataset
######################################################################
library(stats)
library(deldir)
library(combinat)
library(xtable)
library(caret)

work.dir <- "/proj/stat5703w/ass2"
setwd(work.dir)

############################################################
# Load and format seeds dataset
############################################################
seeds <- read.table("seeds_dataset.txt", header=FALSE,
                    colClasses = c(rep("numeric", 7),
                                   "factor"),
                    col.names=c("area", # area A,
                                "perimeter", # perimeter P,
                                "compactness", # compactness C = 4*pi*A/P^2, 
                                "kernel.length", # length of kernel, 
                                "kernel.width", # width of kernel, 
                                "asymmetric.coefficient", # asymmetry coefficient,
                                "kernel.groove.length", # length of kernel groove.
                                "class" # class of weath: {karma, rosa, canadian}
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
# Plot elementary graphs about seed
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
           main=paste("Hierarchical clustering with method '",  method, "'", sep=""))
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
      correct.classes.clust <- find.best.cluster.labels(classes.clust, seeds.class, 3)
      
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
   correct.cluster <- find.best.cluster.labels(seeds.kmeans$cluster, seeds.class, 3)
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
