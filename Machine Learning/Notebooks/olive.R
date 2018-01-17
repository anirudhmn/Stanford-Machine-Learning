# PCA olive oil example, Alex Ioannidis 
# note, you will need to install the two libraries below (and dependencies) first

library(scatterplot3d)
library(Hmisc)
setwd("/Users/boron/Desktop/example")


olive <- read.table("olive.txt", header = FALSE ) 

# PCA

pca=princomp(olive[,3:10], cor = TRUE)
summary(pca)
plot(pca$sd , xlab= "Principal Component Number", ylab="Standard Deviation" )
pcoms <- predict(pca)

# Plot

plot(pcoms[,1:2], type = "n")
text(pcoms[,1:2], labels=olive$V2, col=olive$V1, cex=1)
scatterplot3d(pcoms[,1:3], xlab="1st pc", ylab="2nd pc", zlab="3rd pc", angle = 120 , type = "n")
text(pcoms[,1:2], labels=olive$V2, col=olive$V1, cex=1)

# Gap-like Statistic

pca=princomp(olive[,3:10], cor = TRUE)
var=pca$sdev^2/sum(pca$sdev^2)
var_cum=cumsum(var)
size = dim( olive[,3:10] )

colmax=apply(olive[,3:10],2,max)
colmin=apply(olive[,3:10],2,min)

gap=matrix(nrow=40,ncol=size[2])

for( i in 1:40) {
	pca_unif = princomp(t(t(matrix(runif(size[1]*size[2]),size[1],size[2]))*(colmax-colmin)+colmin), cor = TRUE)
	var_unif=pca_unif$sdev^2/sum(pca_unif$sdev^2)
	var_cum_unif=cumsum(var_unif)
	gap[i,]=var_cum-var_cum_unif
	mean=apply(gap,2,mean)
	sd=apply(gap,2,sd)
}

plot(mean,type="o", xlab="# Components", ylab="Gap Test")
errbar(1:8,mean,mean-sd,mean+sd, xlab="# Components", ylab="Gap Test")
