data_dir = '/Users/schau/data/10x/xae_split_50_overlap_200218'
project_dir = '/Users/schau/projects/XAE'
xae_results_dir = file.path(project_dir, 'results/10x/200218')
save_dir = file.path(project_dir, 'analysis/XAE/200218', basename(xae_results_dir))


library(tidyverse)
library(RColorBrewer)
library(ggpubr)
library(gplots)
library(Rtsne)
library(umap)
library(cluster)
library(Rphenograph)
library(scales)
library(SnapATAC)
library(pROC)
library(PRROC)

theme_set(theme_pubr())
myPalette <- colorRampPalette(rev(brewer.pal(11, "Spectral")))
sc <- scale_colour_gradientn(colours = myPalette(100), limits=c(-3.5, 3.5))
sf <- scale_fill_gradientn(colours = myPalette(100), limits=c(-3.5, 3.5))
disf <- scale_fill_brewer(palette="Set1", type='div')
disc <- scale_color_brewer(palette="Set1", type='div')

dir.create(save_dir, recursive=T)

# labels
labels = read.table(file.path(data_dir, 'pbmc_labels.csv'), header=T, sep=',')

# names
pbmc_A_annot = read.table(file.path(data_dir, 'pbmc_A_colannot.csv'), header=T, sep=',')
pbmc_B_annot = read.table(file.path(data_dir, 'pbmc_B_colannot.csv'), header=T, sep=',')

# domain encodings
A_encodings = read.table(file.path(xae_results_dir, 'A_encodings.csv'), header=F, sep=',')
names(A_encodings) = paste0('L', seq(1, ncol(A_encodings)))
A_encodings$Set = 'A'
A_encodings$tsne1 = labels$tsne1
A_encodings$tsne2 = labels$tsne2
A_encodings$Cluster = labels$Cluster

B_encodings = read.table(file.path(xae_results_dir, 'B_encodings.csv'), header=F, sep=',')
names(B_encodings) = paste0('L', seq(1, ncol(B_encodings)))
B_encodings$Set = 'B'
B_encodings$tsne1 = labels$tsne1
B_encodings$tsne2 = labels$tsne2
B_encodings$Cluster = labels$Cluster

encodings.df = rbind(A_encodings, B_encodings)
encodings.df$Cluster = factor(encodings.df$Cluster)

# cross-domain encodings
A2B_encodings = read.table(file.path(xae_results_dir, 'A2B_encodings.csv'), header=F, sep=',')
B2A_encodings = read.table(file.path(xae_results_dir, 'B2A_encodings.csv'), header=F, sep=',')
xencodings.df = rbind(A2B_encodings, B2A_encodings)

# load in column info for domains A and B
A.annot = read.table(file.path(data_dir, 'pbmc_A_colannot.csv'), header=T, sep=',')
A.gate_weights = read.table(file.path(xae_results_dir, 'A_gate_weights.csv'), header=T, sep=',')
A.annot$gate_weight = A.gate_weights$gate_weights
A.annot$domain = 'A'



B.data = read.table(file.path(data_dir, 'pbmc_B.csv'), header=T, sep=',')

B.annot = read.table(file.path(data_dir, 'pbmc_B_colannot.csv'), header=T, sep=',')
B.gate_weights = read.table(file.path(xae_results_dir, 'B_gate_weights.csv'), header=T, sep=',')

B.annot$gate_weight = B.gate_weights$gate_weights

B.annot$domain = 'B'

all_annot = rbind(A.annot, B.annot)

# loss terms
loss.df = read.table(file.path(xae_results_dir, 'xae_loss.csv'), header=T, sep=',')


ggplot(A.annot) +
  geom_point(aes(x=gate_weight^2, y=colvar, col=col_split)) +
  disc +
  ggsave(file.path(save_dir, 'A_annotations.jpg'))


#ggplot(A.annot) +
#  geom_point(aes(x=gate_weight^2, y=log(colmean), col=col_split))


#A.annot$CV = A.annot$colmean / A.annot$colvar

#ggplot(A.annot) +
#  geom_point(aes(x=gate_weight^2, y=log(CV), col=col_split))

