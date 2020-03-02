library(data.table)
library(ggplot2)
library(Rtsne)

setwd("~/Documents/numerai_scripts/")
comp <- "datasets/numerai_dataset_200/"

train <- fread(paste0(comp, "numerai_training_data.csv"))
test <- fread(paste0(comp, "numerai_tournament_data.csv"))
used <- fread(".temp/output")
submission <- fread(paste0(comp, "predictions.csv"))

dir.create("analyse_run")


test[,.(target_kazutsugi), era]

plotFeature <- function(feature.col) {
  
  x <- train[,list(.N), list(target = train[,target_kazutsugi],
                             feature = train[,feature.col, with = F][[1]],
                             era = train$era)]
  plot(x$target, x$feature, cex = x$N/sum(x$N) * 10000,
       main = names(train)[feature.col])
}

plotFeature(4)

cov <- cor(as.matrix(train[, c(grep("feature_", names(train)),
                               grep("target_", names(train))), with = F]))

cov.dt <- as.data.table(cov)
cov.dt$lhs <- rownames(cov)



cov.dt <- melt(cov.dt, id.vars = "lhs", variable.name = 'rhs', value.name = 'corr')
cov.dt$target <- sapply(cov.dt$lhs, grepl, pattern = 'target_') | sapply(cov.dt$rhs, grepl, pattern = 'target_')

plot(cov.dt[lhs != rhs]$corr, cex = 0.1 + cov.dt$target,
     col = ifelse(cov.dt$target, "red", "blue"))

plot(cov.dt[(lhs != rhs) & sapply(lhs, grepl, pattern = 'target_')][,sort(corr)], cex = 0.1)

plot(cov.dt[lhs != rhs][lhs %in% gsub("*_2$", "",used[probability > 0.1]$variable)][(target)]$corr, 
     # cex = 0.1 + cov.dt$target,
     col = ifelse(cov.dt$target, "red", "blue"))

plot(cov.dt[sapply(lhs, grepl, pattern = 'target_') & (corr > 0) & (lhs != rhs)][, sort(corr)])

ggplot(cov.dt) + geom_tile()

tsne <- Rtsne(as.matrix(train[, grep("feature_", names(train)), with = F]))
tsne.data <- data.table(tsne$Y)
tsne.data$target <- train$target_kazutsugi
plot(tsne$Y, cex = train$target_kazutsugi)

sapply(train, function(x) length(unique(x)))
