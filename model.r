#!/usr/bin/env Rscript

# Example classifier on Numerai data using logistic regression classifier.
library(BoomSpikeSlab)
library(data.table)

# Set seed for reproducibility
set.seed(0)

# Load the data from the CSV files
train <- fread("~/Downloads/numerai_datasets 3/numerai_training_data.csv", head=T)
test <- fread("~/Downloads/numerai_datasets 3/numerai_tournament_data.csv", head=T)

print("Training...")

createMap <- function(name, degree) {
  if (degree == 1) return(name)
  paste0("I(", name, "^",degree,")")
}

createMappings <- function(name, range = 1:3) paste0(sapply(range, createMap,name = name), collapse = " + ")

createFunction <- function(field.names) {
  fun <- paste0(createMappings(field.names), collapse = " + ")
  formula(paste0("target ~ ", fun))
}

# This is your model that will learn to predict. Your model is trained on the numerai_training_data
model <- logit.spike(createFunction(names(train[-1])), data=train, niter = 500)

print("Predicting...")
# Your trained model is now used to make predictions on the numerai_tournament_data
predictions <- predict(model, test[,-1, with = F], type="response")
prob <- apply(predictions[,-c(1:200)], 1, mean)
test$probability <- prob
pred <- test[,c("t_id", "probability"), with = F]

print("Writing predictions to predictions.csv")
# Save the predictions out to a CSV file
write.csv(pred, file="~/Downloads/numerai_datasets 3/predictionsPolynomialSpike.csv", quote=F, row.names=F)
# Now you can upload your predictions on numer.ai

