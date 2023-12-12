# Load libraries
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(xgboost)
library(ggcorrplot)
library(pROC)
library(dplyr)

# Load dataset
dataset <- read.csv("C:/Users/Benjamin/development/credit-card-fraud-detection/datatset/card_transdata.csv")

# Data Manipulation
dataset <- na.omit(dataset)

# EDA
summary(dataset)
head(dataset)
dim(dataset)
plot_list <- list()
num_vars <- names(dataset)[sapply(dataset, is.numeric)]  # Identify numeric variables

for (var in num_vars) {
    p <- ggplot(dataset, aes_string(x = var)) +
         geom_histogram(binwidth = 10, fill = "blue", color = "black") +
         labs(title = paste("Distribution of", var), x = var, y = "Count") +
         theme_minimal()
    print(p)  # Explicitly print the plot
    Sys.sleep(1)  # Add a short delay to ensure plot is rendered
}




# Split Data
set.seed(123)
partition <- createDataPartition(dataset$fraud, p = 0.8, list = FALSE)
training <- dataset[partition, ]
testing <- dataset[-partition, ]

# Calculate class weights
class_weights <- ifelse(training$fraud == 1, (1 / table(training$fraud)[2]), (1 / table(training$fraud)[1]))

# Logistic Regression Model with class weights
log_model <- glm(fraud ~ ., data = training, family = binomial(), weights = class_weights)
predictions <- predict(log_model, testing, type = "response")
predictions <- ifelse(predictions > 0.5, 1, 0)
confusionMatrix(factor(predictions), factor(testing$fraud))

# Decision Tree Model
tree_model <- rpart(fraud ~ ., data = training, method = "class", weights = class_weights)
predictions <- predict(tree_model, testing, type = "class")
predictions_factor <- factor(predictions, levels = c(0, 1))
testing_fraud_factor <- factor(testing$fraud, levels = c(0, 1))
confusionMatrix(predictions_factor, testing_fraud_factor)
rpart.plot(tree_model, type = 3, box.palette = "RdBu", shadow.col = "gray", branch = 1, extra = 102, under = TRUE, cex = 0.6, tweak = 1.2)

# Gradient Boosting Model
train_data_xgb <- xgb.DMatrix(data = as.matrix(training[, -ncol(training)]), label = training$fraud)
valid_data_xgb <- xgb.DMatrix(data = as.matrix(testing[, -ncol(testing)]), label = testing$fraud)
scale_pos_weight <- sum(training$fraud == 0) / sum(training$fraud == 1)
params <- list(booster = "gbtree", objective = "binary:logistic", eta = 0.1, max_depth = 6, scale_pos_weight = scale_pos_weight)
watchlist <- list(train = train_data_xgb, eval = valid_data_xgb)
xgb_model <- xgb.train(params, train_data_xgb, nrounds = 100, watchlist, early_stopping_rounds = 10)
xgb_predictions <- predict(xgb_model, valid_data_xgb, ntreelimit = xgb_model$best_ntreelimit)
xgb_predictions <- ifelse(xgb_predictions > 0.5, 1, 0)
confusionMatrix(factor(xgb_predictions), factor(testing$fraud))

# Additional Metrics
precision <- posPredValue(factor(xgb_predictions), factor(testing$fraud), positive = "1")
recall <- sensitivity(factor(xgb_predictions), factor(testing$fraud), positive = "1")
f1_score <- (2 * precision * recall) / (precision + recall)
roc_obj <- roc(testing$fraud, as.numeric(xgb_predictions))
auc <- auc(roc_obj)
plot(roc_obj, main = "ROC Curve")

# Print metrics
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1_score))
print(paste("AUC:", auc))

# Feature Importance for XGBoost
importance_matrix <- xgb.importance(feature_names = colnames(train_data_xgb), model = xgb_model)
xgb.plot.importance(importance_matrix)
