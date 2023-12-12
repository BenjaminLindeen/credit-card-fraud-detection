# Load libraries
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(xgboost)
library(pROC)
library(dplyr)

# Load dataset
dataset <- read.csv("C:/Users/Benjamin/development/credit-card-fraud-detection/datatset/card_transdata.csv")

# Data Manipulation
dataset <- na.omit(dataset)

# Apply log transformation to skewed variables
small_constant <- 0.0000001
dataset <- dataset %>%
  mutate(distance_from_home_log = log(ifelse(distance_from_home <= 0, small_constant, distance_from_home)),
         distance_from_last_transaction_log = log(ifelse(distance_from_last_transaction <= 0, small_constant, distance_from_last_transaction)),
         ratio_to_median_purchase_price_log = log(ifelse(ratio_to_median_purchase_price <= 0, small_constant, ratio_to_median_purchase_price)))

# Remove original skewed variables
dataset <- dataset %>%
  select(distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price)

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

# Linear Regression Model
lin_model <- lm(fraud ~ ., data = training)
summary(lin_model)  # This will give you the model summary including coefficients for each variable

# Predictions from the linear model
predictions_lin <- predict(lin_model, testing)
# Thresholding at 0.5 to determine class labels, not recommended for actual classification tasks
predicted_classes_lin <- ifelse(predictions_lin > 0.5, 1, 0)

# Evaluate predictions
confusionMatrix(factor(predicted_classes_lin), factor(testing$fraud))

# ... [rest of your code]

# Decision Tree Model using the log-transformed variables
tree_model <- rpart(fraud ~ ., data = training, method = "class", weights = class_weights)
predictions <- predict(tree_model, newdata = testing, type = "class")
predictions_factor <- factor(predictions, levels = c(0, 1))
testing_fraud_factor <- factor(testing$fraud, levels = c(0, 1))
confusionMatrix(predictions_factor, testing_fraud_factor)
rpart.plot(tree_model, type = 3, box.palette = "RdBu", shadow.col = "gray", branch = 1, extra = 102, under = TRUE, cex = 0.6, tweak = 1.2)

# Gradient Boosting Model using the log-transformed variables
train_data_xgb <- xgb.DMatrix(data = as.matrix(training[, !names(training) %in% "fraud"]), label = training$fraud)
valid_data_xgb <- xgb.DMatrix(data = as.matrix(testing[, !names(testing) %in% "fraud"]), label = testing$fraud)
scale_pos_weight <- sum(training$fraud == 0) / sum(training$fraud == 1)
params <- list(booster = "gbtree", objective = "binary:logistic", eta = 0.1, max_depth = 6, scale_pos_weight = scale_pos_weight)
watchlist <- list(train = train_data_xgb, eval = valid_data_xgb)
xgb_model <- xgb.train(params, train_data_xgb, nrounds = 1, watchlist, early_stopping_rounds = 10)
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
