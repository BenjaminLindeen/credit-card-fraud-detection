# Load libraries
library(caret)
library(ggplot2)
library(smotefamily)
library(rpart)
library(rpart.plot)
library(xgboost)
library(ggcorrplot)
library(pROC)

# Load dataset
dataset <- read.csv("C:/Users/Benjamin/development/school/ie5533/credit-card-fraud-detection/datatset/card_transdata.csv")

# EDA
summary(dataset)
head(dataset)
dim(dataset)
ggplot(dataset, aes(x = distance_from_home)) + geom_histogram(binwidth = 10, fill = "blue", color = "black")
ggplot(dataset, aes(x = as.factor(fraud))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Class Distribution in Dataset", x = "Fraud", y = "Count")

# Data Manipulation
dataset <- na.omit(dataset)

ggplot(dataset, aes(x = distance_from_home)) + geom_histogram(binwidth = 10, fill = "blue", color = "black")
ggplot(dataset, aes(x = as.factor(fraud))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Class Distribution in Dataset", x = "Fraud", y = "Count")

# Split Data
set.seed(123)
partition <- createDataPartition(dataset$fraud, p = 0.8, list = FALSE)
training <- dataset[partition,]
testing <- dataset[-partition,]

# Define Features and Labels for SMOTE
features <- training[, -ncol(training)]  # Exclude the target variable
labels <- training$fraud                # Target variable

# Applying SMOTE
dup_size <- 944.43  # Adjust as needed
smote_data <- SMOTE(features, labels, K = 5, dup_size = dup_size)

# Combine the synthetic samples with the original training data
synthetic_samples2 <- smote_data$data
synthetic_samples2 <- synthetic_samples2[, !names(synthetic_samples2) %in% "class"]
synthetic_samples2$fraud <- rep(1, nrow(synthetic_samples2))  # Assign the minority class label

# Add the target variable to the training set for the features
training_features2 <- training[, -ncol(training)]
training_features2$fraud <- training$fraud


# Combine datasets
training_balanced1 <- rbind(training_features2, synthetic_samples2)

# Check the class distribution
table(training_balanced1$fraud)

# Logistic Regression Model
training_balanced$fraud <- as.numeric(as.character(training_balanced$fraud))
log_model <- glm(fraud ~ ., data = training_balanced, family = binomial())
predictions <- predict(log_model, testing, type = "response")
predictions <- ifelse(predictions > 0.5, 1, 0)
confusionMatrix(factor(predictions), factor(testing$fraud))

# Decision Tree Model
tree_model <- rpart(fraud ~ ., data = training_balanced, method = "class")
predictions <- predict(tree_model, testing, type = "class")
predictions_factor <- factor(predictions, levels = c(0, 1))
testing_fraud_factor <- factor(testing$fraud, levels = c(0, 1))
confusionMatrix(predictions_factor, testing_fraud_factor)
rpart.plot(tree_model, type = 3, box.palette = "RdBu", shadow.col = "gray", branch = 1, extra = 102, under = TRUE, cex = 0.6, tweak = 1.2)

# Artificial Neural Network
nn_model <- neuralnet(fraud ~ ., data = training_balanced)
nn_predictions <- compute(nn_model, testing[, !names(testing) %in% "fraud"])
nn_predictions <- ifelse(nn_predictions$net.result > 0.5, 1, 0)
confusionMatrix(factor(nn_predictions), factor(testing$fraud))

# Gradient Boosting Model
train_data_xgb <- xgb.DMatrix(data = as.matrix(training_balanced[, -ncol(training_balanced)]), label = training_balanced$fraud)
valid_data_xgb <- xgb.DMatrix(data = as.matrix(testing[, -ncol(testing)]), label = testing$fraud)
params <- list(booster = "gbtree", objective = "binary:logistic", eta = 0.1, max_depth = 6, lambda = 1, alpha = 0)
watchlist <- list(train = train_data_xgb, eval = valid_data_xgb)
xgb_model <- xgb.train(params, train_data_xgb, nrounds = 100, watchlist, early_stopping_rounds = 10)
xgb_predictions <- predict(xgb_model, valid_data_xgb, ntreelimit = xgb_model$best_ntreelimit)
xgb_predictions <- ifelse(xgb_predictions > 0.5, 1, 0)
confusionMatrix(factor(xgb_predictions), factor(testing$fraud))

# Additional Metrics
table(testing$fraud)
precision <- posPredValue(factor(xgb_predictions), factor(testing$fraud), positive = "1")
recall <- sensitivity(factor(xgb_predictions), factor(testing$fraud), positive = "1")
f1_score <- (2 * precision * recall) / (precision + recall)
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1_score))