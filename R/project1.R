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
dataset <- read.csv("C:/Users/Benjamin/development/credit-card-fraud-detection/datatset/card_transdata.csv")

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

# Split Data
set.seed(123)
partition <- createDataPartition(dataset$fraud, p = 0.8, list = FALSE)
training <- dataset[partition,]
testing <- dataset[-partition,]

# Define Features and Labels for SMOTE
features <- training[, -ncol(training)]  # Exclude the target variable
labels <- training$fraud                # Target variable

# Count of non-fraudulent transactions
count_non_fraud <- nrow(training[training$fraud == 0,])
print(count_non_fraud)

# Count of fraudulent transactions
count_fraud <- nrow(training[training$fraud == 1,])
print(count_fraud)

# Calculate dup_size for a 1:1 ratio
dup_size <- count_non_fraud/ count_fraud
print(dup_size)
# Apply SMOTE with the calculated dup_size
smote_data <- SMOTE(features, labels, K = 5, dup_size = 1)


# Combine the synthetic samples with the original training data
synthetic_samples1 <- smote_data$data
synthetic_samples1 <- synthetic_samples1[, !names(synthetic_samples1) %in% "class"]
synthetic_samples1$fraud <- rep(1, nrow(synthetic_samples1))  # Assign the minority class label

# Add the target variable to the training set for the features
training_features1 <- training[, -ncol(training)]
training_features1$fraud <- training$fraud

# Combine datasets
training_balanced1 <- rbind(training_features1, synthetic_samples1)

# Check the class distribution
table(training_balanced1$fraud)
# Proceed with the rest of the model training and evaluation...
# (Logistic Regression, Decision Tree, ANN, Gradient Boosting, etc.)

# Define Features and Labels for SMOTE