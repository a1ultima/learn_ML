
# WORKING DIR
setwd("~/Desktop/udemy_ML/1_Data_Preprocessing/Data_Preprocessing_andy")

# DATA INPUT
dataset = read.csv("Data.csv")

# MAIN

#
# MISSING DATA
#

# attributes(dataset)   # allows one to see data column header names

dataset$Age <- ifelse(is.na(dataset$Age),
                      ave( dataset$Age, FUN = function(x) mean(x, na.rm=TRUE) ),
                      dataset$Age
                      )

dataset$Salary <- ifelse( is.na(dataset$Salary), 
                          ave( dataset$Salary, FUN = function(x) mean(x, na.rm=TRUE)),
                          dataset$Salary
                          )
#
# encoding categorical data
#

# we encode a single column that specifies country (spain, france, etc.) into "factors"
dataset$Country <- factor(
  dataset$Country,
  levels = c("France","Spain","Germany"),
  labels = c(1,2,3)  # it doesn't matter which numbers to use
)

dataset$Purchased <- factor(
  dataset$Purchased,
  levels = c("Yes", "No"),
  labels = c(1,0)
)
#
# split data into training and testing sets
#

# Load the caTools library for trai/test set splitting
required_lib <- "caTools"
#install.packages("caTools")
if(required_lib %in% rownames(installed.packages()) == FALSE) {install.packages(required_lib)}
library(caTools)

set.seed(123)

split <- sample.split(dataset$Purchased, SplitRatio = 0.8 )

training_set[,2:3]  <- subset( dataset[,2:3], split == TRUE )
test_set[, 2:3]      <- subset( dataset[,2:3], split == FALSE )

#
# Scale features (makes sure no feature dominates another, all are -1-to-1)
#
training_set[, 2:3] <- scale( training_set[, 2:3]) # all rows, col 2 and 3
test_set[, 2:3] <- scale( test_set[, 2:3])
