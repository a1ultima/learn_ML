
setwd("~/Desktop/udemy_ML/2_Regression/Regression_andy/Section 4 - Simplae Linear Regression")

dataset <- read.csv("Salary_Data.csv")

library(caTools)

set.seed(123)

split <- sample.split(dataset$Salary, SplitRatio = 2/3)

training_set <- subset( dataset, split == TRUE)
test_set     <- subset( dataset, split == FALSE)

# FItting simple linear regression onto training set

regressor <- lm(formula = Salary ~ YearsExperience, 
                data = training_set)
summary(regressor)

# Predict values from training set regressin model

y_pred <- predict( regressor, newdata = test_set)

summary(y_pred)

#
# Plot results
#

#install.packages("ggplot2")
library(ggplot2)

# training set

ggplot() +
  geom_point( aes( y = training_set$Salary, 
                   x = training_set$YearsExperience),
                   colour = "red") +
  geom_line( aes(  y = predict(regressor, newdata = training_set), 
                   x = training_set$YearsExperience),
                   colour = "blue") + 
  ggtitle("Salary vs Experience (training set)") +
  xlab("Experience") +
  ylab("Salary")

# test set

ggplot() + 
  geom_point ( aes( y = test_set$Salary, 
                    x = test_set$YearsExperience), 
                    colour = "red") +
  geom_line(   aes( y = predict(regressor, newdata = training_set), 
                    x = training_set$YearsExperience),
                    colour = "blue") +
  ggtitle("Salary vs Experience (test set)") +
  xlab("Experience") + 
  ylab("Salary")


