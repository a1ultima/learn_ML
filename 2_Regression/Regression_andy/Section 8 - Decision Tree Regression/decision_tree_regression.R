# Regression Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the Regression Model to the dataset
# Create your regressor here
#install.packages("rpart")
library("rpart")

regressor = rpart(formula = dataset$Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1) # default is 20,.. 
                  #..but we only have 10 observations, so it will yeild a..
                  #..split that averages a value over all observations if..
                  #..minsplit wasn't set to 1 (allows splits to contain 1 obs)
)

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Regression Model results
# install.packages('ggplot2')
library(ggplot2)

x_grid = seq(from=min(dataset$Level), to=max(dataset$Level), by=0.01)

ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary), colour="red")+
  geom_line(aes(x=x_grid,y=predict(regressor, newdata = data.frame(Level = x_grid))), colour="blue")+
  ggtitle("Decision Tree Regression")+
  xlab("Level")+
  ylab("Salary")



