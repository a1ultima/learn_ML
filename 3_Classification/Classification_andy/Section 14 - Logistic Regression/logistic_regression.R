setwd("~/Desktop/udemy_ML/3_Classification/Classification_andy/Section 14 - Logistic Regression")

#
#
# Logistic Regression
#
#


#
# Preprocessing
#

# Importing the dataset
dataset = read.csv("./Social_Network_Ads.csv")
dataset = dataset[,3:5] # Purchased ~ Salary + Age

# Split into training vs. test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set  = subset(dataset, split == TRUE)
test_set      = subset(dataset, split == FALSE)

# Scale features
training_set[,1:2] = scale(training_set[,1:2])
test_set[,1:2]     = scale(test_set[,1:2])

# Logistic Regression Classifier
classifier = glm( formula = Purchased ~ .,
                  family = "binomial",
                  data = training_set)
#
# Predictions
#

# Probabilities 
prob_pred = predict( classifier, 
                     type = "response",
                     newdata = test_set[-3]) # remove the field with actual outcomes
# Outcomes (True/False)
y_pred = ifelse( prob_pred > 0.5, 1, 0)

# Confusion matrix
cm = table(test_set[, 3], y_pred)

#
# Visualisation
#
set = training_set

Xgrid_age    = seq( min(set[, 1]) -1, max(set[, 1]) +1, by = 0.01 )
Xgrid_salary = seq( min(set[, 2]) -1, max(set[, 2]) +1, by = 0.01 )

Xmesh_age_vs_salary = expand.grid(Xgrid_age, Xgrid_salary)

colnames(Xmesh_age_vs_salary) = c("Age","EstimatedSalary")
       
prob_pred_mesh <- predict( classifier,
                         type = "response",
                         newdata = Xmesh_age_vs_salary)

y_pred_mesh <- ifelse( prob_pred_mesh > 0.5, 1, 0)

plot( set[-3],
      main = "Logistic Regression",
      xlim = range(Xgrid_age),
      ylim = range(Xgrid_salary))

# Not entirely sure what countour contributes: it doesn't seem to make a difference to the plot
contour( Xgrid_age, Xgrid_salary,
         matrix(as.numeric(y_pred_mesh), length(Xgrid_age), length(Xgrid_salary)),
         add = TRUE)

points( Xmesh_age_vs_salary, pch = ".", 
        col = ifelse(y_pred_mesh == 1, "springgreen3", "tomato"))
points( set, pch = 21, 
        bg = ifelse(set[, 3] == 1, "green4", "red3"))

