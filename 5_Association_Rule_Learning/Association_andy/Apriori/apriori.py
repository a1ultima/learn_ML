# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Generating correct input format
transactions = [[str(dataset.values[i,j]) for i in range(0,7501)] for j in range(0,20)]

# Running the Apriori algorithm
from apyori import apriori

results = apriori(transactions, min_support = 0.003, min_lift = 4, min_confidence = 0.2, min_length = 2)

## Intuition:
# - Support(M) = #transactions inc. M / #Total transactions, i.e. M's popularity
# - Confidence(M->N) = #Transactions with rule M->N / #transactions inc. (M)
# - Lift(M->N) = Condifence(M->N) / support(M), i.e. how many times more M->N is observed vs. random, >1 the better

# - we decide Support(M) min to be 0.003 since 3*7/7501 is an item that is bought >3 times per day (21 times per week)
# - we decide min confidence using trial-and-error, we start w/ depreciating value until we have a workable #rules output
# - support >4 means a rule that is 4 or more times more relevant than random (1)

results = list(results) # click on the relevant fields to find the embedded stats for each rule