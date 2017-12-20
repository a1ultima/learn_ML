# ECLAT algorithm in Python

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Generating correct input format
transactions = [[str(dataset.values[j,i]) for i in range(0,20)] for j in range(0,7501)]

def eclat(transactions_list, output_filename, min_support = 0.002):
    
    """ Implements the eclat algorithm on a list of lists containing transactions 
    in the format of the video. Data output is written to a file specified by 
    output_filename argument.
    
    transactions_list: list of lists, each list representing a transaction, e.g. 
    
        dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
        transactions_list = [[str(dataset.values[j,i]) for i in range(0,20)] for j in range(0,7501)]
    
    output_filename: string to specify output filename, e.g. "eclat_out.tsv"
    
    min_support: skips all combinations of items with supoort < min_support, 
        e.g. 0.002
    
    """
    
    #
    # Per-transaction combination method (22:55)
    #
    
    import time
    
    t_start = time.time()
    
    combos_TO_counts = {}
    
    for transaction in transactions:
        
        goods = list(np.unique(transaction))    
        length = len(goods)
        
        for k in range(2,length+1):
            
            k_combos = list(combinations(goods, k))
            for combo in k_combos:
                
                if set(combo).issubset(transaction):                
                    try:
                        combos_TO_counts[combo] += 1
                    except(KeyError):
                        combos_TO_counts[combo] = 1
    
    t_end = time.time()
    t_duration = t_end - t_start
    
    #
    # Calculate supports for combinations of goods
    #
    combo_support_vec = []
    
    for combo in combos_TO_counts.keys():
        # NOTE: Support(M) = #transactions inc. M / #Total transactions, 
        #   i.e. M's popularity
        support = float(combos_TO_counts[combo])/len(transactions)
        combo_support_vec.append((combo, support))

    #
    # Sort in order of support
    #
    combo_support_vec.sort(key=lambda x: float(x[1]), reverse=True)

    #
    # Create tab-separated output file (skipping sets w/ < min_support)
    #
    # Note: first column is the set of goods, and the second column is the support
    with open("./eclat_out.tsv","w") as fo:
            for combo, support in combo_support_vec:
                if support<min_support:
                    continue
                else:
                    fo.write(", ".join(combo)+"\t"+str(support)+"\n")
    
    print("Completion time (seconds):"+t_duration)
    
    return combo_support_vec
    
    
combos_vs_supports = eclat(transactions, "./eclat.tsv", min_support = 0.002)

## Running the Apriori algorithm
#from apyori import apriori
#
#results = apriori(transactions, min_support = 0.003, min_lift = 4, min_confidence = 0.2, min_length = 2)
#
### Intuition:
## - Support(M) = #transactions inc. M / #Total transactions, i.e. M's popularity
## - Confidence(M->N) = #Transactions with rule M->N / #transactions inc. (M)
## - Lift(M->N) = Condifence(M->N) / support(M), i.e. how many times more M->N is observed vs. random, >1 the better
#
## - we decide Support(M) min to be 0.003 since 3*7/7501 is an item that is bought >3 times per day (21 times per week)
## - we decide min confidence using trial-and-error, we start w/ depreciating value until we have a workable #rules output
## - support >4 means a rule that is 4 or more times more relevant than random (1)
#
#results = list(results) # click on the relevant fields to find the embedded stats for each rule