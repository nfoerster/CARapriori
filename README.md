# Class association rule mining (CARapriori)

Based on pseudo code from:
http://facweb.cs.depaul.edu/mobasher/classes/ect584/Lectures/Liu-Ch2-4.pdf

# How to use python CAR apriori?

## Import
```python
import carapriori as cp
```

## Example


```python
# Example dataframe of transactions
df = pd.DataFrame({         'STUDENT':     ['STUDENT',  'STUDENT',  np.nan,     np.nan,      np.nan,      np.nan,      np.nan], 
                            'TEACH':       ['TEACH',    np.nan,    'TEACH',    np.nan,      np.nan,      np.nan,      np.nan],
                            'SCHOOL':      ['SCHOOL',   'SCHOOL',   'SCHOOL',   np.nan,      np.nan,      np.nan,      np.nan],
                            'CITY':        [np.nan,     np.nan,     'CITY',     np.nan,      np.nan,      np.nan,      'CITY'],
                            'GAME':        [np.nan,     np.nan,     'GAME',     np.nan,      np.nan,      'GAME',      'GAME'],
                            'BASEBALL':    [np.nan,     np.nan,     np.nan,     'BASEBALL',  np.nan,      'BASEBALL',  np.nan],
                            'BASKETBALL':  [np.nan,     np.nan,     np.nan,     'BASKETBALL','BASKETBALL',np.nan,      'BASKETBALL'],
                            'SPECTATOR':   [np.nan,     np.nan,     np.nan,     np.nan,      'SPECTATOR', np.nan,      np.nan],
                            'PLAYER':      [np.nan,     np.nan,     np.nan,     np.nan,      'PLAYER',    np.nan,      np.nan],
                            'COACH':       [np.nan,     np.nan,     np.nan,     np.nan,      np.nan,      'COACH',     np.nan],
                            'TEAM':        [np.nan,     np.nan,     np.nan,     np.nan,      np.nan,      'TEAM',      'TEAM'],
                            'EDUCATION':   ['EDUCATION','EDUCATION','EDUCATION',np.nan,      np.nan,      np.nan,      np.nan],
                            'SPORT':       [np.nan,     np.nan,     np.nan,     'SPORT',     'SPORT',     'SPORT',     'SPORT']})

# Preprocess the dataframe to transactions, string/int replacement dict and int/string inverse dict
transactions, replacement_dict, inverse_dict = cp.preprocess_data(df)

# Create the CARapriori object with the transaction lists of integers
cararpriori = cp.CARapriori(transactions)

# Split the labels to classes and ids
ids, classes = cp.split_classes_ids(replacement_dict, ['EDUCATION','SPORT'])

# Run the algorithm
rules = cararpriori.run(ids, classes, 0.15, 0.66, 3)

# Post process the data to a readable dataframe
final = cp.postprocess_data(rules, inverse_dict)
```

```
                  LHS        RHS  Support  Confidence
0        [TEAM, GAME]      SPORT    0.286       1.000
1             [TEACH]  EDUCATION    0.286       1.000
2              [GAME]      SPORT    0.429       0.667
3   [SCHOOL, STUDENT]  EDUCATION    0.286       1.000
4            [SCHOOL]  EDUCATION    0.429       1.000
5     [TEACH, SCHOOL]  EDUCATION    0.286       1.000
6              [TEAM]      SPORT    0.286       1.000
7           [STUDENT]  EDUCATION    0.286       1.000
8          [BASEBALL]      SPORT    0.286       1.000
9   [STUDENT, SCHOOL]  EDUCATION    0.286       1.000
10       [BASKETBALL]      SPORT    0.429       1.000
```

preprocess_data translates a dataframe to a set of transactions and a string/int dict/inverse_dict. With the transactions at hand you can create an CARapriori object. Next you can define the right hand labels by calling split_classes_ids. With the ids and classes the next step is calling the run function. Additionally you can define the minimum support, minimum confidence and the maximum rule length. After getting the rules calling the postprocess_data function to get a readable dataframe back.
