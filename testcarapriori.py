import carapriori as cp
import pandas as pd
import numpy as np

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
transactions, replacement_dict, inverse_dict = cp.preprocess_data(df)
cararpriori = cp.CARapriori(transactions)
ids, classes = cp.split_classes_ids(replacement_dict, ['EDUCATION','SPORT'])

rules = cararpriori.run(ids, classes, 0.15, 0.66, 3)
final = cp.postprocess_data(rules, inverse_dict)
print(final)