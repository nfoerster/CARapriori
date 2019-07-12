import numpy as np

"""
Preprocesses a dataframe to a transaction list

Parameters:
[DataFrame] df = the pandas dataframe to preprocess

Returns:
[list] list of preprocessed transactions
[dict] dict of string to int replacements
[dict] dict of int to string replacements

"""
def preprocess_data(df):
    values = set([np.nan])
    for col in df.columns:
        values = values.union(set(df[col].unique()))
    replacement_dict = {k: v for v, k in enumerate(values)}
    inverse_dict = dict(map(reversed, replacement_dict.items()))
    preprocessed_df = df.replace(replacement_dict)
    transactions = [[element for element in row if element != 0] for row in preprocessed_df.values.tolist()]

    return transactions, replacement_dict, inverse_dict

"""
Postprocess to a dataframe from rules set

Parameters:
[set] rules = the rules set

Returns:
[DataFrame] returns the rules as a dataframe

"""
def postprocess_data(rules, inverse_dict):
    import pandas as pd
    df = pd.DataFrame(rules, columns=['LHS', 'RHS', 'Support', 'Confidence'])
    df['RHS'] = df['RHS'].apply(lambda x: inverse_dict[x])
    df['LHS'] = df['LHS'].apply(lambda x: [inverse_dict[y] for y in list(x)] )

    return df

"""
Splits replacement dict values to ids and classes

Parameters:
[dict] replacement_dict = dict of string to int replacements
[list] classes_list = list of strings for classes

Returns:
[set] the ids for LHS
[set] the classes for RHS

"""
def split_classes_ids(replacement_dict, classes_list):
    classes = set([replacement_dict[i] for i in classes_list])
    ids = set(replacement_dict.values()) - classes
    return ids, classes

class CARapriori:

    def __init__(self, transactions):
        self.transactions = transactions

    """
    Performs the init pass

    Parameters:
    [list] transactions = The transaction list
    [set] ids = list of all occuring ids
    [set] target_ids = list of all class ids
    [float] min_support = Minimum specified support
    [float] min_confidence = Minimum specified confidence

    Returns:
    [dict] condsupCount_pruned = Returns a dict for all pruned candidates for counting occurences in transactions
    [dict] rulesupCount_pruned = Returns a dict for all pruned candidates for counting occurences together with target classes
    [set] target_ids = Set of all class ids
    """
    def init_pass(self, ids, target_ids, min_support, min_confidence):
        
        candidates = self.car_candidate_gen(target_ids, ids)
        condsupCount, rulesupCount = self.init_counters(candidates)
        
        condsupCount, rulesupCount = self.search(self.transactions, candidates, target_ids, condsupCount, rulesupCount)
        
        counters_rc_pruned, rulesupCount_pruned = self.prune(len(self.transactions), condsupCount, rulesupCount, min_support, min_confidence)
        
        return counters_rc_pruned, rulesupCount_pruned

    """
    Used to initate counters to count support and confidence

    Parameters:
    [set] candidate_sets = Returns a set of all expanded test sets

    Returns:
    [dict] condsupCount = Returns an empty dict for all candidates for counting occurences in transactions
    [dict] rulesupCount = Returns an empty dict for all candidates for counting occurences together with target classes

    """
    def init_counters(self, candidate_sets):
        rulesupCount = {}
        condsupCount = {}
        for c in candidate_sets:
            rulesupCount[c] = 0
            condsupCount[c[0]] = 0
            
        return condsupCount, rulesupCount

    """
    Used to generate new testable permutations

    Parameters:
    [list] target_ids = list of all class ids
    [set] f_k1 = set of tuples of items
    [set] c_condition_set = set of already explored items

    Returns:
    [list] c = List of candidates
    """
    def car_candidate_gen(self, target_ids, f_k, c_condition_set = set()):
        c = list()
        for class_ in target_ids:
            for item_ in f_k:
                item_set = c_condition_set.copy()
                if isinstance(item_, tuple):
                    item_set.add(item_[0])
                else:
                    item_set.add(item_)
                item_set = tuple(item_set)
                c.append(tuple([item_set,class_]))
        return c
        
    """
    Used to create new test sets

    Parameters:
    [set] last_pruned = the remaining item ids
    [set] target_ids = Set of all class ids

    Returns:
    [set] candidate_sets = Returns a set of all expanded test sets

    """
    def expand(self, last_pruned, target_ids):
        common_set = set()
        
        for key in last_pruned.keys():
            common_set.add(key[0])
        
        candidate_sets = set()
        for key in last_pruned.keys():
            new_set = common_set.copy()
            new_set.remove(key[0])
            
            candidates = self.car_candidate_gen(target_ids, new_set, set(key[0]))
            candidate_sets = candidate_sets.union(candidates)
        
        return candidate_sets

    """
    Search the transaction list and count occurences.

    Parameters:
    [list] transactions = The transaction list
    [set] candidate_sets = A set of all expanded test sets
    [set] target_ids = Set of all class ids
    [dict] condsupCount = An empty dict for all candidates for counting occurences in transactions
    [dict] rulesupCount = An empty dict for all candidates for counting occurences together with target classes

    Returns:
    [dict] condsupCount = Returns a dict for all candidates for counting occurences in transactions
    [dict] rulesupCount = Returns a dict for all candidates for counting occurences together with target classes

    """
    def search(self, transactions, candidate_sets, target_ids, condsupCount, rulesupCount):
        for t in transactions:
            t_set = set(t)
            classes_in_trans = t_set.intersection(target_ids)
            found_in_transaction = {}
            
            for c in candidate_sets:
                
                items_set = set(c[0])
                items_in_trans = t_set.intersection(items_set)
                
                if items_in_trans == items_set:
                    t_item_set = tuple(items_set)
                    if t_item_set not in found_in_transaction:
                        condsupCount[t_item_set] += 1
                        found_in_transaction[t_item_set] = True
                    
                    if c[1] in classes_in_trans:
                        rulesupCount[tuple(c)] += 1
                        
        return condsupCount, rulesupCount

    """
    Prunes the results based on the given counters and thresholds

    Parameters:
    [int] transactions_length = The transaction list length
    [dict] condsupCount = A dict for all candidates for counting occurences in transactions
    [dict] rulesupCount = A dict for all candidates for counting occurences together with target classes
    [float] min_support = Minimum specified support
    [float] min_confidence = Minimum specified confidence

    Returns:
    [dict] condsupCount_pruned = Returns a dict for all pruned candidates for counting occurences in transactions
    [dict] rulesupCount_pruned = Returns a dict for all pruned candidates for counting occurences together with target classes

    """
    def prune(self, transactions_length, condsupCount, rulesupCount, min_support, min_confidence):
        condsupCount_pruned = dict()
        rulesupCount_pruned = dict()
        
        for key, val in condsupCount.items():
            if val > 0:
                support = round(val/transactions_length, 3)
                if support >= min_support:
                    condsupCount_pruned[key] = support
            
        for key, val in rulesupCount.items():
            if val > 0 and key[0] in condsupCount_pruned:
                confidence = round(val/condsupCount[key[0]], 3)
                if confidence >= min_confidence:
                    rulesupCount_pruned[key] = confidence
        
        return condsupCount_pruned, rulesupCount_pruned


    """
    Add the rules to the set

    Parameters:
    [set] rules = New generated rules
    [dict] condsupCount_pruned = A dict for all pruned candidates for counting occurences in transactions
    [dict] rulesupCount_pruned = A dict for all pruned candidates for counting occurences together with target classes

    Returns:
    [bool] Returns True if new rules are added
    """
    def add_rules(self, rules, counters_rc_pruned, rulesupCount_pruned):
    
        rules_before = len(rules)
        for key, val in rulesupCount_pruned.items():
            rules.add(tuple([key[0],key[1],counters_rc_pruned[key[0]],val]))   
        rules_after = len(rules)
        
        #return True if new rules added
        return rules_after > rules_before 
        
    """
    Main function

    Parameters:
    [set] ids = list of all occuring ids
    [set] target_ids = list of all class ids
    [float] min_support = Minimum specified support
    [float] min_confidence = Minimum specified confidence
    [int] max_length = Maximum rule length to search for

    Returns:
    [set] Returns the data mined rules
    """
    def car_apriori(self, ids, target_ids, min_support=0.15, min_confidence=0.7, max_length=2):
        rules = set()
        
        #inital pass
        counters_rc_pruned, rulesupCount_pruned = self.init_pass(ids, target_ids, min_support, min_confidence)
        
        #try to add new rules
        rules_added = self.add_rules(rules, counters_rc_pruned, rulesupCount_pruned)

        if rules_added:
            for iteration in range(max_length):

                #expand new candidates
                candidate_sets = self.expand(rulesupCount_pruned, target_ids)
                #init counters
                counters_rc, rulesupCount = self.init_counters(candidate_sets)
                #search for test sets
                counters_rc, rulesupCount = self.search(self.transactions, candidate_sets, target_ids, counters_rc, rulesupCount)
                #prune
                counters_rc_pruned, rulesupCount_pruned = self.prune(len(self.transactions), counters_rc, rulesupCount, min_support, min_confidence)
                #add
                rules_added = self.add_rules(rules, counters_rc_pruned, rulesupCount_pruned)

                if not rules_added:
                #early stopping
                    break


        return rules
    
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import time
    import cProfile
    transactions = pd.DataFrame({'STUDENT':     ['STUDENT',  'STUDENT',  np.nan,     np.nan,      np.nan,      np.nan,      np.nan], 
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

    
    transactions, replacement_dict, inverse_dict = preprocess_data(transactions)
    print(transactions)

    print(replacement_dict)

    started = int(round(time.time() * 1000))

    cararpriori = CARapriori(transactions)
    ids, classes = split_classes_ids(replacement_dict, ['EDUCATION','SPORT'])

    rules = cararpriori.car_apriori(ids, classes, 0.15, 0.66, 3)
    df = postprocess_data(rules, inverse_dict)
    time_needed = (int(round(time.time() * 1000)) - started)

    print('CARapriori finished, time needed:{} ms'.format(time_needed))
    print(df)