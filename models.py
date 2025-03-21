from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

ps = PorterStemmer()



def validateQuery(query):
    lower_query = query.lower()
    
    if not lower_query.strip():
        return {"terms": [], "operators": [], "valid": False}  
        #return {"Error" : "Query can not be Empty."}
    
    words = lower_query.strip().split()
    
    if len(words)>5:
        return {"terms": [], "operators": [], "valid": False}  
        #return {"Error": "Query Can Contain 3 Terms and 2 Operators At Most."}
    
    
    valid_operators = {"and","or","not"}

    terms = []
    operators = []
    expecting_term = True # tracking what should come next
    
    for word in words:
        if word in valid_operators:
            
            if word == "not" and expecting_term:
                # NOT is valid at the start or after another operator
                operators.append(word)
                expecting_term = True  # expecting a term after NOT
                
            elif expecting_term:
                return {"Error": "Invalid Operator Placement!"}
            
            else:
                operators.append(word)
                expecting_term = True # now next must be an term
                
        else:
            
            if not expecting_term:
                return {"Error": "Operators Must Separate Terms Correctly."}
            stemmed_word = ps.stem(word)
            terms.append(stemmed_word)
            expecting_term = False  # next must be an operator or end of query
    
    
    if expecting_term and operators:  # query should not end with AND/OR
        return {"terms": [], "operators": [], "valid": False}  
        #return {"Error": "Query can not end with an operator."}
    
    
    if len(terms) == 1 and not operators:
        return {"terms":terms, "operators":None, "valid":True}
    
    return {"terms":terms, "operators":operators, "valid":True}
 

 


def merge_AND(l1,l2):
    # intersection of 2 lists 
    intersected = []
    
    for doc in l1:
        if doc in l2:
            intersected.append(doc)
    return intersected


def merge_OR(l1,l2):
    # unions of 2 lists
    union_set = set()
    
    for doc in l1:
        union_set.add(doc)
    
    for doc in l2:
        union_set.add(doc)
    
    return sorted(union_set) 



def merge_NOT(all_docs,l1):
    # return NOT operation
    complement_docs = []
    
    for doc in all_docs:
        if doc not in l1:
            complement_docs.append(doc)
    return complement_docs






def boolean_search(terms,operator,inverted_index):
    
    # fetch all docs first of all
    all_doc = set()
    
    for docs in inverted_index.values():
        for doc in docs:
            all_doc.add(doc) # adding each  docID to the set
            
    all_doc = sorted(all_doc)
    
    
    # 1) Handle single term quries:
    if len(terms) == 1:
        return inverted_index.get(terms[0],[])  # if term present in invrtd index return IDs list else []
    
    
    
    # geting the posting list for all terms
    l1 = inverted_index.get(terms[0],[])
    l2 = inverted_index.get(terms[1],[])
    
    
    if len(terms) == 3:
        l3 = inverted_index.get(terms[2],[])
    else:
        l3 = []  # when theres no 3rd term
        
        
    

    # processing the first operation
    if operator[0] == "and":
        result = merge_AND(l1,l2)
    elif operator[0] == "or":
        result = merge_OR(l1,l2)
    elif operator[0] == "not":
        result = merge_NOT(all_doc,l2)
    else:
        return [] # invalid operator
    
    
    # if there is second operation
    if len(operator) == 2:
        if operator[1] == 'and':
            result = merge_AND(result, l3)
        elif operator[1] == 'or':
            result = merge_OR(result, l3)
        elif operator[1] == 'not':
            result = merge_NOT(all_doc, result)
        else:
            return []  # Invalid operator
    return result

    
    





def compute_metrics(retrieved, relevant):   # for calculating Precision, Recall, F1-Score
   
    retrieved = set(retrieved)
    relevant = set(relevant)
    
    # True Positives: intersection of retrieved and relavant docIDs
    true_positives = len(retrieved.intersection(relevant))
    
    # Precision: what frction of retrieved docs are relevant
    if len(retrieved) > 0:
        precision = true_positives / len(retrieved)
    else:
        precision = 0
    
    
    
    # Recall: what fraction of all relevant docs were retrieved
    if len(relevant) > 0:
        recall = true_positives / len(relevant)
    else:
        recall = 0
    
    
    
    # F1-Score: Harmonic mean of precision and recall
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
        
    return precision, recall, f1








ps = PorterStemmer()



def ProximityQuery_search(query,positional_index):
    
    
    lower_query = query.lower()
    
    if not lower_query.strip():
        return {"Error" : "Query can not be Empty."}
    
    
    # query : term   term   /   int   (handling all edge cases for spaces and whatever user inputs)
    pattren = r'^(\w+)\s+(\w+)\s*/\s*(\d+)$'
    
    match = re.match(pattren, lower_query)
    
    if not match:
        return {"Error": "Invalid proximity query format. Expected format: 'term1 term2 /k', e.g., 'neural information /2'"}
    

    term1_raw, term2_raw, dist_str = match.groups()
    
    
    # storing the stemmmed words
    term1 = ps.stem(term1_raw)
    term2 = ps.stem(term2_raw)
    
    if dist_str.isdigit():
        dist = int(dist_str)
    else:
        return {"Error": "Invalid proximity value. Please provide a numeric value after '/'."}

    
    result_docs = set()
    
    
    # chk if both terms persent in positional index
    if term1 not in positional_index or term2 not in positional_index:
        return {"terms": [term1, term2], "distance": dist, "result": []}

    
    # if they both appear in index
    docs_term1 = set(positional_index[term1].keys())  # get docIDs of term1
    docs_term2 = set(positional_index[term2].keys())  # get docIDs of term2
    
    # get the common docID by intersecting
    common_docs = docs_term1.intersection(docs_term2)
    
    
    # now for each common doc chk for positions using 2-pointer technique
    for doc in common_docs:
        pos1 = positional_index[term1][doc]
        pos2 = positional_index[term2][doc]
        
        i = 0
        j = 0
        while i<len(pos1) and j<len(pos2):
            
            gap = abs(pos1[i] - pos2[j])
            if gap <= dist:
                result_docs.add(doc)
                break   # stop chk futher positions of this doc
            
            if pos1[i] < pos2[j]:
                i += 1
            else:
                j += 1
            
    result_docs = sorted(result_docs)
    return {"terms": [term1_raw, term2_raw], "distance": dist, "result": result_docs}

        
        

