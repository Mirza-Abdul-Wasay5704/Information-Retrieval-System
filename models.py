import nltk
import os
import re
import json
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data properly
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

ps = PorterStemmer()

# Load the pre-saved vocab, IDF, and TF-IDF vectors
def load_data():
    """Load necessary data for all search models"""
    try:
        # Load vocab_index from JSON
        with open('vocab_index.json', 'r') as f:
            vocab_index = json.load(f)

        # Load IDF from JSON
        with open('idf.json', 'r') as f:
            idf = json.load(f)

        # Load TF-IDF vectors from JSON
        with open('tfidf_vectors.json', 'r') as f:
            tfidf_vectors = json.load(f)

        return vocab_index, idf, tfidf_vectors
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return {}, {}, {}









# **********************************************************************************************************
# Query Validation (handles boolean queries)
# **********************************************************************************************************
def validateQuery(query):
    """Validate boolean query syntax"""
    if not query or not query.strip():
        return {"valid": False, "error": "Query cannot be empty."}

    lower_query = query.lower()
    words = lower_query.strip().split()

    if len(words) > 5:
        return {"valid": False, "error": "Query too long. Maximum 5 terms allowed."}

    valid_operators = {"and", "or", "not"}
    terms = []
    operators = []
    expecting_term = True

    for word in words:
        if word in valid_operators:
            if word == "not" and expecting_term:
                operators.append(word)
                expecting_term = True
            elif expecting_term:
                return {"valid": False, "error": "Invalid operator placement!"}
            else:
                operators.append(word)
                expecting_term = True
        else:
            if not expecting_term:
                return {"valid": False, "error": "Operators must separate terms correctly."}
            stemmed_word = ps.stem(word)
            terms.append(stemmed_word)
            expecting_term = False

    if expecting_term and operators:
        return {"valid": False, "error": "Query cannot end with an operator."}

    return {"terms": terms, "operators": operators, "valid": True}














# **********************************************************************************************************
# Boolean Search Logic
# **********************************************************************************************************
def merge_AND(l1, l2):
    """Intersection of two document lists"""
    return [doc for doc in l1 if doc in l2]

def merge_OR(l1, l2):
    """Union of two document lists"""
    return sorted(set(l1).union(l2))

def merge_NOT(all_docs, l1):
    """Documents in all_docs but not in l1"""
    return [doc for doc in all_docs if doc not in l1]
def boolean_search(terms, operators, inverted_index):
    """Execute boolean search query"""
    if not terms:
        return []
        
    
    all_docs = set(doc for docs in inverted_index.values() for doc in docs)
    all_docs = sorted(all_docs)

    # Handle single term query
    if len(terms) == 1:
        # Special case for "NOT term"
        if operators and operators[0] == "not":
            # Return all documents that DON'T contain the term
            excluded_docs = inverted_index.get(terms[0], [])
            return [doc for doc in all_docs if doc not in excluded_docs]
        return inverted_index.get(terms[0], [])

    # Get document lists for terms
    l1 = inverted_index.get(terms[0], [])
    
    # Special handling for NOT as first operator
    if operators and operators[0] == "not":
        l2 = inverted_index.get(terms[1], [])
        result = [doc for doc in all_docs if doc not in l2]
        
        # If there are more terms/operators
        if len(terms) > 2 and len(operators) > 1:
            l3 = inverted_index.get(terms[2], [])
            if operators[1] == "and":
                result = merge_AND(result, l3)
            elif operators[1] == "or":
                result = merge_OR(result, l3)
            
        return result
    
    # Handle normal binary operators (AND, OR)
    if len(terms) > 1:
        l2 = inverted_index.get(terms[1], [])
        
        if not operators:
            return l1
            
        if operators[0] == "and":
            result = merge_AND(l1, l2)
        elif operators[0] == "or":
            result = merge_OR(l1, l2)
        else:
            return []

        # Handle third term if present
        if len(terms) > 2 and len(operators) > 1:
            l3 = inverted_index.get(terms[2], [])
            if operators[1] == "and":
                result = merge_AND(result, l3)
            elif operators[1] == "or":
                result = merge_OR(result, l3)
            elif operators[1] == "not":
                # For operations like "term1 AND term2 NOT term3"
                to_exclude = set(l3)
                result = [doc for doc in result if doc not in to_exclude]

        return result
    
    return []










# **********************************************************************************************************
# Proximity Query Logic
# **********************************************************************************************************
def ProximityQuery_search(query, positional_index):
    """Execute proximity search query"""
    if not query or not query.strip():
        return {"valid": False, "error": "Query cannot be empty."}

    lower_query = query.lower()
    pattern = r'^(\w+)\s+(\w+)\s*/\s*(\d+)$'
    match = re.match(pattern, lower_query)

    if not match:
        return {"valid": False, "error": "Invalid proximity query format. Expected format: 'term1 term2 /k', e.g., 'neural information /2'"}

    term1_raw, term2_raw, dist_str = match.groups()
    term1 = ps.stem(term1_raw)
    term2 = ps.stem(term2_raw)

    if dist_str.isdigit():
        dist = int(dist_str)
    else:
        return {"valid": False, "error": "Invalid proximity value. Please provide a numeric value after '/'."}

    result_docs = set()

    if term1 not in positional_index or term2 not in positional_index:
        return {"valid": True, "terms": [term1_raw, term2_raw], "distance": dist, "result": []}

    docs_term1 = set(positional_index[term1].keys())
    docs_term2 = set(positional_index[term2].keys())
    common_docs = docs_term1.intersection(docs_term2)

    for doc in common_docs:
        pos1 = positional_index[term1][doc]
        pos2 = positional_index[term2][doc]
        i, j = 0, 0
        while i < len(pos1) and j < len(pos2):
            gap = abs(pos1[i] - pos2[j])
            if gap <= dist:
                result_docs.add(doc)
                break
            if pos1[i] < pos2[j]:
                i += 1
            else:
                j += 1

    return {"valid": True, "terms": [term1_raw, term2_raw], "distance": dist, "result": sorted(result_docs)}











# **********************************************************************************************************
# Vector Space Model Logic (Cosine Similarity)
# **********************************************************************************************************
def preprocess_query(query):
    """Preprocess VSM query"""
    if not query or not query.strip():
        return {"valid": False, "error": "Query cannot be empty."}

    query = query.lower()
    # Simple tokenization as fallback if NLTK fails
    try:
        tokens = word_tokenize(query)
    except LookupError:
        # Fallback tokenization
        tokens = re.findall(r'\b\w+\b', query)
    
    terms = [ps.stem(word) for word in tokens if word not in string.punctuation]

    if not terms:
        return {"valid": False, "error": "Query must contain valid terms."}

    return {"valid": True, "terms": terms}

def vectorize_query(query_terms, vocab_index, idf):
    """Convert query to TF-IDF vector"""
    vocab = list(vocab_index.keys())
    query_tf_vector = np.zeros(len(vocab))

    for term in query_terms:
        if term in vocab_index:
            index = vocab_index[term]
            query_tf_vector[index] += 1

    query_tfidf_vector = [query_tf_vector[i] * idf.get(vocab[i], 0) for i in range(len(query_tf_vector))]
    return query_tfidf_vector

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    num = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num / denom if denom != 0 else 0

def get_top_k_similar_docs(query, tfidf_vectors, vocab_index, idf, alpha=0.001, k=10):
    """Get top k similar documents using VSM"""
    result = preprocess_query(query)

    if not result.get("valid", False):
        return {"valid": False, "error": result.get("error", "Invalid query")}

    query_terms = result['terms']
    query_vec = vectorize_query(query_terms, vocab_index, idf)

    scores = [(doc_id, cosine_similarity(query_vec, doc_vec)) for doc_id, doc_vec in tfidf_vectors.items() 
              if cosine_similarity(query_vec, doc_vec) > alpha]
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return {"valid": True, "results": sorted_scores[:k]}










# **********************************************************************************************************
# Reading the Documents
# **********************************************************************************************************

def load_document_content(doc_id):
    try:
        
        with open(f"Abstracts/{doc_id}.txt", "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "valid": True,
            "doc_id": doc_id,
            "content": content
        }
    except FileNotFoundError:
    
        try:
            with open(f"documents/{doc_id}.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                content = data.get("content", "")
            return {
                "valid": True,
                "doc_id": doc_id,
                "content": content
            }
        except FileNotFoundError:
            return {
                "valid": False,
                "error": f"Document with ID {doc_id} not found"
            }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Error loading document {doc_id}: {str(e)}"
        }