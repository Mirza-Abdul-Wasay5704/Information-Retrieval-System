# Information Retrieval System

![Information Retrieval System](https://img.shields.io/badge/IR-System-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)
![NLTK](https://img.shields.io/badge/NLTK-3.8-yellow)

A comprehensive web-based Information Retrieval System built with Streamlit that provides multiple search paradigms: Boolean, Proximity, and Vector Space Model (VSM).

## 📚 Overview

This Information Retrieval System allows users to search through a collection of documents using three powerful search methods:

1. **Boolean Search**: Find documents matching logical combinations (AND, OR, NOT) of search terms
2. **Proximity Search**: Find documents where specified terms appear within a defined distance of each other
3. **Vector Space Model (VSM)**: Find documents similar to a query based on cosine similarity

The system uses inverted indices, positional indices, and TF-IDF vectors to enable efficient search operations and fast retrieval of relevant documents.

## 🔍 Features

- **Boolean Search Engine**
  - Support for AND, OR, NOT operators
  - Complex query parsing and validation
  - Efficient retrieval using inverted index

- **Proximity Search Engine**
  - Find terms that appear within a specified word distance
  - Simple syntax for proximity queries (e.g., "neural network /2")
  - Uses positional index for efficient retrieval

- **Vector Space Model Search**
  - Semantic similarity search based on cosine similarity
  - TF-IDF weighting for terms
  - Customizable similarity threshold (alpha)
  - Ranks documents by similarity score

- **User-Friendly Interface**
  - Clean and intuitive UI built with Streamlit
  - Interactive document viewer
  - Search time metrics for performance analysis
  - Responsive design with tabs for different search methods

## 🖥️ System Architecture

The system is built with a modular architecture:

1. **Preprocessing Pipeline**
   - Document tokenization using NLTK
   - Stemming with Porter Stemmer
   - Stopwords removal
   - Building inverted and positional indices

2. **Search Models**
   - Boolean model using set operations (AND, OR, NOT)
   - Proximity model using positional information
   - Vector Space Model using TF-IDF and cosine similarity

3. **UI Layer**
   - Streamlit-based interactive interface
   - Document viewer with tabs for content and metadata
   - Search result presentation with ranking

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Mirza-Abdul-Wasay5704/Information-Retrieval-System.git
   cd Information-Retrieval-System
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run main.py
   ```

## 📋 Requirements

- Python 3.8+
- Streamlit 1.32.0
- NLTK
- NumPy
- scikit-learn (for vector operations)
- Additional dependencies listed in `requirements.txt`

## 💻 Usage

### Boolean Search

Enter queries using Boolean operators (AND, OR, NOT) in the Boolean Search tab:

```
machine AND learning
neural OR network
deep AND (learning NOT supervised)
algorithm NOT genetic
```

The system will return documents that match the Boolean expression.

### Proximity Search

Enter proximity queries in the format "term1 term2 /N" where N is the maximum word distance:

```
neural network /2
machine learning /5
data mining /3
```

This will return documents where the terms appear within N words of each other.

### Vector Space Model (VSM) Search

Enter any natural language query and adjust the similarity threshold (alpha):

```
machine learning applications in healthcare
neural networks for image classification
```

The system will return documents ranked by their similarity to the query, filtering out documents with similarity below the alpha threshold.

## 🏗️ Project Structure

```
information-retrieval-system/
├── main.py                    # Main Streamlit application
├── models.py                  # Search logic and query processing
├── Abstracts/                 # Corpus of document files
├── inverted_index_updt.json   # Inverted index for Boolean search
├── positional_index_updt.json # Positional index for Proximity search
├── vocab_index.json           # Vocabulary index for VSM
├── idf.json                   # IDF values for VSM
├── tfidf_vectors.json         # Pre-computed TF-IDF vectors
├── Preprocessing & Updated Index Building.ipynb  # Index generation
├── Boolean_Models.ipynb       # Boolean query implementation
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## 🔧 Technical Details

### Preprocessing
- **Tokenization**: Documents are split into tokens using NLTK's word_tokenize
- **Stemming**: Porter Stemmer reduces words to their root form
- **Stopword Removal**: Common words are filtered out
- **Index Creation**: Terms are mapped to documents with additional metadata

### Boolean Search
- **Inverted Index**: Maps terms to document IDs for efficient retrieval
- **Query Validation**: Ensures proper syntax and structure
- **Set Operations**: Implements AND, OR, NOT using merge algorithms

### Proximity Search
- **Positional Index**: Maps terms to document IDs with position information
- **Distance Calculation**: Computes word distances between terms
- **Efficient Retrieval**: Only examines documents containing both terms

### Vector Space Model
- **TF-IDF Vectors**: Documents and queries represented as weighted vectors
- **Cosine Similarity**: Measures angle between query and document vectors
- **Ranking**: Orders results by similarity score
- **Threshold Filtering**: Adjustable similarity threshold (alpha)

### Performance Optimization
- **Pre-computed Indices**: Stores indices for faster startup
- **Efficient Data Structures**: Optimized for memory usage and speed
- **Lazy Loading**: Loads components as needed

## 📊 Evaluation

The system has been evaluated on a corpus of 448 document abstracts, showing excellent performance:
- Boolean queries execute in milliseconds
- Proximity search provides accurate contextual results
- VSM search offers relevant semantic matches with customizable thresholds

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Mirza-Abdul-Wasay5704/Information-Retrieval-System.git/issues).

## 📝 License

This project is [MIT](LICENSE) licensed.

## 👨‍💻 Author

- **Mirza Abdul Wasay**

## 🔍 Future Work

- **Query Expansion**: Enhance queries with synonyms and related terms
- **Spelling Correction**: Implement "Did you mean...?" functionality
- **Advanced Ranking**: Incorporate PageRank-like algorithms
- **Personalization**: User profiles for tailored search results
- **Multilingual Support**: Extend to handle multiple languages

---

Made with ❤️ for Information Retrieval
