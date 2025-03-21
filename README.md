# Information Retrieval System

![Information Retrieval System](https://img.shields.io/badge/IR-System-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)

A streamlined web-based Information Retrieval System built with Streamlit that supports Boolean and Proximity search queries over document collections.

## 📚 Overview

This Information Retrieval System allows users to search through indexed document collections using two powerful search methods:

1. **Boolean Search**: Find documents matching logical combinations (AND, OR, NOT) of search terms
2. **Proximity Search**: Find documents where specified terms appear within a defined distance of each other

The system uses inverted indices and positional indices to enable efficient search operations and fast retrieval of matching documents.

## 🔍 Features

- **Boolean Search Engine**
  - Support for AND, OR, NOT operators
  - Complex query parsing with nested expressions
  - Efficient document retrieval

- **Proximity Search Engine**
  - Find terms that appear within a specified word distance
  - Simple syntax for proximity queries
  - Returns documents matching proximity constraints

- **User-Friendly Interface**
  - Clean and intuitive UI built with Streamlit
  - Visual feedback for search results
  - Example queries to help users get started
  - Response time metrics for performance analysis

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/information-retrieval-system.git
   cd information-retrieval-system
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
- Additional dependencies listed in `requirements.txt`

## 💻 Usage

### Boolean Search

Enter queries using Boolean operators (AND, OR, NOT) in the Boolean Search tab:

```
machine AND learning
neural OR network
(deep AND learning) OR ai
algorithm NOT genetic
```

### Proximity Search

Enter proximity queries in the format "term1 term2 /N" where N is the maximum word distance:

```
neural network /2
machine learning /5
data mining /3
```

## 🏗️ Project Structure

```
information-retrieval-system/
├── main.py                    # Main Streamlit application
├── models.py                  # Search logic and query processing
├── inverted_index_updt.json   # Inverted index for Boolean search
├── positional_index_updt.json # Positional index for Proximity search
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## 🔧 Technical Details

- **Inverted Index**: Maps terms to document IDs for Boolean search
- **Positional Index**: Maps terms to document IDs with word positions for Proximity search
- **Query Validation**: Ensures proper syntax for both search types
- **Performance Metrics**: Tracks and displays search execution time

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/information-retrieval-system/issues).

## 📝 License

This project is [MIT](LICENSE) licensed.

## 👨‍💻 Author

- **Mirza Abdul Wasay**

---

Made with ❤️ for Information Retrieval
