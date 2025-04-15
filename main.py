import streamlit as st
import json
import time
from models import validateQuery, boolean_search, ProximityQuery_search, get_top_k_similar_docs, load_data, load_document_content

st.set_page_config(page_title="Information Retrieval System", page_icon="🌐")

# Loading indexes at Startup
@st.cache_data
def load_indexes():
    """Load inverted and positional indexes"""
    try:
        with open("inverted_index_updt.json", "r") as f:
            inverted_index = {k: v for line in f for k, v in json.loads(line).items()}

        with open("positional_index_updt.json", "r") as f:
            positional_index = {k: v for line in f for k, v in json.loads(line).items()}

        return inverted_index, positional_index
    except Exception as e:
        st.error(f"Error loading indexes: {e}")
        return {}, {}

# Load all required data on startup
inverted_index, positional_index = load_indexes()
vocab_index, idf, tfidf_vectors = load_data()

# Initialize session state for document viewer
if 'show_doc' not in st.session_state:
    st.session_state.show_doc = False
if 'selected_doc_id' not in st.session_state:
    st.session_state.selected_doc_id = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "boolean"

# Function to show document content
def show_document(doc_id):
    st.session_state.show_doc = True
    st.session_state.selected_doc_id = doc_id

# Function to go back to search results
def back_to_search():
    st.session_state.show_doc = False
    st.session_state.selected_doc_id = None

# Main UI
st.title("📚 Information Retrieval System")

# Show document content if a document is selected
if st.session_state.show_doc and st.session_state.selected_doc_id:
    doc_result = load_document_content(st.session_state.selected_doc_id)
    
    if doc_result["valid"]:
        st.button("← Back to Search Results", on_click=back_to_search)
        st.subheader(f"📄 Document ID: {st.session_state.selected_doc_id}")
        
        # Create tabs for different document views
        doc_tab1, doc_tab2 = st.tabs(["Document Content", "Document Info"])
        
        with doc_tab1:
            st.markdown(doc_result["content"])
        
        with doc_tab2:
            st.json({
                "Document ID": st.session_state.selected_doc_id,
                "Content Length": len(doc_result["content"]),
                "Word Count": len(doc_result["content"].split())
            })
    else:
        st.error(doc_result["error"])
        st.button("← Back to Search Results", on_click=back_to_search)

else:
    # Tabs for Boolean, Proximity, and Vector Space Model (VSM)
    tab1, tab2, tab3 = st.tabs(["🔍 Boolean Search", "📏 Proximity Search", "✨ Vector Space Model (VSM)"])

    # Boolean Search Tab
    with tab1:
        st.session_state.current_tab = "boolean"
        st.subheader("🔍 Boolean Search")
        query = st.text_input("Enter Boolean Query (e.g., deep AND learning):")

        if st.button("Search", key="boolean_search"):
            if query:
                start_time = time.time()  # Start Timer
                query_data = validateQuery(query)

                if query_data.get("valid", False):
                    terms, ops = query_data["terms"], query_data.get("operators", [])
                    results = boolean_search(terms, ops, inverted_index)
                    search_time = round(time.time() - start_time, 4)  # End Timer

                    st.success(f"✅ {len(results)} documents found in {search_time:.4f} seconds!")

                    if results:
                        st.write("#### 📄 Matching Documents:")
                        for doc_id in results:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"📄 **Document ID:** `{doc_id}`")
                            with col2:
                                st.button("View Document", key=f"view_{doc_id}_boolean", 
                                         on_click=show_document, args=(doc_id,))
                    else:
                        st.warning("No matching documents found.")
                else:
                    st.error(f"Invalid Boolean Query! {query_data.get('error', 'Please check the syntax.')}")

    # Proximity Search Tab
    with tab2:
        st.session_state.current_tab = "proximity"
        st.subheader("📏 Proximity Search")
        prox_query = st.text_input("Enter Proximity Query (e.g., neural information /2):")

        if st.button("Search", key="proximity_search"):
            if prox_query:
                start_time = time.time()  # Start Timer
                result_data = ProximityQuery_search(prox_query, positional_index)
                search_time = round(time.time() - start_time, 4)  # End Timer

                if not result_data.get("valid", False):
                    st.error(result_data.get("error", "Invalid proximity query"))
                else:
                    terms = result_data["terms"]
                    dist = result_data["distance"]
                    results = result_data["result"]
                    st.success(f"✅ {len(results)} documents found in {search_time:.4f} seconds!")
                    st.write(f"### 📊 Terms: `{terms[0]}` & `{terms[1]}`")
                    st.write(f"### 🔢 Maximum Distance: `{dist}`")

                    if results:
                        st.write("#### 📄 Matching Documents:")
                        for doc_id in results:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"📄 **Document ID:** `{doc_id}`")
                            with col2:
                                st.button("View Document", key=f"view_{doc_id}_proximity", 
                                         on_click=show_document, args=(doc_id,))
                    else:
                        st.warning("No matching documents found.")

    # Vector Space Model (VSM) Tab
    with tab3:
        st.session_state.current_tab = "vsm"
        st.subheader("🔠 Vector Space Model (VSM) Search")
        vsm_query = st.text_input("Enter your search query:")

        # Define the number of top similar documents to display
        top_k = st.slider("Select number of top results", min_value=1, max_value=20, value=10)

        if st.button("Search", key="vsm_search"):
            if vsm_query:
                start_time = time.time()  # Start Timer
                results = get_top_k_similar_docs(vsm_query, tfidf_vectors, vocab_index, idf, alpha=0.001, k=top_k)
                search_time = round(time.time() - start_time, 4)  # End Timer

                if not results.get("valid", False):
                    st.error(results.get("error", "Invalid query"))
                else:
                    st.success(f"✅ Search completed in {search_time:.4f} seconds!")
                    st.write(f"### Top {top_k} Similar Documents:")
                    
                    if results["results"]:
                        for i, (doc_id, score) in enumerate(results["results"], start=1):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.markdown(f"**{i})** 📄 Document ID: `{doc_id}`")
                            with col2:
                                st.markdown(f"Similarity: `{score:.4f}`")
                            with col3:
                                st.button("View Document", key=f"view_{doc_id}_vsm", 
                                         on_click=show_document, args=(doc_id,))
                    else:
                        st.warning("No matching documents found.")
            else:
                st.warning("Please enter a query to search.")

# Footer
st.markdown("""
---
<p style="text-align: center;">Developed By Mirza Abdul Wasay</p>
""", unsafe_allow_html=True)