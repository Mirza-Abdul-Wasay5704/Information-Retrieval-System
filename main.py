import streamlit as st
import json
import time
from models import validateQuery, boolean_search, ProximityQuery_search


st.set_page_config(page_title="Information Retrieval System", page_icon="🌐")


# loading indexes at Startup 
@st.cache_data
def load_indexes():
    with open("inverted_index_updt.json", "r") as f:
        inverted_index = {k: v for line in f for k, v in json.loads(line).items()}

    with open("positional_index_updt.json", "r") as f:
        positional_index = {k: v for line in f for k, v in json.loads(line).items()}

    return inverted_index, positional_index


inverted_index, positional_index = load_indexes()




# Streamlit App
st.title("📚 Information Retrieval System")

# differ tabs for Boolean and Proximity Search
tab1, tab2 = st.tabs(["🔍 Boolean Search", "📏 Proximity Search"])



# Boolean Search Tab
with tab1:
    st.subheader("🔍 Boolean Search")
    query = st.text_input("Enter Boolean Query (e.g., deep AND learning):")

    if st.button("Search", key="boolean_search"):
        if query:
            start_time = time.time()  # Start Timer
            query_data = validateQuery(query)

            if isinstance(query_data, dict) and "valid" in query_data:
                if query_data["valid"]:
                    terms, ops = query_data["terms"], query_data["operators"]
                    results = boolean_search(terms, ops, inverted_index)
                    search_time = round(time.time() - start_time, 4)  # End Timer

                    st.success(f"✅ {len(results)} documents found in {search_time:.4f} seconds!")


                    #st.write(f"### 📝 Results Found: {len(results)} (in {search_time} seconds)")
                    
                    if results:
                        st.write("#### 📄 Matching Documents:")
                        for doc_id in results:
                            st.markdown(f"- 📄 **Document ID:** `{doc_id}`")
                    else:
                        st.warning("No matching documents found.")
                else:
                    st.error("Invalid Boolean Query! Please check the syntax.")
            else:
                st.error("Query validation failed. Please enter a correct Boolean query.")




# Proximity Search Tab
with tab2:
    st.subheader("📏 Proximity Search")
    prox_query = st.text_input("Enter Proximity Query (e.g., neural information /2):")

    if st.button("Search", key="proximity_search"):
        if prox_query:
            start_time = time.time()  # Start Timer
            result_data = ProximityQuery_search(prox_query, positional_index)
            search_time = round(time.time() - start_time, 4)  # End Timer

            if isinstance(result_data, dict) and "Error" in result_data:
                st.error(result_data["Error"])
            elif isinstance(result_data, dict) and "terms" in result_data:
                terms = result_data["terms"]
                dist = result_data["distance"]
                results = result_data["result"]
                st.success(f"✅ {len(results)} documents found in {search_time:.4f} seconds!")
                st.write(f"### 📊 Terms: `{terms[0]}` & `{terms[1]}`")
                st.write(f"### 🔢 Maximum Distance: `{dist}`")
                
                #st.write(f"### 📝 Results Found: {len(results)} (in {search_time} seconds)")
                
                
                if results:
                    st.write("#### 📄 Matching Documents:")
                    for doc_id in results:
                        st.markdown(f"- 📄 **Document ID:** `{doc_id}`")
                else:
                    st.warning("No matching documents found.")
            else:
                st.error("Proximity query validation failed. Please check the input format.")



# Footer
st.markdown("""
---
<p style="text-align: center;">Developed By Mirza Abdul Wasay</p>
""", unsafe_allow_html=True)