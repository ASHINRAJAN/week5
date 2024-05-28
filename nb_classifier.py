# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Function to load and process data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    # Ensure the CSV has the correct headers
    if data.columns[0] != 'text' or data.columns[1] != 'label':
        data.columns = ['text', 'label']
    return data

# Function to train and evaluate the model
def train_and_evaluate(data):
    # Print the columns of the DataFrame to debug the issue
    st.write("Columns in the dataset:", data.columns)

    # Ensure the column names are correct
    if 'text' not in data.columns or 'label' not in data.columns:
        st.error("The dataset must contain 'text' and 'label' columns.")
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    
    # Using TfidfVectorizer instead of CountVectorizer
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('nb', MultinomialNB())
    ])

    # Hyperparameter tuning
    parameters = {
        'tfidf__max_df': [0.5, 0.75, 1.0],
        'tfidf__min_df': [1, 2, 3],
        'nb__alpha': [0.1, 0.5, 1.0]
    }
    
    grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    return accuracy, precision, recall

# Streamlit app
st.title('Naïve Bayes Document Classifier')

# Instructions for the CSV file
st.write("""
### Instructions:
- The CSV file should contain two columns: `text` and `label`.
- The `text` column should contain the document text.
- The `label` column should contain the corresponding class labels (e.g., `pos` or `neg`).
""")

# Load initial data
data_file = 'document.csv'
data = load_data(data_file)

# Display data
if st.checkbox('Show raw data'):
    st.write(data)

# Train and evaluate model on initial data
if st.button('Train and Evaluate Model on Initial Data'):
    accuracy, precision, recall = train_and_evaluate(data)
    
    if accuracy is not None and precision is not None and recall is not None:
        st.write(f'**Accuracy:** {accuracy:.2f}')
        st.write(f'**Precision:** {precision:.2f}')
        st.write(f'**Recall:** {recall:.2f}')

# Option to upload a new CSV file
st.write("### Upload a New CSV File for Classification")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    new_data = load_data(uploaded_file)
    
    # Display new data
    if st.checkbox('Show new raw data'):
        st.write(new_data)
    
    # Train and evaluate model on new data
    if st.button('Train and Evaluate Model on New Data'):
        accuracy, precision, recall = train_and_evaluate(new_data)
        
        if accuracy is not None and precision is not None and recall is not None:
            st.write(f'**Accuracy:** {accuracy:.2f}')
            st.write(f'**Precision:** {precision:.2f}')
            st.write(f'**Recall:** {recall:.2f}')
