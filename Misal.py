import streamlit as st
import pandas as pd
from datetime import datetime
import time
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import snowflake.connector
from typing import List, Dict, Tuple, Optional
import re
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ddl_analyzer.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
FEEDBACK_FILE = 'user_feedback.csv'
SENSITIVITY_OPTIONS = [
    "Sensitive PII",
    "Non-sensitive PII",
    "Confidential Information",
    "Licensed Data"
]

# Initialize session state for DDL analyzer
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_schema = None
    st.session_state.current_object_type = None
    st.session_state.current_object = None
    st.session_state.analysis_df = None
    st.session_state.editing_enabled = False
    st.session_state.edited_cells = {}
    st.session_state.analysis_complete = False
    st.session_state.csv_processed = False
    st.session_state.csv_file = None

# Snowflake Connection Cache
snowflake_conn = None

def get_llm_response(prompt: str) -> str:
    """Get response from LLM model"""
    try:
        # Implementation depends on your LLM setup
        # This is a placeholder - replace with your actual LLM implementation
        response = "Placeholder LLM response"
        return response
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        raise

def get_snowflake_connection():
    """Get cached Snowflake connection or create new one"""
    global snowflake_conn
    try:
        if snowflake_conn is None:
            env = "dev"
            property_values = fetch_properties(env)
            snowflake_conn = snowflake.connector.connect(
                user=property_values['snowflake.dbUsername'],
                password=property_values['snowflake.dbPassword'],
                account=property_values['snowflake.account'],
                client_session_keep_alive=True,
                client_prefetch_threads=4,
                network_timeout=30
            )
            logger.info("New Snowflake connection established")
        return snowflake_conn
    except Exception as e:
        logger.error(f"Snowflake connection error: {str(e)}")
        st.error(f"Connection error: {str(e)}")
        return None

def close_snowflake_connection():
    """Close Snowflake connection if exists"""
    global snowflake_conn
    if snowflake_conn:
        try:
            snowflake_conn.close()
            snowflake_conn = None
            logger.info("Snowflake connection closed")
        except Exception as e:
            logger.error(f"Error closing Snowflake connection: {str(e)}")

def execute_snowflake_query(query: str, params: tuple = None) -> List:
    """Execute Snowflake query with connection management and error handling"""
    conn = get_snowflake_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        results = cur.fetchall()
        cur.close()
        return results
    except Exception as e:
        logger.error(f"Query execution error: {str(e)}\nQuery: {query}")
        st.error(f"Query execution error: {str(e)}")
        return []

def get_schema_list() -> List[str]:
    """Get list of schemas"""
    try:
        conn = get_snowflake_connection()
        if not conn:
            return []
        
        cur = conn.cursor()
        cur.execute("SHOW SCHEMAS")
        schemas = [row[1] for row in cur.fetchall()]
        cur.close()
        return schemas
    except Exception as e:
        logger.error(f"Error getting schema list: {str(e)}")
        return []

def get_schema_objects(schema: str) -> Dict[str, List[str]]:
    """Get tables and views in schema"""
    try:
        conn = get_snowflake_connection()
        if not conn:
            return {"tables": [], "views": []}
        
        cur = conn.cursor()
        
        # Get tables
        cur.execute(f"SHOW TABLES IN SCHEMA {schema}")
        tables = [row[1] for row in cur.fetchall()]
        
        # Get views
        cur.execute(f"SHOW VIEWS IN SCHEMA {schema}")
        views = [row[1] for row in cur.fetchall()]
        
        cur.close()
        return {"tables": tables, "views": views}
    except Exception as e:
        logger.error(f"Error getting schema objects: {str(e)}")
        return {"tables": [], "views": []}

def get_databases() -> List[str]:
    """Get list of databases with caching"""
    results = execute_snowflake_query("SHOW DATABASES")
    return [row[1] for row in results] if results else []

def get_schemas(database: str) -> List[str]:
    """Get schemas from database with caching"""
    results = execute_snowflake_query(f"SHOW SCHEMAS IN DATABASE {database}")
    return [row[1] for row in results] if results else []

def get_objects(database: str, schema: str, object_type: str) -> List[str]:
    """Get database objects with caching"""
    results = execute_snowflake_query(f"SHOW {object_type}S IN SCHEMA {database}.{schema}")
    return [row[1] for row in results] if results else []

@st.cache_resource
def get_model_and_tokenizer():
    """Loads and caches the model and tokenizer globally"""
    logger.info("Loading model and tokenizer")
    MODEL_ID = "/data/ntracedevpkg/dev/scripts/nhancebot/flant5_sensitivity/AutoModelForSequenceClassification/flant5"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer

def create_classification_prompt(column_name: str, explanation: str) -> str:
    """Create a prompt with reasoning for classification."""
    return (
        f"Classify this data attribute into one of these categories:\n"
        f"- Sensitive PII: user data that if made public can harm user through fraud or theft\n"
        f"- Non-sensitive PII: user data that can be safely made public without harm\n"
        f"- Non-person data: internal company data not related to personal information\n\n"
        f"Attribute Name: {column_name}\n"
        f"Description: {explanation}\n"
        f"Consider the privacy impact and potential for misuse. Classify this as:"
    )

def classify_sensitivity(texts_to_classify: List[str]) -> List[str]:
    """Function to classify a list of texts using the model in batch"""
    logger.info(f"Classifying {len(texts_to_classify)} columns for sensitivity")
    model, tokenizer = get_model_and_tokenizer()
    
    inputs = tokenizer(
        texts_to_classify,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidences, predicted_classes = torch.max(probs, dim=1)
    
    predicted_classes = predicted_classes.cpu().numpy()
    
    id2label = {0: "Sensitive PII", 1: "Non-sensitive PII", 2: "Non-person data"}
    predicted_labels = [id2label[class_id] for class_id in predicted_classes]
    
    return predicted_labels

def get_ddl_and_samples(database: str, schema: str, object_name: str, object_type: str) -> Tuple[str, Dict]:
    """Get DDL and sample values for object"""
    ddl_result = execute_snowflake_query(
        f"SELECT GET_DDL('{object_type}', '{database}.{schema}.{object_name}')"
    )
    ddl = ddl_result[0][0] if ddl_result else ""

    # Extract create statement from view DDL if needed
    if object_type == 'VIEW':
        create_match = re.search(r'create\s+(?:or\s+replace\s+)?view.*?as\s+(.+?)(?:;|\Z)', 
                               ddl, re.IGNORECASE | re.DOTALL)
        if create_match:
            ddl = create_match.group(1).strip()

    # Get sample values for columns
    samples = {}
    try:
        sample_query = f"SELECT * FROM {database}.{schema}.{object_name} LIMIT 5"
        results = execute_snowflake_query(sample_query)
        if results:
            columns = [desc[0] for desc in cur.description]
            for i, col in enumerate(columns):
                samples[col] = [row[i] for row in results]
    except Exception as e:
        logger.warning(f"Could not fetch samples: {str(e)}")

    return ddl, samples

def load_existing_feedback(schema: str, table: str) -> Optional[pd.DataFrame]:
    """Load existing feedback from CSV file"""
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        filtered = df[(df['schema'] == schema) & (df['table'] == table)]
        if not filtered.empty:
            return pd.DataFrame({
                'Column Name': filtered['column_name'],
                'Explanation': filtered['explanation'],
                'Data Sensitivity': filtered['sensitivity']
            })
    return None

def save_feedback(schema: str, table: str, feedback_df: pd.DataFrame):
    """Save or update feedback in CSV file"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_records = []
    
    for _, row in feedback_df.iterrows():
        new_records.append({
            'schema': schema,
            'table': table,
            'column_name': row['Column Name'],
            'explanation': row['Explanation'],
            'sensitivity': row['Data Sensitivity'],
            'timestamp': now
        })
    
    new_df = pd.DataFrame(new_records)
    
    if os.path.exists(FEEDBACK_FILE):
        existing_df = pd.read_csv(FEEDBACK_FILE)
        existing_df = existing_df[~((existing_df['schema'] == schema) & 
                                  (existing_df['table'] == table))]
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df
    
    updated_df.to_csv(FEEDBACK_FILE, index=False)

[Previous CSS and display functions remain unchanged...]

def handle_csv_upload():
    """Handle CSV file upload and analysis"""
    if 'csv_processed' not in st.session_state:
        st.session_state.csv_processed = False
        
    if 'csv_file' not in st.session_state:
        st.session_state.csv_file = None
        
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None and uploaded_file != st.session_state.csv_file:
        st.session_state.csv_file = uploaded_file
        st.session_state.csv_processed = False
        st.session_state.analysis_complete = False
        st.session_state.analysis_df = None
    
    if uploaded_file is not None and st.button("🔍 Analyze Structure"):
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'attribute_name' not in df.columns:
                st.error("CSV must contain 'attribute_name' column")
                return
                
            attributes = [str(attr).strip() for attr in df['attribute_name'].tolist()]
            attributes = [attr for attr in attributes if attr]
            
            all_results = []
            BATCH_SIZE = 30
            
            for i in range(0, len(attributes), BATCH_SIZE):
                batch = attributes[i:i + BATCH_SIZE]
                
                batch_prompt = f"""For each column name below, provide a clear, concise explanation of what it represents in a database context.
                Format the response as a JSON with column names as keys and explanations as values.
                
                Column names:
                {', '.join(batch)}"""
                
                with st.spinner(f'Processing columns {i+1}-{min(i+BATCH_SIZE, len(attributes))} of {len(attributes)}...'):
                    descriptions_json = get_llm_response(batch_prompt)
                    batch_analysis = json.loads(descriptions_json)
                    
                    batch_classification_prompts = [
                        create_classification_prompt(col, explanation) 
                        for col, explanation in batch_analysis.items()
                    ]
                    
                    batch_predictions = classify_sensitivity(batch_classification_prompts)
                    
                    batch_transformed = [
                        "Confidential Information" if pred == "Non-person data" else pred 
                        for pred in batch_predictions
                    ]
                    
                    batch_results = [
                        {
                            "Column Name": col,
                            "Explanation": explanation,
                            "Data Sensitivity": sensitivity
                        }
                        for (col, explanation), sensitivity in zip(batch_analysis.items(), batch_transformed)
                    ]
                    all_results.extend(batch_results)
            
            st.session_state.analysis_df = pd.DataFrame(all_results)
            st.session_state.analysis_complete = True
            st.session_state.csv_processed = True
            
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            logger.error(f"CSV processing error: {str(e)}")

[Rest of the main() function and if __name__ == "__main__" block remain the same...]
