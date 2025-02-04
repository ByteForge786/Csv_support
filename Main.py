def is_allowed_user(username):
    """Check if user is allowed to see CSV upload option"""
    return 'gowdas' in username.lower()

def handle_csv_upload():
    """Handle CSV file upload and analysis"""
    # Only trigger file upload if session isn't already processing a CSV
    if 'csv_processed' not in st.session_state:
        st.session_state.csv_processed = False
        
    if 'csv_file' not in st.session_state:
        st.session_state.csv_file = None
        
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    # Handle new file upload
    if uploaded_file is not None and uploaded_file != st.session_state.csv_file:
        st.session_state.csv_file = uploaded_file
        st.session_state.csv_processed = False
        st.session_state.analysis_complete = False
        st.session_state.analysis_df = None
        
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'attribute_name' not in df.columns:
                st.error("CSV must contain 'attribute_name' column")
                return
                
            # Batch process descriptions
            attributes = df['attribute_name'].tolist()
            batch_prompt = f"""For each column name below, provide a clear, concise explanation of what it represents in a database context.
            Format the response as a JSON with column names as keys and explanations as values.
            
            Column names:
            {', '.join(attributes)}"""
            
            # Get batch descriptions
            descriptions_json = get_llm_response(batch_prompt)
            analysis = eval(descriptions_json)
            
            # Create classification prompts
            classification_prompts = [
                create_classification_prompt(col, explanation) 
                for col, explanation in analysis.items()
            ]
            
            # Get sensitivity predictions using existing pipeline
            sensitivity_predictions = classify_sensitivity(classification_prompts)
            
            # Transform predictions
            transformed_predictions = [
                "Confidential Information" if pred == "Non-person data" else pred 
                for pred in sensitivity_predictions
            ]
            
            # Prepare results data
            results_data = [
                {
                    "Column Name": col,
                    "Explanation": explanation,
                    "Data Sensitivity": sensitivity
                }
                for (col, explanation), sensitivity in zip(analysis.items(), transformed_predictions)
            ]
            
            st.session_state.analysis_df = pd.DataFrame(results_data)
            st.session_state.analysis_complete = True
            st.session_state.csv_processed = True
            
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            logger.error(f"CSV processing error: {str(e)}")

def main():
    st.set_page_config(page_title="DDL Analyzer", page_icon="🔍", layout="wide")
    apply_custom_css()
    
    st.title("🔍 DDL Analyzer")
    st.markdown("Analyze your database objects structure and get intelligent insights.")

    # Initialize session state for analysis type if not exists
    if 'analysis_type' not in st.session_state:
        st.session_state.analysis_type = "Database Object"

    # Sidebar selections
    with st.sidebar:
        st.header("Analysis Type")
        
        # Only show CSV option to allowed users
        username = os.getenv('USER', '')
        
        if is_allowed_user(username):
            analysis_type = st.radio(
                "Select Analysis Type",
                ["Database Object", "CSV Upload"],
                key='analysis_type_radio'
            )
            
            # Handle analysis type change
            if analysis_type != st.session_state.analysis_type:
                st.session_state.analysis_type = analysis_type
                # Reset states when switching
                if analysis_type == "Database Object":
                    st.session_state.csv_file = None
                    st.session_state.csv_processed = False
                else:
                    st.session_state.current_schema = None
                    st.session_state.current_object = None
                    st.session_state.current_object_type = None
                st.session_state.analysis_complete = False
                st.session_state.analysis_df = None
                st.experimental_rerun()
        
        if st.session_state.analysis_type == "Database Object":
            # Original database object selection code
            st.header("Object Selection")
            schemas = get_schema_list()
            selected_schema = st.selectbox("1. Select Schema", schemas)
            
            if selected_schema != st.session_state.current_schema:
                st.session_state.current_schema = selected_schema
                st.session_state.analysis_complete = False
                st.session_state.analysis_df = None
            
            object_type = st.radio("2. Select Object Type", ["TABLE", "VIEW"])
            if object_type != st.session_state.current_object_type:
                st.session_state.current_object_type = object_type
                st.session_state.analysis_complete = False
                st.session_state.analysis_df = None
            
            schema_objects = get_schema_objects(selected_schema)
            object_list = schema_objects["tables"] if object_type == "TABLE" else schema_objects["views"]
            selected_object = st.selectbox(f"3. Select {object_type}", object_list)
            
            if selected_object != st.session_state.current_object:
                st.session_state.current_object = selected_object
                st.session_state.analysis_complete = False
                st.session_state.analysis_df = None

    # Main content area
    if st.session_state.analysis_type == "CSV Upload":
        handle_csv_upload()
        
        # Display results only if CSV is processed
        if st.session_state.csv_processed and st.session_state.analysis_complete:
            st.subheader("📊 Analysis Results")
            display_editable_table()
    else:
        # Your existing database object analysis code
        if all([st.session_state.current_schema, st.session_state.current_object_type, st.session_state.current_object]):
            try:
                # Rest of your existing main function code for database objects
                ddl, samples = get_ddl_and_samples(st.session_state.current_schema, 
                                                 st.session_state.current_object, 
                                                 st.session_state.current_object_type)
                
                # Display DDL
                st.subheader("📝 DDL Statement")
                with st.expander("View DDL", expanded=True):
                    st.code(ddl, language='sql')
                    if samples:
                        st.subheader("Sample Values")
                        for column, values in samples.items():
                            st.write(f"**{column}**: {', '.join(str(v) for v in values)}")
                
                # Your existing analysis button and logic here...
                
            except Exception as e:
                st.error("Error analyzing DDL. Please try again.")
                logger.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
