def main():
    st.set_page_config(page_title="DDL Analyzer", page_icon="🔍", layout="wide")
    apply_custom_css()
    
    st.title("🔍 DDL Analyzer")
    st.markdown("Analyze your database objects structure and get intelligent insights.")

    # Define allowed users
    ALLOWED_USERS = ['gowdas']  # Add more users as needed
    current_user = os.getenv('USER', '').lower()

    # Sidebar selections
    with st.sidebar:
        st.header("Analysis Type")
        
        # Show CSV upload option only for allowed users
        if any(user in current_user for user in ALLOWED_USERS):
            analysis_type = st.radio("Select Analysis Type", ["Database Objects", "Upload CSV"])
        else:
            analysis_type = "Database Objects"

        if analysis_type == "Database Objects":
            st.header("Object Selection")
            # Original database object selection code
            schemas = get_schema_list()
            selected_schema = st.selectbox("1. Select Schema", schemas)
            
            if selected_schema != st.session_state.current_schema:
                st.session_state.current_schema = selected_schema
                st.session_state.analysis_complete = False
                st.session_state.analysis_df = None
                st.session_state.edited_cells = {}
            
            object_type = st.radio("2. Select Object Type", ["TABLE", "VIEW"])
            
            if object_type != st.session_state.current_object_type:
                st.session_state.current_object_type = object_type
                st.session_state.analysis_complete = False
                st.session_state.analysis_df = None
                st.session_state.edited_cells = {}
            
            schema_objects = get_schema_objects(selected_schema)
            object_list = schema_objects["tables"] if object_type == "TABLE" else schema_objects["views"]
            selected_object = st.selectbox(f"3. Select {object_type}", object_list)

            if selected_object != st.session_state.current_object:
                st.session_state.current_object = selected_object
                st.session_state.analysis_complete = False
                st.session_state.analysis_df = None
                st.session_state.edited_cells = {}

    # Main content area
    if analysis_type == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file with attribute_name column", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'attribute_name' not in df.columns:
                st.error("CSV must contain 'attribute_name' column")
                return
                
            if not st.session_state.analysis_complete:
                if st.button("🔍 Analyze Structure"):
                    with st.spinner("Analyzing structure and predicting sensitivity..."):
                        # Generate prompts for analysis
                        prompts = [
                            create_classification_prompt(attr_name, "")  # Empty explanation for now
                            for attr_name in df['attribute_name']
                        ]
                        
                        # Get sensitivity predictions
                        sensitivity_predictions = classify_sensitivity(prompts)
                        
                        # Transform predictions
                        transformed_predictions = [
                            "Confidential Information" if pred == "Non-person data" else pred 
                            for pred in sensitivity_predictions
                        ]
                        
                        # Create results DataFrame
                        st.session_state.analysis_df = pd.DataFrame({
                            "Column Name": df['attribute_name'],
                            "Explanation": "",  # Empty explanations initially
                            "Data Sensitivity": transformed_predictions
                        })
                        
                        st.session_state.analysis_complete = True
            
            # Display results using existing function
            if st.session_state.analysis_complete:
                display_editable_table()

    elif all([selected_schema, object_type, selected_object]):
        # Original database analysis code
        try:
            ddl, samples = get_ddl_and_samples(selected_schema, selected_object, object_type)
            # ... rest of the original code ...
        except Exception as e:
            st.error("Error analyzing DDL. Please try again.")
            logger.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
