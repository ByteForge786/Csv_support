def is_allowed_user(username):
    """Check if user is allowed to see CSV upload option"""
    return 'gowdas' in username.lower()

def handle_csv_upload():
    """Handle CSV file upload and analysis"""
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Reset session state for CSV analysis
        st.session_state.analysis_complete = False
        st.session_state.analysis_df = None
        st.session_state.edited_cells = {}
        
        try:
            df = pd.read_csv(uploaded_file)
            
            # Verify attribute_name column exists
            if 'attribute_name' not in df.columns:
                st.error("CSV must contain 'attribute_name' column")
                return
                
            # Create analysis format similar to DDL pipeline
            analysis = {}
            for attr in df['attribute_name']:
                # Generate prompt for analysis
                prompt = f"Provide a clear, concise explanation of what the column '{attr}' represents in a database context."
                explanation = get_llm_response(prompt)  # Using existing LLM function
                analysis[attr] = explanation
            
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
            
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            logger.error(f"CSV processing error: {str(e)}")

def main():
    st.set_page_config(page_title="DDL Analyzer", page_icon="🔍", layout="wide")
    apply_custom_css()
    
    st.title("🔍 DDL Analyzer")
    st.markdown("Analyze your database objects structure and get intelligent insights.")

    # Sidebar selections
    with st.sidebar:
        st.header("Analysis Type")
        
        # Only show CSV option to allowed users
        username = os.getenv('USER', '')  # Get current username
        analysis_type = "Database Object"
        
        if is_allowed_user(username):
            analysis_type = st.radio(
                "Select Analysis Type",
                ["Database Object", "CSV Upload"]
            )
        
        if analysis_type == "Database Object":
            # Original database object selection code
            st.header("Object Selection")
            schemas = get_schema_list()
            selected_schema = st.selectbox("1. Select Schema", schemas)
            
            # Rest of your existing sidebar code...
            # [Keep all existing database object selection logic]

    # Main content area
    if analysis_type == "CSV Upload":
        handle_csv_upload()
    else:
        # Your existing database object analysis code
        if all([selected_schema, object_type, selected_object]):
            try:
                # Rest of your existing main function code...
                # [Keep all existing database analysis logic]
            except Exception as e:
                st.error("Error analyzing DDL. Please try again.")
                logger.error(f"Analysis error: {str(e)}")

    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.analysis_df is not None:
        st.subheader("📊 Analysis Results")
        display_editable_table()

if __name__ == "__main__":
    main()
