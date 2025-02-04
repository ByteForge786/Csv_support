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
    
    # Add analyze button
    if uploaded_file is not None and st.button("🔍 Analyze Structure"):
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'attribute_name' not in df.columns:
                st.error("CSV must contain 'attribute_name' column")
                return
                
            # Convert all attributes to strings and clean them
            attributes = [str(attr).strip() for attr in df['attribute_name'].tolist()]
            attributes = [attr for attr in attributes if attr]
            
            # Initialize results container
            all_results = []
            BATCH_SIZE = 30
            
            # Process in batches of 30
            for i in range(0, len(attributes), BATCH_SIZE):
                batch = attributes[i:i + BATCH_SIZE]
                
                # Create prompt for current batch
                batch_prompt = f"""For each column name below, provide a clear, concise explanation of what it represents in a database context.
                Format the response as a JSON with column names as keys and explanations as values.
                
                Column names:
                {', '.join(batch)}"""
                
                # Get batch descriptions
                with st.spinner(f'Processing columns {i+1}-{min(i+BATCH_SIZE, len(attributes))} of {len(attributes)}...'):
                    descriptions_json = get_llm_response(batch_prompt)
                    batch_analysis = eval(descriptions_json)
                    
                    # Create classification prompts for this batch
                    batch_classification_prompts = [
                        create_classification_prompt(col, explanation) 
                        for col, explanation in batch_analysis.items()
                    ]
                    
                    # Get sensitivity predictions for this batch
                    batch_predictions = classify_sensitivity(batch_classification_prompts)
                    
                    # Transform predictions
                    batch_transformed = [
                        "Confidential Information" if pred == "Non-person data" else pred 
                        for pred in batch_predictions
                    ]
                    
                    # Add batch results
                    batch_results = [
                        {
                            "Column Name": col,
                            "Explanation": explanation,
                            "Data Sensitivity": sensitivity
                        }
                        for (col, explanation), sensitivity in zip(batch_analysis.items(), batch_transformed)
                    ]
                    all_results.extend(batch_results)
            
            # Create final dataframe with all results
            st.session_state.analysis_df = pd.DataFrame(all_results)
            st.session_state.analysis_complete = True
            st.session_state.csv_processed = True
            
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            logger.error(f"CSV processing error: {str(e)}")
            
    # Display results if CSV is processed
    if st.session_state.csv_processed and st.session_state.analysis_complete:
        st.subheader("📊 Analysis Results")
        display_editable_table()
