if not st.session_state.analysis_complete:
    if st.button("🔍 Analyze Structure"):
        try:
            # Check for existing analysis first
            existing_analysis = load_existing_feedback(selected_schema, selected_object)
            
            if existing_analysis is not None:
                st.session_state.analysis_df = existing_analysis
            else:
                # Split DDL into batches
                ddl_batches = split_ddl_into_batches(ddl, batch_size=30)
                
                # Process each batch
                all_results = []
                total_batches = len(ddl_batches)
                
                for batch_idx, ddl_batch in enumerate(ddl_batches, 1):
                    with st.spinner(f'Processing batch {batch_idx}/{total_batches}...'):
                        # Process current batch
                        batch_analysis = process_ddl_batch(ddl_batch)
                        
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
                            for (col, explanation), sensitivity in zip(batch_analysis.items(), 
                                                                     batch_transformed)
                        ]
                        all_results.extend(batch_results)
                
                # Create final dataframe
                st.session_state.analysis_df = pd.DataFrame(all_results)
            
            st.session_state.analysis_complete = True
            st.success("✅ Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"Error analyzing DDL: {str(e)}")
            logger.error(f"DDL analysis error: {str(e)}")



def split_ddl_into_batches(ddl: str, batch_size: int = 30) -> list:
    """Split DDL into batches while preserving CREATE statement"""
    # Extract the CREATE statement and columns
    create_match = re.match(r'(CREATE.*?)\((.*)\)', ddl, re.DOTALL | re.IGNORECASE)
    if not create_match:
        raise ValueError("Could not parse DDL statement")
        
    create_stmt = create_match.group(1)
    columns_text = create_match.group(2)
    
    # Split columns and clean
    columns = [col.strip() for col in columns_text.split(',')]
    
    # Group columns into batches
    batches = []
    for i in range(0, len(columns), batch_size):
        batch_columns = columns[i:i + batch_size]
        batch_ddl = f"{create_stmt}({','.join(batch_columns)})"
        batches.append(batch_ddl)
    
    return batches

def process_ddl_batch(ddl_batch: str) -> dict:
    """Process a single batch of DDL"""
    # Generate prompt for current batch
    prompt = f"""Analyze this DDL statement and provide an explanation for each column:
    {ddl_batch}
    For each column, provide a clear, concise explanation of what the column represents.
    Format as JSON: {{"column_name": "explanation of what this column represents"}}"""
    
    # Get analysis for batch
    return eval(get_llm_response(prompt))

# Modify the existing analysis section in main():
if not st.session_state.analysis_complete:
    if st.button("🔍 Analyze Structure"):
        with st.spinner("Analyzing structure and predicting sensitivity..."):
            try:
                # Check for existing analysis first
                existing_analysis = load_existing_feedback(st.session_state.current_schema, 
                                                        st.session_state.current_object)
                
                if existing_analysis is not None:
                    st.session_state.analysis_df = existing_analysis
                else:
                    # Split DDL into batches
                    ddl_batches = split_ddl_into_batches(ddl, batch_size=30)
                    
                    # Process each batch
                    all_results = []
                    total_batches = len(ddl_batches)
                    
                    progress_bar = st.progress(0)
                    for batch_idx, ddl_batch in enumerate(ddl_batches, 1):
                        # Process current batch
                        batch_analysis = process_ddl_batch(ddl_batch)
                        
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
                            for (col, explanation), sensitivity in zip(batch_analysis.items(), 
                                                                     batch_transformed)
                        ]
                        all_results.extend(batch_results)
                        
                        # Update progress
                        progress = batch_idx / total_batches
                        progress_bar.progress(progress)
                        st.write(f"Processing batch {batch_idx}/{total_batches}...")
                    
                    # Create final dataframe
                    st.session_state.analysis_df = pd.DataFrame(all_results)
                
                st.session_state.analysis_complete = True
                
            except Exception as e:
                st.error(f"Error analyzing DDL: {str(e)}")
                logger.error(f"DDL analysis error: {str(e)}")
