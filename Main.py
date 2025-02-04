def process_csv_pipeline():
    """Separate pipeline for processing uploaded CSV files"""
    uploaded_file = st.file_uploader("Upload CSV file containing 'attribute_name' column", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'attribute_name' not in df.columns:
                st.error("CSV must contain 'attribute_name' column")
                return
            
            if st.button("🔍 Analyze CSV Attributes"):
                with st.spinner("Analyzing attributes..."):
                    # Create classification prompts for each attribute
                    prompts = [
                        create_classification_prompt(attr_name, "Analyze the sensitivity of this column name") 
                        for attr_name in df['attribute_name']
                    ]
                    
                    # Get predictions using existing classification function
                    sensitivities = classify_sensitivity(prompts)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        "Column Name": df['attribute_name'],
                        "Explanation": ["" for _ in df['attribute_name']],  # Empty initially
                        "Data Sensitivity": sensitivities
                    })
                    
                    # Display results using the existing editor
                    st.dataframe(
                        results_df,
                        column_config={
                            "Column Name": st.column_config.TextColumn("Column Name", width="medium"),
                            "Explanation": st.column_config.TextColumn("Explanation", width="large"),
                            "Data Sensitivity": st.column_config.SelectboxColumn(
                                "Data Sensitivity",
                                width="medium",
                                options=SENSITIVITY_OPTIONS
                            )
                        },
                        hide_index=True
                    )
                    
                    # Add download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        f"sensitivity_analysis_{datetime.now():%Y%m%d_%H%M}.csv",
                        "text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            logger.error(f"CSV processing error: {str(e)}")

def main():
    st.set_page_config(page_title="DDL Analyzer", page_icon="🔍", layout="wide")
    apply_custom_css()
    
    st.title("🔍 DDL Analyzer")
    
    # Only show CSV option to allowed users
    ALLOWED_USERS = ['gowdas']
    current_user = os.getenv('USER', '').lower()
    
    with st.sidebar:
        if any(user in current_user for user in ALLOWED_USERS):
            analysis_type = st.radio("Select Analysis Type", ["Database Objects", "Upload CSV"])
        else:
            analysis_type = "Database Objects"
    
    if analysis_type == "Upload CSV":
        process_csv_pipeline()
    else:
        # Original database analysis code remains unchanged
        original_pipeline()
