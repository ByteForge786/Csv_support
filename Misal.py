def handle_csv_upload():
    """Handle CSV file upload and analysis"""
    # Add template download section
    st.markdown("### 📋 CSV Template")
    st.markdown("Your CSV should contain an `attribute_name` column with database column names.")
    
    # Create sample template data
    template_data = {
        'attribute_name': ['customer_id', 'email_address', 'birth_date']
    }
    template_df = pd.DataFrame(template_data)
    
    # Download template button
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_template,
        file_name="ddl_analyzer_template.csv",
        mime="text/csv",
    )
    
    st.markdown("---")  # Add separator
    
    # Existing upload code
    if 'csv_processed' not in st.session_state:
        st.session_state.csv_processed = False
    # ... rest of the existing code


# Add example format section
    with st.expander("👀 View Expected CSV Format"):
        st.markdown("""
        ### Required CSV Format:
        
        Your CSV file should:
        - Have a header row
        - Contain an `attribute_name` column
        - One column name per row
        
        Example:
        ```
        attribute_name
        customer_id
        email_address
        birth_date
        phone_number
        ```
        """)
        
        # Show sample preview
        st.markdown("### Sample Preview:")
        st.dataframe(template_df, use_container_width=True)
