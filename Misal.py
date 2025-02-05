
@st.cache_data(ttl=3600)
def get_database_list() -> List[str]:
    """Fetches list of databases - cached globally for 1 hour"""
    logger.info("Fetching database list")
    try:
        query = """
        SELECT DATABASE_NAME as name 
        FROM information_schema.databases 
        WHERE DATABASE_NAME NOT LIKE 'INFORMATION_SCHEMA%'
        ORDER BY DATABASE_NAME
        """
        df = get_data_sf(query)
        databases = df['name'].tolist()
        logger.info(f"Cached {len(databases)} databases globally")
        return databases
    except Exception as e:
        logger.error(f"Error fetching databases: {str(e)}")
        raise

@st.cache_data(ttl=3600)
def get_schema_list(database: str) -> List[str]:
    """Fetches list of schemas for given database - cached globally for 1 hour"""
    logger.info(f"Fetching schema list for database: {database}")
    try:
        query = f"""
        SELECT SCHEMA_NAME as name 
        FROM {database}.information_schema.schemata 
        WHERE SCHEMA_NAME NOT LIKE 'INFORMATION_SCHEMA%'
        ORDER BY SCHEMA_NAME
        """
        df = get_data_sf(query)
        schemas = df['name'].tolist()
        logger.info(f"Cached {len(schemas)} schemas for database {database}")
        return schemas
    except Exception as e:
        logger.error(f"Error fetching schemas for {database}: {str(e)}")
        raise

@st.cache_data(ttl=3600)
def get_schema_objects(database: str, schema: str) -> Dict[str, List[str]]:
    """Fetches objects for a schema in given database - cached globally for 1 hour"""
    logger.info(f"Fetching objects for database: {database}, schema: {schema}")
    try:
        # Get tables
        tables_query = f"""
        SELECT TABLE_NAME as name
        FROM {database}.information_schema.tables
        WHERE TABLE_SCHEMA = '{schema}'
        AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        tables_df = get_data_sf(tables_query)
        
        # Get views
        views_query = f"""
        SELECT TABLE_NAME as name
        FROM {database}.information_schema.views
        WHERE TABLE_SCHEMA = '{schema}'
        ORDER BY TABLE_NAME
        """
        views_df = get_data_sf(views_query)
        
        result = {
            "tables": tables_df['name'].tolist(),
            "views": views_df['name'].tolist()
        }
        logger.info(f"Cached objects for {database}.{schema}: {len(result['tables'])} tables, {len(result['views'])} views")
        return result
    except Exception as e:
        logger.error(f"Error fetching objects for {database}.{schema}: {str(e)}")
        raise




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
