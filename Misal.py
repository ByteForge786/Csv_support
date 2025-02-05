def main():
    st.set_page_config(page_title="DDL Analyzer", page_icon="🔍", layout="wide")
    apply_custom_css()
    
    st.title("🔍 DDL Analyzer")
    st.markdown("Analyze your database objects structure and get intelligent insights.")

    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_database = None  # Add this line
        st.session_state.current_schema = None
        st.session_state.current_object_type = None
        st.session_state.current_object = None
        st.session_state.analysis_df = None
        st.session_state.editing_enabled = False
        st.session_state.edited_cells = {}
        st.session_state.analysis_complete = False
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
            
            if analysis_type != st.session_state.analysis_type:
                st.session_state.analysis_type = analysis_type
                if analysis_type == "Database Object":
                    st.session_state.csv_file = None
                    st.session_state.csv_processed = False
                else:
                    st.session_state.current_database = None  # Reset database
                    st.session_state.current_schema = None
                    st.session_state.current_object = None
                    st.session_state.current_object_type = None
                st.session_state.analysis_complete = False
                st.session_state.analysis_df = None
                st.experimental_rerun()
        
        if st.session_state.analysis_type == "Database Object":
            st.header("Object Selection")
            
            # 1. Database Selection
            databases = get_database_list()
            selected_database = st.selectbox("1. Select Database", databases)
            
            if selected_database != st.session_state.current_database:
                st.session_state.current_database = selected_database
                st.session_state.current_schema = None
                st.session_state.current_object_type = None
                st.session_state.current_object = None
                st.session_state.analysis_complete = False
                st.session_state.analysis_df = None
                st.experimental_rerun()
            
            # 2. Schema Selection
            if st.session_state.current_database:
                schemas = get_schema_list(st.session_state.current_database)
                selected_schema = st.selectbox("2. Select Schema", schemas)
                
                if selected_schema != st.session_state.current_schema:
                    st.session_state.current_schema = selected_schema
                    st.session_state.analysis_complete = False
                    st.session_state.analysis_df = None
                
                # 3. Object Type Selection
                if st.session_state.current_schema:
                    object_type = st.radio("3. Select Object Type", ["TABLE", "VIEW"])
                    if object_type != st.session_state.current_object_type:
                        st.session_state.current_object_type = object_type
                        st.session_state.analysis_complete = False
                        st.session_state.analysis_df = None
                    
                    # 4. Object Selection
                    if st.session_state.current_object_type:
                        schema_objects = get_schema_objects(
                            st.session_state.current_database,
                            st.session_state.current_schema
                        )
                        object_list = schema_objects["tables"] if object_type == "TABLE" else schema_objects["views"]
                        selected_object = st.selectbox(f"4. Select {object_type}", object_list)
                        
                        if selected_object != st.session_state.current_object:
                            st.session_state.current_object = selected_object
                            st.session_state.analysis_complete = False
                            st.session_state.analysis_df = None

    # Main content area remains the same but needs database context
    if st.session_state.analysis_type == "CSV Upload":
        handle_csv_upload()
        
        if st.session_state.csv_processed and st.session_state.analysis_complete:
            st.subheader("📊 Analysis Results")
            display_editable_table()
    else:
        if all([st.session_state.current_database,  # Add database check
               st.session_state.current_schema, 
               st.session_state.current_object_type, 
               st.session_state.current_object]):
            try:
                # Get DDL and samples with database context
                ddl, samples = get_ddl_and_samples(
                    st.session_state.current_database,
                    st.session_state.current_schema, 
                    st.session_state.current_object, 
                    st.session_state.current_object_type)
                
                # Rest of the code remains the same
                # ...


def save_feedback(database, schema, table, feedback_df):
    """Save or update feedback in CSV file"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_records = []
    
    for _, row in feedback_df.iterrows():
        new_records.append({
            'database': database,  # Add database
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
        existing_df = existing_df[~((existing_df['database'] == database) &  # Add database check
                                  (existing_df['schema'] == schema) & 
                                  (existing_df['table'] == table))]
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df
    
    updated_df.to_csv(FEEDBACK_FILE, index=False)
    
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
