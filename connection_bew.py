import streamlit as st
import snowflake.connector
import pandas as pd

def get_snowflake_connection():
    try:
        env = "dev"
        property_values = fetch_properties(env)
        conn = snowflake.connector.connect(
            user=property_values['snowflake.dbUsername'],
            password=property_values['snowflake.dbPassword'],
            account=property_values['snowflake.account']
        )
        return conn
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def get_databases(conn):
    cur = conn.cursor()
    cur.execute("SHOW DATABASES")
    return [row[1] for row in cur.fetchall()]

def get_schemas(conn, database):
    cur = conn.cursor()
    cur.execute(f"SHOW SCHEMAS IN DATABASE {database}")
    return [row[1] for row in cur.fetchall()]

def get_objects(conn, database, schema, object_type):
    cur = conn.cursor()
    cur.execute(f"SHOW {object_type}S IN SCHEMA {database}.{schema}")
    return [row[1] for row in cur.fetchall()]

def get_ddl(conn, database, schema, object_name, object_type):
    cur = conn.cursor()
    cur.execute(f"SELECT GET_DDL('{object_type}', '{database}.{schema}.{object_name}')")
    return cur.fetchone()[0]

def main():
    st.title("Snowflake Object Explorer")
    
    conn = get_snowflake_connection()
    if not conn:
        return

    # Database selection
    databases = get_databases(conn)
    selected_db = st.selectbox("Select Database", databases)

    if selected_db:
        # Schema selection
        schemas = get_schemas(conn, selected_db)
        selected_schema = st.selectbox("Select Schema", schemas)

        if selected_schema:
            # Object type selection
            object_type = st.selectbox("Select Object Type", ["TABLE", "VIEW"])
            
            # Object selection
            objects = get_objects(conn, selected_db, selected_schema, object_type)
            selected_object = st.selectbox(f"Select {object_type}", objects)

            if selected_object:
                # Show DDL
                ddl = get_ddl(conn, selected_db, selected_schema, selected_object, object_type)
                st.code(ddl, language='sql')

if __name__ == "__main__":
    main()





# Add this to your session state initialization
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

# Add this new function to get databases
def get_database_list():
    """Get list of available databases"""
    try:
        # Use your snowflake connection to get databases
        conn = get_snowflake_connection()
        cur = conn.cursor()
        cur.execute("SHOW DATABASES")
        databases = [row[1] for row in cur.fetchall()]
        return databases
    except Exception as e:
        logger.error(f"Error fetching databases: {str(e)}")
        return []

# Modify your get_schema_list function to include database parameter
def get_schema_list(database):
    """Get list of schemas for selected database"""
    try:
        conn = get_snowflake_connection()
        cur = conn.cursor()
        cur.execute(f"SHOW SCHEMAS IN DATABASE {database}")
        schemas = [row[1] for row in cur.fetchall()]
        return schemas
    except Exception as e:
        logger.error(f"Error fetching schemas: {str(e)}")
        return []

# Modify your get_schema_objects function to include database parameter
def get_schema_objects(database, schema):
    """Get tables and views in the schema"""
    try:
        conn = get_snowflake_connection()
        cur = conn.cursor()
        
        # Get tables
        cur.execute(f"SHOW TABLES IN {database}.{schema}")
        tables = [row[1] for row in cur.fetchall()]
        
        # Get views
        cur.execute(f"SHOW VIEWS IN {database}.{schema}")
        views = [row[1] for row in cur.fetchall()]
        
        return {"tables": tables, "views": views}
    except Exception as e:
        logger.error(f"Error fetching schema objects: {str(e)}")
        return {"tables": [], "views": []}

# Modify the sidebar section in main() function
if st.session_state.analysis_type == "Database Object":
    st.header("Object Selection")
    
    # 1. Database selection
    databases = get_database_list()
    selected_database = st.selectbox("1. Select Database", databases)
    
    if selected_database != st.session_state.current_database:
        st.session_state.current_database = selected_database
        st.session_state.current_schema = None
        st.session_state.current_object = None
        st.session_state.analysis_complete = False
        st.session_state.analysis_df = None
    
    # 2. Schema selection
    if selected_database:
        schemas = get_schema_list(selected_database)
        selected_schema = st.selectbox("2. Select Schema", schemas)
        
        if selected_schema != st.session_state.current_schema:
            st.session_state.current_schema = selected_schema
            st.session_state.current_object = None
            st.session_state.analysis_complete = False
            st.session_state.analysis_df = None
    
    # 3. Object type selection
    object_type = st.radio("3. Select Object Type", ["TABLE", "VIEW"])
    if object_type != st.session_state.current_object_type:
        st.session_state.current_object_type = object_type
        st.session_state.current_object = None
        st.session_state.analysis_complete = False
        st.session_state.analysis_df = None
    
    # 4. Object selection
    if selected_database and selected_schema:
        schema_objects = get_schema_objects(selected_database, selected_schema)
        object_list = schema_objects["tables"] if object_type == "TABLE" else schema_objects["views"]
        selected_object = st.selectbox(f"4. Select {object_type}", object_list)
        
        if selected_object != st.session_state.current_object:
            st.session_state.current_object = selected_object
            st.session_state.analysis_complete = False
            st.session_state.analysis_df = None
