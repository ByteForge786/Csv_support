def split_ddl_into_batches(ddl: str, batch_size: int = 5, include_tags: bool = True) -> List[str]:
    """
    Split DDL into batches while preserving CREATE statement structure and column attributes
    
    Args:
        ddl (str): The complete DDL statement
        batch_size (int): Number of columns per batch (default 5)
        include_tags (bool): Whether to include TAG clauses in output (default True)
        
    Returns:
        List[str]: List of DDL statements, each containing a batch of numbered columns
    """
    try:
        # Extract create statement and columns
        create_pattern = r'(create\s+or\s+replace\s+(?:view|table)\s+[\w\._]+\s*\()([^;]+)(\)?;?)'
        match = re.match(create_pattern, ddl.strip(), re.IGNORECASE | re.DOTALL)
        
        if not match:
            raise ValueError(f"Invalid DDL format: {ddl[:100]}...")
            
        create_header = match.group(1)
        columns_text = match.group(2)
        closing = match.group(3) or ');'

        # Split into lines
        lines = columns_text.split('\n')
        
        # Process lines to group columns with their dummy values and tags
        columns = []
        current_column = []
        column_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('--'):
                # Add dummy values to current column
                current_column.append(line)
            elif line.startswith('primary key'):
                # Handle primary key separately
                if current_column:
                    columns.append('\n    '.join(current_column))
                    current_column = []
                columns.append(line)
            else:
                # If we have a previous column, add it to our list
                if current_column:
                    columns.append('\n    '.join(current_column))
                
                # Remove tags if include_tags is False
                if not include_tags and 'WITH TAG' in line:
                    line = re.sub(r'\s+WITH\s+TAG\s*\([^)]+\)', '', line)
                
                current_column = [line]
                
        # Add the last column if exists
        if current_column:
            columns.append('\n    '.join(current_column))
            
        # Clean up columns
        columns = [col.strip().rstrip(',') for col in columns if col.strip()]
        
        # Group into batches with column numbering
        batches = []
        for i in range(0, len(columns), batch_size):
            batch_columns = columns[i:i + batch_size]
            
            # Number and format columns
            numbered_columns = []
            for j, col in enumerate(batch_columns, i + 1):
                if col.startswith('primary key'):
                    numbered_columns.append(col)
                else:
                    # Add column number at the start
                    col_lines = col.split('\n')
                    col_lines.insert(0, f"-- Column {j}")
                    numbered_columns.append('\n    '.join(col_lines))
            
            # Add commas between columns
            formatted_columns = ',\n    '.join(numbered_columns)
            
            # Create complete DDL for this batch
            batch_ddl = f"{create_header}\n    {formatted_columns}\n{closing}"
            batches.append(batch_ddl)
        
        return batches
        
    except Exception as e:
        logger.error(f"Error splitting DDL: {str(e)}")
        raise

def test_ddl_splitter():
    """Test the DDL splitter with various cases"""
    
    # Test case with mixed column formats
    test_ddl = """
    create or replace view HYBRID_TEST_VIEW(
        SIMPLE_COLUMN,
        TYPED_COLUMN VARCHAR(100),
        -- Dummy Samples: value1, value2, value3
        COLUMN_WITH_DUMMY,
        COLUMN_WITH_TAG WITH TAG (REFERENCE.DATA='19'),
        -- Dummy Samples: 2024-01-01, 2024-01-02
        TYPED_WITH_TAG_AND_DUMMY NUMBER(38,0) WITH TAG (REFERENCE.DATA='20'),
        NULLABLE_COLUMN VARCHAR(50),
        -- Dummy Sample: {
        -- "key": "value"
        -- }
        JSON_DUMMY_COLUMN,
        NOT_NULL_COLUMN NUMBER(10) NOT NULL,
        -- Dummy Samples: 1, 2, 3
        SIMPLE_WITH_DUMMY,
        TAG_ONLY WITH TAG (COMPLEX.TAG='value', OTHER.TAG='test'),
        primary key (SIMPLE_COLUMN)
    );"""
    
    # Test with tags included
    print("\nTesting with tags included:")
    print("=" * 80)
    try:
        batches = split_ddl_into_batches(test_ddl, include_tags=True)
        for i, batch in enumerate(batches, 1):
            print(f"\nBatch {i}:")
            print(batch)
            print("-" * 80)
    except Exception as e:
        print(f"Error processing with tags: {str(e)}")
        
    # Test with tags excluded
    print("\nTesting with tags excluded:")
    print("=" * 80)
    try:
        batches = split_ddl_into_batches(test_ddl, include_tags=False)
        for i, batch in enumerate(batches, 1):
            print(f"\nBatch {i}:")
            print(batch)
            print("-" * 80)
    except Exception as e:
        print(f"Error processing without tags: {str(e)}")

if __name__ == "__main__":
    test_ddl_splitter()
