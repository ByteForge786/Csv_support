import re
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ddl_splitter.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def split_ddl_into_batches(ddl: str, batch_size: int = 5) -> List[str]:
    """
    Split DDL into batches while preserving CREATE statement structure and column attributes
    
    Args:
        ddl (str): The complete DDL statement
        batch_size (int): Number of columns per batch (default 5)
        
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
    
    # Test case 1: Hybrid mix of column formats
    test_case_1 = """
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
    
    # Test case 2: CREATE TABLE with mixed formats
    test_case_2 = """
    create or replace TABLE MIXED_FORMAT_TABLE (
        ID NUMBER(38,0) NOT NULL,
        -- Dummy Sample: SAMPLE_VALUE
        NAME VARCHAR(200),
        -- Dummy Samples: 2024-07-04, 2024-07-05
        DATE_COLUMN DATE WITH TAG (DATA.SENSITIVITY='HIGH'),
        SIMPLE_COL,
        -- Dummy Samples: 100.5, 200.5
        AMOUNT NUMBER(10,2),
        STATUS VARCHAR(50) WITH TAG (DATA.TYPE='STATUS'),
        -- Dummy Sample: active
        ACTIVE_FLAG WITH TAG (DATA.TYPE='FLAG'),
        TIMESTAMP_COL TIMESTAMP_NTZ(9),
        -- Dummy Samples: {"status": "new"}
        JSON_DATA VARIANT,
        REGULAR_COLUMN VARCHAR(100),
        primary key (ID)
    );"""

    test_cases = [test_case_1, test_case_2]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTesting Case {i}:")
        print("=" * 80)
        print("Original DDL:")
        print(test_case)
        print("\nSplit into batches:")
        try:
            batches = split_ddl_into_batches(test_case)
            for j, batch in enumerate(batches, 1):
                print(f"\nBatch {j}:")
                print(batch)
                print("-" * 80)
        except Exception as e:
            print(f"Error processing test case {i}: {str(e)}")

if __name__ == "__main__":
    test_ddl_splitter()
