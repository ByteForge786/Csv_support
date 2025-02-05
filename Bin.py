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
        List[str]: List of DDL statements, each containing a batch of columns
        
    Raises:
        ValueError: If DDL cannot be parsed properly
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
        
        # Pattern to match complete column definitions
        column_pattern = r'''
            # Optional dummy samples in comments
            (?:(?:--[^\n]*\n)*)?
            # Column name
            (?:[\w\._]+)
            # Optional datatype
            (?:\s+(?:VARCHAR|NUMBER|TIMESTAMP_NTZ|VARIANT|TIMESTAMP|DATE)\s*(?:\([^)]+\))?)?
            # Optional tag clause with nested parentheses
            (?:\s+(?:WITH\s+TAG\s*\([^)]+\)))?
            # Optional constraints
            (?:\s+(?:NOT\s+NULL|PRIMARY\s+KEY|DEFAULT[^,]+))?
            # Capture any remaining part until the next column or end
            [^,]*?
            # Look ahead for next column or end
            (?=\s*,|\s*$)
        '''
        
        # Get all column definitions with their comments
        columns = []
        current_comments = []
        
        for line in columns_text.split('\n'):
            line = line.strip()
            if line.startswith('--'):
                current_comments.append(line)
            elif line and not line.startswith('--'):
                if current_comments:
                    columns.append('\n'.join(current_comments + [line]))
                    current_comments = []
                else:
                    columns.append(line)
        
        # Clean up columns
        columns = [col.strip().rstrip(',') for col in columns if col.strip()]
        
        # Group into batches
        batches = []
        for i in range(0, len(columns), batch_size):
            batch_columns = columns[i:i + batch_size]
            
            # Add commas between columns
            formatted_columns = ',\n    '.join(batch_columns)
            
            # Create complete DDL for this batch
            batch_ddl = f"{create_header}\n    {formatted_columns}\n{closing}"
            batches.append(batch_ddl)
        
        return batches
        
    except Exception as e:
        logger.error(f"Error splitting DDL: {str(e)}")
        raise

def test_ddl_splitter():
    """Test the DDL splitter with various cases"""
    
    # Test case 1: Simple CREATE TABLE
    test_case_1 = """
    create or replace table KSMM_MASTER_20250128 (
        SK_KSMM_LEVEL_ID NUMBER(38,0) NOT NULL,
        KSMM_LEVEL_NAME VARCHAR(100),
        KSMM_LEVEL_NUMBER NUMBER(38,0),
        KSMM_LEVEL_DESC VARCHAR(1000),
        SRCUPDATEDTS TIMESTAMP_NTZ(9),
        AUDITCREATEDTS TIMESTAMP_NTZ(9),
        AUDITUPDATEDTS TIMESTAMP_NTZ(9),
        CREATE_USER VARCHAR(100),
        UPDATE_USER VARCHAR(100),
        UPDATE_TS TIMESTAMP_NTZ(9),
        CREATE_TS TIMESTAMP_NTZ(9),
        primary key (SK_KSMM_LEVEL_ID)
    );"""
    
    # Test case 2: CREATE VIEW with tags and dummy samples
    test_case_2 = """
    create or replace view RMEP_NHANCE_EXCEPTION_VIEW(
        EXCEPTION_ID,
        -- Dummy Sample: post_trade_rmep|8000060|GLOBAL
        COBDATE WITH TAG (NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_CONCEPT='19', NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_SENSITIVITY='CI'),
        -- Dummy Samples: 2024-11-27, 2024-12-19, 2024-12-35
        AGE WITH TAG (NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_CONCEPT='19'),
        -- Dummy Samples: 2, 8, 2
        SK_OWNER_ID,
        -- Dummy Samples: 1352, 1352, 1352
        ASSIGNEDTO WITH TAG (NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_CONCEPT='19'),
        -- Dummy Samples: pupakos, quinnelk, shelorak
        CATEGORY WITH TAG (NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_CONCEPT='19'),
        -- Dummy Samples: Non Genuine, Non Genuine, To be Investigated
        CLOSURECOB WITH TAG (NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_CONCEPT='19')
    );"""
    
    # Test case 3: CREATE TABLE with JSON dummy samples
    test_case_3 = """
    create or replace TABLE EXCEPTION_ATTRIBUTE_BKP20241223 (
        EXCEPTION_ID VARCHAR(5000),
        -- Dummy Sample: TRADEACCOUNT_COUNTERPARTY_INTE
        DATAVALUE VARIANT,
        -- Dummy Sample: {
        -- "CobDate": "2024-07-04",
        -- }
        COBDATE VARCHAR(5000),
        -- Dummy Samples: 2024-07-04, 2024-07-04, 2024-07-04
        SK_OWNER_ID NUMBER(38,0),
        -- Dummy Samples: 12, 12, 12
        CREATE_TS TIMESTAMP_NTZ(9)
        -- Dummy Samples: 2024-07-05 17:55:49.395000, 2024-07-05 17:55:49.395000
    );"""
    
    test_cases = [test_case_1, test_case_2, test_case_3]
    
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
