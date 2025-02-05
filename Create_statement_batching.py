import re
import json
from typing import List, Dict, Union, Optional
import unittest

class DDLParser:
    def __init__(self):
        # Regex patterns
        self.create_pattern = re.compile(
            r'create\s+or\s+replace\s+(TABLE|VIEW)\s+([^\s]+)\s*\((.*)\)',
            re.DOTALL | re.IGNORECASE
        )
        
        self.column_pattern = re.compile(
            r'''
            ^\s*
            (\w+)                              # Column name
            (?:\s+([^-\s]+(?:\([^)]+\))?)?)?  # Optional datatype
            (?:\s+WITH\s+TAG\s+(\([^)]+\)))?  # Optional TAG clause
            (?:\s*--\s*Dummy\s*Samples?:?\s*(.+?))? # Optional dummy samples
            \s*$
            ''',
            re.VERBOSE | re.IGNORECASE | re.DOTALL
        )

    def parse_ddl(self, ddl: str, include_tags: bool = False) -> Dict:
        """
        Parse DDL statement and extract components
        
        Args:
            ddl (str): The DDL statement
            include_tags (bool): Whether to include TAG clauses with column names
            
        Returns:
            dict: Dictionary containing parsed DDL information
        """
        # Extract create statement components
        create_match = self.create_pattern.match(ddl)
        if not create_match:
            raise ValueError("Invalid DDL format")
            
        obj_type, obj_name, columns_text = create_match.groups()
        
        # Split columns while preserving nested parentheses
        columns = self._split_columns(columns_text)
        
        # Process each column
        parsed_columns = []
        for col in columns:
            if col_info := self._parse_column(col, include_tags):
                parsed_columns.append(col_info)
        
        return {
            'type': obj_type,
            'name': obj_name,
            'columns': parsed_columns
        }
    
    def _split_columns(self, columns_text: str) -> List[str]:
        """Split columns while handling nested parentheses"""
        columns = []
        current_col = []
        paren_count = 0
        
        for char in columns_text:
            if char == '(' and not current_col:
                continue
            elif char == '(':
                paren_count += 1
                current_col.append(char)
            elif char == ')':
                paren_count -= 1
                current_col.append(char)
            elif char == ',' and paren_count == 0:
                if current_col:
                    columns.append(''.join(current_col).strip())
                    current_col = []
            else:
                current_col.append(char)
        
        if current_col:
            columns.append(''.join(current_col).strip())
        
        return [col for col in columns if col.strip()]
    
    def _parse_column(self, column_text: str, include_tags: bool) -> Optional[Dict]:
        """Parse individual column definition"""
        match = self.column_pattern.match(column_text)
        if not match:
            return None
            
        col_name, datatype, tag, samples = match.groups()
        
        # Process samples if present
        processed_samples = None
        if samples:
            samples = samples.strip()
            try:
                if samples.startswith('{'):
                    # Handle JSON-like samples
                    processed_samples = samples
                else:
                    # Handle comma-separated samples
                    processed_samples = [s.strip() for s in samples.split(',')]
            except:
                processed_samples = samples
        
        # Build column info
        column_info = {
            'name': f"{col_name} WITH TAG {tag}" if include_tags and tag else col_name,
            'datatype': datatype,
            'samples': processed_samples
        }
        
        if tag and include_tags:
            column_info['tag'] = tag
            
        return column_info


class TestDDLParser(unittest.TestCase):
    def setUp(self):
        self.parser = DDLParser()
        
    def test_view_with_tags(self):
        ddl = """
        create or replace view RMEP_NHANCE_EXCEPTION_VIEW(
            EXCEPTION_ID,
            -- Dummy Sample: post_trade_rmep|8000060|GLOBAL
            COBDATE WITH TAG (NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_CONCEPT='19', NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_SENSITIVITY='CI'),
            -- Dummy Samples: 2024-11-27, 2024-12-19, 2024-12-31
            SK_OWNER_ID,
            -- Dummy Samples: 1352, 1352, 1352
            CATEGORY WITH TAG (NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_CONCEPT='19')
            -- Dummy Samples: Non Genuine, Non Genuine, To be Investigated
        )
        """
        result = self.parser.parse_ddl(ddl, include_tags=True)
        self.assertEqual(result['type'].upper(), 'VIEW')
        self.assertTrue(any('WITH TAG' in col['name'] for col in result['columns']))
        
    def test_table_without_tags(self):
        ddl = """
        create or replace TABLE KSMM_MASTER_20250128 (
            SK_KSMM_LEVEL_ID NUMBER(38,0) NOT NULL,
            KSMM_LEVEL_NAME VARCHAR(100),
            KSMM_LEVEL_NUMBER(38,0),
            KSMM_LEVEL_DESC VARCHAR(1000),
            SRCUPDATEDTS TIMESTAMP_NTZ(9),
            primary key (SK_KSMM_LEVEL_ID)
        );
        """
        result = self.parser.parse_ddl(ddl, include_tags=False)
        self.assertEqual(result['type'].upper(), 'TABLE')
        self.assertTrue(all('WITH TAG' not in col['name'] for col in result['columns']))
        
    def test_json_dummy_samples(self):
        ddl = """
        create or replace TABLE EXCEPTION_ATTRIBUTE_BKP20241223 (
            EXCEPTION_ID VARCHAR(5000),
            -- Dummy Sample: {
            "CobDate": "2024-07-04",
            }
            DATAVALUE VARIANT,
            CREATE_TS TIMESTAMP_NTZ(9)
        );
        """
        result = self.parser.parse_ddl(ddl)
        json_sample_col = next(col for col in result['columns'] if col['samples'])
        self.assertTrue(isinstance(json_sample_col['samples'], str))
        self.assertTrue('"CobDate"' in json_sample_col['samples'])
        
    def test_edge_cases(self):
        # Test various edge cases in one DDL
        ddl = """
        create or replace VIEW complex_view (
            col1,  -- Empty column
            col2 WITH TAG (tag.with.dots='value'),  -- Tag with dots
            col3 NUMBER(38,0),  -- Just datatype
            col4 WITH TAG (nested=(sub1, sub2)),  -- Nested tag
            col5  -- Dummy Samples: val1, val2, val3  -- Multiple comments
        );
        """
        result = self.parser.parse_ddl(ddl, include_tags=True)
        self.assertTrue(len(result['columns']) > 0)
        
    def test_invalid_ddl(self):
        invalid_ddl = "create table missing_parentheses"
        with self.assertRaises(ValueError):
            self.parser.parse_ddl(invalid_ddl)

def main():
    # Example usage
    parser = DDLParser()
    
    # Example DDL from your images
    ddl = """
    create or replace view RMEP_NHANCE_EXCEPTION_VIEW(
        EXCEPTION_ID,
        -- Dummy Sample: post_trade_rmep|8000060|GLOBAL
        COBDATE WITH TAG (NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_CONCEPT='19', NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_SENSITIVITY='CI'),
        -- Dummy Samples: 2024-11-27, 2024-12-19, 2024-12-31
        AGE WITH TAG (NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_CONCEPT='19'),
        -- Dummy Samples: 2, 8, 2
        SK_OWNER_ID,
        -- Dummy Samples: 1352, 1352, 1352
        ASSIGNEDTO WITH TAG (NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_CONCEPT='19'),
        -- Dummy Samples: aupakos, quinnelk, shelorak
        CATEGORY WITH TAG (NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_CONCEPT='19'),
        -- Dummy Samples: Non Genuine, Non Genuine, To be Investigated
        CLOSURECOB WITH TAG (NUCLEUS_METAHUB.GOVERNANCE_STATIC_REFERENCES.DATA_CONCEPT='19')
    )
    """
    
    # Parse with tags
    result_with_tags = parser.parse_ddl(ddl, include_tags=True)
    print("\nParsed DDL with tags:")
    print(json.dumps(result_with_tags, indent=2))
    
    # Parse without tags
    result_without_tags = parser.parse_ddl(ddl, include_tags=False)
    print("\nParsed DDL without tags:")
    print(json.dumps(result_without_tags, indent=2))
    
    # Run tests
    unittest.main(argv=[''], exit=False)

if __name__ == "__main__":
    main()
