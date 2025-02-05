def format_ddl_batches(parsed_ddl: Dict, batch_size: int = 30) -> List[str]:
    """
    Format parsed DDL into batches with CREATE statement headers
    
    Args:
        parsed_ddl (Dict): Parsed DDL information
        batch_size (int): Number of columns per batch
        
    Returns:
        List[str]: List of DDL batches with headers
    """
    obj_type = parsed_ddl['type']
    obj_name = parsed_ddl['name']
    columns = parsed_ddl['columns']
    
    # Create header
    header = f"create or replace {obj_type} {obj_name}"
    
    # Split columns into batches
    batches = []
    for i in range(0, len(columns), batch_size):
        batch_columns = columns[i:i + batch_size]
        
        # Format column definitions
        column_defs = []
        for col in batch_columns:
            col_def = col['name']
            if col.get('datatype'):
                col_def += f" {col['datatype']}"
            if col.get('samples'):
                samples = col['samples']
                if isinstance(samples, list):
                    col_def += f"\n    -- Dummy Samples: {', '.join(samples)}"
                else:
                    col_def += f"\n    -- Dummy Sample: {samples}"
            column_defs.append(col_def)
        
        # Create batch DDL
        batch_ddl = f"{header}(\n    " + ",\n    ".join(column_defs) + "\n)"
        batches.append(batch_ddl)
    
    return batches

# Modify main() to use the new format
def main():
    parser = DDLParser()
    
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
    
    try:
        # Parse with tags
        print("\nDDL Batches with tags:")
        print("-" * 80)
        result_with_tags = parser.parse_ddl(ddl, include_tags=True)
        batches_with_tags = format_ddl_batches(result_with_tags, batch_size=3)
        for i, batch in enumerate(batches_with_tags, 1):
            print(f"\nBatch {i}:")
            print(batch)
            print("-" * 80)
        
        # Parse without tags
        print("\nDDL Batches without tags:")
        print("-" * 80)
        result_without_tags = parser.parse_ddl(ddl, include_tags=False)
        batches_without_tags = format_ddl_batches(result_without_tags, batch_size=3)
        for i, batch in enumerate(batches_without_tags, 1):
            print(f"\nBatch {i}:")
            print(batch)
            print("-" * 80)
        
    except Exception as e:
        print(f"Error parsing DDL: {str(e)}")

if __name__ == "__main__":
    main()


#=============working cide below==========
import re
import json
from typing import List, Dict, Union, Optional
import unittest

class DDLParser:
    def __init__(self):
        # Modified regex patterns to be more flexible with whitespace
        self.create_pattern = re.compile(
            r'''
            \s*create\s+or\s+replace\s+   # CREATE OR REPLACE with flexible spacing
            (TABLE|VIEW)\s+               # TABLE or VIEW
            ([^\s(]+)\s*                  # Object name
            \(\s*                         # Opening parenthesis with optional whitespace
            (.*?)                         # Column definitions (non-greedy)
            \s*\)\s*                      # Closing parenthesis with optional whitespace
            ''',
            re.DOTALL | re.IGNORECASE | re.VERBOSE
        )
        
        self.column_pattern = re.compile(
            r'''
            ^\s*
            (\w+)                                 # Column name
            (?:\s+([^\s,]+(?:\([^)]+\))?)?)?     # Optional datatype
            (?:\s+WITH\s+TAG\s+(\([^)]+\)))?     # Optional TAG clause
            (?:\s*--\s*Dummy\s*Samples?:?\s*(.+?))?\s*  # Optional dummy samples
            $
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
        # Clean input DDL
        ddl = ddl.strip()
        
        # Extract create statement components
        create_match = self.create_pattern.match(ddl)
        if not create_match:
            raise ValueError(f"Invalid DDL format: {ddl}")
            
        obj_type, obj_name, columns_text = create_match.groups()
        
        # Split columns while preserving nested parentheses
        columns = self._split_columns(columns_text)
        
        # Process each column
        parsed_columns = []
        for col in columns:
            if col_info := self._parse_column(col.strip(), include_tags):
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
        in_comment = False
        
        for char in columns_text:
            if char == '-' and current_col and current_col[-1] == '-':
                in_comment = True
            elif char == '\n':
                in_comment = False
            
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            
            if char == ',' and paren_count == 0 and not in_comment:
                columns.append(''.join(current_col).strip())
                current_col = []
            else:
                current_col.append(char)
        
        if current_col:
            columns.append(''.join(current_col).strip())
        
        return [col for col in columns if col.strip()]
    
    def _parse_column(self, column_text: str, include_tags: bool) -> Optional[Dict]:
        """Parse individual column definition"""
        # Handle comments in column text
        lines = column_text.split('\n')
        col_def = lines[0].strip()
        samples = None
        
        # Look for dummy samples in subsequent lines
        for line in lines[1:]:
            if 'Dummy Sample' in line:
                samples = line.split(':', 1)[1].strip() if ':' in line else line.split('Dummy Sample', 1)[1].strip()
        
        match = self.column_pattern.match(col_def)
        if not match:
            return None
            
        col_name, datatype, tag, pattern_samples = match.groups()
        
        # Use samples from either pattern or subsequent lines
        samples = pattern_samples or samples
        
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
            'datatype': datatype
        }
        
        if processed_samples:
            column_info['samples'] = processed_samples
            
        if tag and include_tags:
            column_info['tag'] = tag
            
        return column_info


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
    
    try:
        # Parse with tags
        result_with_tags = parser.parse_ddl(ddl, include_tags=True)
        print("\nParsed DDL with tags:")
        print(json.dumps(result_with_tags, indent=2))
        
        # Parse without tags
        result_without_tags = parser.parse_ddl(ddl, include_tags=False)
        print("\nParsed DDL without tags:")
        print(json.dumps(result_without_tags, indent=2))
        
    except Exception as e:
        print(f"Error parsing DDL: {str(e)}")

if __name__ == "__main__":
    main()
