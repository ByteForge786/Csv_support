import re
from typing import List, Dict, Optional

class DDLParser:
    def __init__(self):
        # Simplified regex patterns
        self.create_match = re.compile(
            r'create\s+or\s+replace\s+(TABLE|VIEW)\s+(\w+)\s*\((.*)\)',
            re.DOTALL | re.IGNORECASE
        )

    def parse_ddl(self, ddl: str, include_tags: bool = False) -> Dict:
        """Parse DDL statement and extract components"""
        ddl = ddl.strip()
        match = self.create_match.search(ddl)
        
        if not match:
            raise ValueError("Invalid DDL format")
            
        obj_type, obj_name, columns_text = match.groups()
        columns = self._split_columns(columns_text)
        
        return {
            'type': obj_type,
            'name': obj_name,
            'columns': columns
        }

    def _split_columns(self, text: str) -> List[Dict]:
        """Split and process columns"""
        # Remove newlines between tags to preserve them
        text = re.sub(r'(\([^)]+)\n([^)]+\))', r'\1 \2', text)
        
        # Split on commas, preserving newlines for comments
        columns = []
        current = []
        paren_level = 0
        
        for char in text:
            if char == '(':
                paren_level += 1
            elif char == ')':
                paren_level -= 1
            
            if char == ',' and paren_level == 0:
                columns.append(''.join(current))
                current = []
            else:
                current.append(char)
                
        if current:
            columns.append(''.join(current))
        
        # Process each column
        processed_columns = []
        for col in columns:
            col_info = self._process_column(col.strip(), include_tags)
            if col_info:
                processed_columns.append(col_info)
                
        return processed_columns

    def _process_column(self, col_text: str, include_tags: bool) -> Optional[Dict]:
        """Process a single column definition"""
        lines = [line.strip() for line in col_text.split('\n') if line.strip()]
        if not lines:
            return None
            
        # First line contains column definition
        col_def = lines[0]
        
        # Extract components
        tag_match = re.search(r'WITH\s+TAG\s+(\([^)]+\))', col_def)
        tag = tag_match.group(1) if tag_match else None
        
        # Remove tag from column definition if present
        if tag:
            col_def = col_def[:tag_match.start()].strip()
            
        # Get column name and datatype
        parts = col_def.split()
        col_name = parts[0]
        datatype = ' '.join(parts[1:]) if len(parts) > 1 else None
        
        # Look for dummy samples in other lines
        samples = None
        for line in lines[1:]:
            if 'Dummy Sample' in line:
                sample_text = line.split(':', 1)[1].strip() if ':' in line else line.split('Dummy Sample', 1)[1].strip()
                if sample_text.startswith('{'):
                    samples = sample_text
                else:
                    samples = [s.strip() for s in sample_text.split(',')]
                break
        
        # Build result
        result = {
            'name': f"{col_name} WITH TAG {tag}" if include_tags and tag else col_name,
        }
        
        if datatype:
            result['datatype'] = datatype
        if samples:
            result['samples'] = samples
        if tag and include_tags:
            result['tag'] = tag
            
        return result

def format_ddl_batches(parsed_ddl: Dict, batch_size: int = 30) -> List[str]:
    """Format parsed DDL into batches"""
    obj_type = parsed_ddl['type']
    obj_name = parsed_ddl['name']
    columns = parsed_ddl['columns']
    
    # Create batches
    batches = []
    for i in range(0, len(columns), batch_size):
        batch_columns = columns[i:i + batch_size]
        
        # Format column definitions
        col_defs = []
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
            col_defs.append(col_def)
        
        # Create batch DDL
        batch_ddl = f"create or replace {obj_type} {obj_name}(\n    " + ",\n    ".join(col_defs) + "\n)"
        batches.append(batch_ddl)
    
    return batches

def main():
    # Your test DDL
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
    
    parser = DDLParser()
    try:
        print("\nDDL Batches with tags:")
        print("-" * 80)
        parsed_with_tags = parser.parse_ddl(ddl, include_tags=True)
        batches = format_ddl_batches(parsed_with_tags, batch_size=3)
        for i, batch in enumerate(batches, 1):
            print(f"\nBatch {i}:")
            print(batch)
            print("-" * 80)
        
        print("\nDDL Batches without tags:")
        print("-" * 80)
        parsed_without_tags = parser.parse_ddl(ddl, include_tags=False)
        batches = format_ddl_batches(parsed_without_tags, batch_size=3)
        for i, batch in enumerate(batches, 1):
            print(f"\nBatch {i}:")
            print(batch)
            print("-" * 80)
            
    except Exception as e:
        print(f"Error parsing DDL: {str(e)}")

if __name__ == "__main__":
    main()
