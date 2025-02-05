def __init__(self):
    # Modified regex patterns to better capture all columns
    self.create_pattern = re.compile(
        r'''
        \s*create\s+or\s+replace\s+   # CREATE OR REPLACE with flexible spacing
        (TABLE|VIEW)\s+               # TABLE or VIEW
        ([^\s(]+)\s*                  # Object name
        \(                            # Opening parenthesis
        (.*?)                         # Column definitions (non-greedy) - EVERYTHING between parentheses
        \)                           # Closing parenthesis
        ''',
        re.DOTALL | re.IGNORECASE | re.VERBOSE
    )

def _split_columns(self, columns_text: str) -> List[str]:
    """Split columns while handling nested parentheses and comments"""
    columns = []
    current_col = []
    paren_count = 0
    in_comment = False
    
    # First clean up the input text
    lines = columns_text.strip().split('\n')
    cleaned_text = ''
    for line in lines:
        line = line.strip()
        if line:
            cleaned_text += line + '\n'
    
    # Now process character by character
    for i, char in enumerate(cleaned_text):
        # Handle comments
        if char == '-' and i > 0 and cleaned_text[i-1] == '-':
            in_comment = True
        elif char == '\n':
            in_comment = False
            
        # Handle parentheses
        if not in_comment:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
        
        # Split on comma only if we're not in a comment and not inside parentheses
        if char == ',' and paren_count == 0 and not in_comment:
            col_text = ''.join(current_col).strip()
            if col_text:  # Only add non-empty columns
                columns.append(col_text)
            current_col = []
        else:
            current_col.append(char)
    
    # Add the last column
    col_text = ''.join(current_col).strip()
    if col_text:
        columns.append(col_text)
    
    # Clean up columns and remove empty ones
    cleaned_columns = []
    for col in columns:
        # Split into lines to handle comments properly
        col_lines = col.split('\n')
        cleaned_col_lines = []
        for line in col_lines:
            line = line.strip()
            if line:
                cleaned_col_lines.append(line)
        if cleaned_col_lines:
            cleaned_columns.append('\n'.join(cleaned_col_lines))
    
    return [col for col in cleaned_columns if col.strip()]

def _parse_column(self, column_text: str, include_tags: bool) -> Optional[Dict]:
    """Parse individual column definition with improved handling"""
    lines = column_text.split('\n')
    col_def = lines[0].strip()
    samples = None
    
    # Look for dummy samples in all lines
    for line in lines:
        if 'Dummy Sample' in line:
            samples = line.split(':', 1)[1].strip() if ':' in line else line.split('Dummy Sample', 1)[1].strip()
            break
    
    # Updated column pattern to be more flexible
    col_pattern = re.compile(
        r'''
        ^\s*
        (\w+)                                    # Column name
        (?:\s+([^\s,]+(?:\([^)]+\))?)?)?        # Optional datatype
        (?:\s+WITH\s+TAG\s+(\([^)]+\)))?        # Optional TAG clause
        \s*$
        ''',
        re.VERBOSE | re.IGNORECASE
    )
    
    match = col_pattern.match(col_def)
    if not match:
        # If no match, try just getting the column name
        simple_name_match = re.match(r'^\s*(\w+)\s*$', col_def)
        if simple_name_match:
            return {
                'name': simple_name_match.group(1),
                'samples': samples if samples else None
            }
        return None
        
    col_name, datatype, tag = match.groups()
    
    # Build column info
    column_info = {
        'name': f"{col_name} WITH TAG {tag}" if include_tags and tag else col_name,
        'datatype': datatype if datatype else None,
    }
    
    if samples:
        try:
            if samples.startswith('{'):
                column_info['samples'] = samples
            else:
                column_info['samples'] = [s.strip() for s in samples.split(',')]
        except:
            column_info['samples'] = samples
            
    if tag and include_tags:
        column_info['tag'] = tag
            
    return column_info
