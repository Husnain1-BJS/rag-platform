import json
import re
from typing import List, Dict, Any


def strip_markdown_links(text: str) -> str:
    """
    Remove markdown links from text, leaving only the link text.
    Example: [click here](http://example.com) -> click here
    """
    # Pattern to match [text](url)
    pattern = r'\[([^\]]*)\]\([^\)]*\)'
    return re.sub(pattern, r'\1', text)


def parse_mitre_attack(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse MITRE ATT&CK STIX 2.0 JSON and extract technique information.
    
    Args:
        file_path: Path to the enterprise-attack.json file
        
    Returns:
        List of dictionaries with keys: technique_id, name, description, 
        tactic, platforms, source, plus CVE-compatible fields:
        cve_id, severity, published_date
    """
    results = []
    
    # Load the STIX 2.0 JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get objects array (STIX 2.0 format)
    objects = data.get('objects', [])
    
    for obj in objects:
        # Filter for attack-pattern objects that are not deprecated or revoked
        if obj.get('type') != 'attack-pattern':
            continue
        if obj.get('x_mitre_deprecated') is True:
            continue
        if obj.get('revoked') is True:
            continue
            
        # Extract technique_id from external_references
        technique_id = None
        external_refs = obj.get('external_references', [])
        for ref in external_refs:
            if ref.get('source_name') == 'mitre-attack':
                technique_id = ref.get('external_id')
                break
        
        if not technique_id:
            continue  # Skip if we can't find the technique ID
            
        # Extract name
        name = obj.get('name', '')
        
        # Extract description and strip markdown links
        description = obj.get('description', '')
        description = strip_markdown_links(description)
        
        # Extract tactics from kill_chain_phases
        tactics = []
        kill_chain_phases = obj.get('kill_chain_phases', [])
        for phase in kill_chain_phases:
            if phase.get('kill_chain_name') == 'mitre-attack':
                tactics.append(phase.get('phase_name', ''))
        
        # Extract platforms
        platforms = obj.get('x_mitre_platforms', [])
        
        # Build result dictionary with CVE-compatible fields
        result = {
            # MITRE-specific fields
            'technique_id': technique_id,
            'name': name,
            'description': description,
            'tactic': tactics,
            'platforms': platforms,
            'source': 'mitre',
            # CVE-compatible fields (to work with existing chunker/embedder)
            'cve_id': technique_id,  # Use technique ID as CVE ID
            'severity': 'INFO',      # MITRE techniques are informational
            'published_date': '2024-01-01T00:00:00.000Z',  # Default date
            'references': [],        # MITRE parser doesn't extract references by default
        }
        results.append(result)
    
    return results


if __name__ == '__main__':
    # Example usage - parse the MITRE ATT&CK enterprise data
    # Note: You need to have the enterprise-attack.json file available
    import os
    import sys
    
    # Look for the file in common locations
    possible_paths = [
        'enterprise-attack.json',
        'datasets/raw/enterprise-attack.json',
        'data/enterprise-attack.json',
        '../enterprise-attack.json',
        '../../enterprise-attack.json'
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if not file_path:
        print("Error: Could not find enterprise-attack.json file")
        print("Please place the MITRE ATT&CK enterprise-attack.json file in one of:")
        for path in possible_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    print(f"Parsing MITRE ATT&CK data from: {file_path}")
    techniques = parse_mitre_attack(file_path)
    
    print(f"Found {len(techniques)} techniques")
    
    if techniques:
        # Print first technique as example
        example = techniques[0]
        print("\nFirst technique example:")
        print(json.dumps(example, indent=2))