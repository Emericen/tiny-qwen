#!/usr/bin/env python3
"""
Script to convert role names in LLaVA JSONL files from:
- "human" -> "user" 
- "gpt" -> "assistant"
"""

import json
import os
from pathlib import Path

def convert_roles_in_file(input_file, output_file=None):
    """Convert roles in a single JSONL file."""
    if output_file is None:
        output_file = input_file
    
    converted_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Convert roles in messages
                if 'messages' in data:
                    for message in data['messages']:
                        if message.get('role') == 'human':
                            message['role'] = 'user'
                        elif message.get('role') == 'gpt':
                            message['role'] = 'assistant'
                
                converted_lines.append(json.dumps(data, ensure_ascii=False))
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {input_file}: {e}")
                continue
    
    # Write converted data back
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in converted_lines:
            f.write(line + '\n')
    
    print(f"Converted {len(converted_lines)} lines in {input_file}")

def main():
    """Convert all JSONL files in the LLaVA-Instruct-150K directory."""
    data_dir = Path("data/LLaVA-Instruct-150K")
    
    if not data_dir.exists():
        print(f"Directory {data_dir} not found!")
        return
    
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print("No JSONL files found in the directory!")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to convert:")
    for file in jsonl_files:
        print(f"  - {file.name}")
    
    for jsonl_file in jsonl_files:
        print(f"\nProcessing {jsonl_file.name}...")
        convert_roles_in_file(jsonl_file)
    
    print("\nRole conversion completed!")

if __name__ == "__main__":
    main()