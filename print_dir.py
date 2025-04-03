#!/usr/bin/env python3
"""
Directory Tree Printer for macOS
Usage: python3 tree.py [directory_path] [max_depth]
If no directory path is provided, current directory is used.
If no max_depth is provided, it will print the entire tree.
"""

import os
import sys
from pathlib import Path

def print_tree(directory, prefix="", max_depth=None, current_depth=0):
    """
    Recursively print the directory tree structure.
    
    Args:
        directory (Path): The directory to print
        prefix (str): Prefix for the current line (used for indentation)
        max_depth (int, optional): Maximum depth to traverse
        current_depth (int): Current depth in the tree
    """
    # Check if we've reached max depth
    if max_depth is not None and current_depth > max_depth:
        return
    
    # Get the directory name
    directory = Path(directory)
    print(f"{prefix}├── {directory.name}/")
    
    # Sort entries: directories first, then files
    try:
        entries = list(directory.iterdir())
        dirs = sorted([e for e in entries if e.is_dir()], key=lambda x: x.name.lower())
        files = sorted([e for e in entries if e.is_file()], key=lambda x: x.name.lower())
        sorted_entries = dirs + files
    except PermissionError:
        print(f"{prefix}│   ├── [Permission Denied]")
        return
    
    # Process all items except the last one
    num_entries = len(sorted_entries)
    for i, entry in enumerate(sorted_entries):
        # Skip hidden files/directories that start with '.'
        if entry.name.startswith('.'):
            continue
            
        # Determine if this is the last item
        is_last = i == num_entries - 1
        
        # Choose the appropriate prefix for child items
        if is_last:
            child_prefix = prefix + "    "
            entry_prefix = prefix + "└── "
        else:
            child_prefix = prefix + "│   "
            entry_prefix = prefix + "├── "
        
        # Print and recurse if directory
        if entry.is_dir():
            print(f"{entry_prefix}{entry.name}/")
            print_tree(entry, child_prefix, max_depth, current_depth + 1)
        else:
            # Print file with size
            size = entry.stat().st_size
            size_str = format_size(size)
            print(f"{entry_prefix}{entry.name} ({size_str})")

def format_size(size_bytes):
    """Format file size in a human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def main():
    # Get directory path from command line or use current directory
    if len(sys.argv) >= 2:
        directory = sys.argv[1]
    else:
        directory = os.getcwd()
    
    # Get max depth if provided
    max_depth = None
    if len(sys.argv) >= 3:
        try:
            max_depth = int(sys.argv[2])
        except ValueError:
            print("Error: max_depth must be an integer")
            sys.exit(1)
    
    # Convert to Path object and ensure it exists
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    # Print the directory name first
    print(f"{directory_path}")
    
    # Start printing the tree
    for entry in sorted(directory_path.iterdir()):
        if entry.name.startswith('.'):
            continue
            
        if entry.is_dir():
            print_tree(entry, "", max_depth)
        else:
            size = entry.stat().st_size
            size_str = format_size(size)
            print(f"├── {entry.name} ({size_str})")

if __name__ == "__main__":
    main()