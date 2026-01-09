import os

def generate_tree(startpath):
    # --------------------------------------------------------------------------
    # 🚫 IGNORE LIST: Add your virtual env folder name here (e.g., 'env', 'venv')
    # --------------------------------------------------------------------------
    ignore_dirs = {
        '.git', '__pycache__', '.vscode', '.idea', 'node_modules',
        'venv', 'env', '.venv', 'virtualenv',  # Virtual Environments
        'dist', 'build', '*.egg-info',         # Python build artifacts
        '__init__.py',                         # Optional: hide init files to save space
        'site-packages', 'lib', 'bin', 'include', 'share' # Library internals
        'archive_offline'
    }
    
    # Specific files to ignore
    ignore_files = {'.DS_Store', 'make_tree.py', '.gitignore', 'LICENSE', 'archive_offline'}

    print(f"📂 {os.path.basename(os.getcwd())}")
    
    for root, dirs, files in os.walk(startpath):
        # 1. REMOVE IGNORED DIRECTORIES IN-PLACE
        # This prevents the script from even entering those folders
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.endswith('.dist-info')]
        
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * level
        
        # Don't print the root folder name again
        if level > 0:
            print(f"{indent}├── {os.path.basename(root)}/")
            
        subindent = '│   ' * (level + 1)
        
        for f in files:
            if f not in ignore_files and not f.endswith('.pyc'):
                print(f"{subindent}├── {f}")

if __name__ == "__main__":
    generate_tree('.')