import os

def print_directory_structure(directory, prefix=''):
    """
    Prints the directory structure of the given directory.

    :param directory: str, path of the directory to traverse
    :param prefix: str, prefix for the current level (used for indentation)
    """
    try:
        # List all the entries in the directory
        entries = os.listdir(directory)
    except PermissionError:
        # If we don't have permissions to list directory contents, skip it
        return

    for entry in entries:
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            print(f"{prefix}{entry}/")
            print_directory_structure(path, prefix + '    ')
        else:
            print(f"{prefix}{entry}")

# Replace 'your_directory_path' with the path of the directory you want to traverse
directory_path = r'C:\Users\darla\OneDrive\Roba Vecchia_onedrive\Documenti\GitHub\GISEle\scripts'
print_directory_structure(directory_path)
