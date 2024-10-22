"""
batch_py2txt.py

This script automates the process of converting Python (.py) files to text (.txt) files for easier sharing and readability.
It traverses through a specified directory, identifies all `.py` files, and converts them to `.txt` format, maintaining
the original structure and content.

Functions:
    - convert_py_to_txt(py_file: str, txt_file: str): Converts a single Python file to a text file.
    - batch_convert_py_to_txt(directory: str): Converts all Python files in the given directory to text format.

Example Usage:
    # To convert a single Python file to text:
    convert_py_to_txt('example.py', 'example.txt')
    
    # To batch convert all Python files in a directory:
    batch_convert_py_to_txt('/path/to/directory')
    
    # Run from the command line:
    python batch_py2txt.py /path/to/directory

Requirements:
    Python 3.x
"""

# %%
import os


# %%
# Function to convert .py files to .txt files
def batch_py2txt(directory):
    # Create a folder called 'txt' if it doesn't exist
    txt_folder = os.path.join(directory, "txt")
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    # Loop over all files in the current directory
    for file_name in os.listdir(directory):
        # Only process files, not folders
        if os.path.isfile(
            os.path.join(directory, file_name)
        ) and file_name.endswith(".py"):
            # Define the input and output file paths
            py_file_path = os.path.join(directory, file_name)
            txt_file_name = file_name.replace(".py", ".txt")
            txt_file_path = os.path.join(txt_folder, txt_file_name)

            # Read content from the .py file and write to the .txt file
            with open(py_file_path, "r", encoding="utf-8") as py_file:
                content = py_file.read()

            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(content)

            print(
                f"Converted {file_name} to {txt_file_name} and saved in 'txt' folder."
            )


# %%
# Execution starts here
if __name__ == "__main__":
    # Get the current directory where the script is executed
    current_directory = os.getcwd()
    batch_py2txt(current_directory)
# %%
