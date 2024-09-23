import os


def write_current_file_to_txt(script_path):
    """Write the contents of the provided script to a .txt file with the same name."""
    # Get the directory and the name of the provided script
    script_directory = os.path.dirname(os.path.abspath(script_path))
    script_name = os.path.splitext(os.path.basename(script_path))[
        0
    ]  # Get the filename without extension

    # Create the output .txt file path with the same name as the .py file
    output_file_path = os.path.join(script_directory, f"{script_name}.txt")

    # Open the provided Python file in read mode and write it to the .txt file
    with open(script_path, "r") as source_file:
        content = source_file.read()

    with open(output_file_path, "w") as output_file:
        output_file.write(content)
    print(f"Written {script_name}.txt to {script_directory}")
