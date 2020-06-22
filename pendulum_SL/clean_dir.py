import os

directory = "./plots"

files_in_dir = os.listdir(directory)
filtered_files = [file for file in files_in_dir if file.endswith(".png")]
for file in filtered_files:
    path_to_file = os.path.join(directory, file)
    os.remove(path_to_file)

print("Directory Cleaned!")
