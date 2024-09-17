import os

def print_directory_structure(root_dir, indent=0):
  # Print the current directory
  print(' ' * indent + os.path.basename(root_dir) + '/')
  
  # Iterate through the items in the directory
  for item in os.listdir(root_dir):
      item_path = os.path.join(root_dir, item)
      if os.path.isdir(item_path):
          # If the item is a directory, recursively print its structure
          print_directory_structure(item_path, indent + 4)
      else:
          # If the item is a file, print its name
          print(' ' * (indent + 4) + item)

if __name__ == "__main__":
  # Specify the root directory you want to print
  root_directory = '.'  # Change this to your desired directory
  print_directory_structure(root_directory)