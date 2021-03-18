import os 
from pathlib import Path
dir_path = os.path.dirname(os.path.realpath(__file__))

file_dir = os.path.join(dir_path, 'from')
small_dir = os.path.join(dir_path, 'small')
large_dir = os.path.join(dir_path, 'large')

check_size = 150000

for file in os.listdir(file_dir):
    if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
        if Path(os.path.join(file_dir, file)).stat().st_size <= check_size:
            # move to small
            os.rename(os.path.join(file_dir, file), os.path.join(small_dir, file))
        else:
            # move to large
            os.rename(os.path.join(file_dir, file), os.path.join(large_dir, file))