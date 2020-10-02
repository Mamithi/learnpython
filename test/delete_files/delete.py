import os
import shutil
import time

def main():
    path = "streams"

    days = 1

    seconds = time.time() - (days * 1 * 60 * 60)
    print(days * 1 * 60 * 60)

    if os.path.exists(path):
        for root_folder, folders, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root_folder, file)

                if seconds > get_file_age(file_path):
                    remove_file(file_path)

def get_file_age(path):
    ctime = os.stat(path).st_ctime
    return ctime

def remove_file(path):
    if not os.remove(path):
        print(f"{path} is removed successfully")
    else:
        print(f"Unable to delete the {path}")


if __name__=='__main__':
    main()