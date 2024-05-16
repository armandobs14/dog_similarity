import shutil
import random
import glob
import os
import sys


def get_file_names(folder):
    return [f.split("/")[-1] for f in glob.glob(folder + "/*")]


def copy_files(source_folder, files_list, des):
    for file in files_list:
        source_file = os.path.join(source_folder, file)

        des_file = os.path.join(des, file)

        print(source_file, des_file)
        shutil.copy2(source_file, des_file)

        print(f"Copied {file} to {des}")
    return


def move_files(source_folder, des):
    files_list = os.listdir(source_folder)
    for file in files_list:
        source_file = os.path.join(source_folder, file)

        des_file = os.path.join(des, file)
        shutil.move(source_file, des_file)

        print(f"Copied {file} to {des}")
    return


def rename_file(file_path, new_name):
    directory = os.path.dirname(file_path)
    new_file_path = os.path.join(directory, new_name)

    os.rename(file_path, new_file_path)
    print(f"File renamed to {new_file_path}")
    return


BASE_PATH = os.getenv("BASE_PATH", ".")
folder_path = f"{BASE_PATH}/Images/*"
op_path_similar = f"{BASE_PATH}/similar_all_images"
op_path_dissimilar = f"{BASE_PATH}/dissimilar_all_images"
tmp = f"{BASE_PATH}/tmp"


folders_list = glob.glob(folder_path)
folders_list = list(
    set(folders_list).difference(
        set(
            [
                f"{BASE_PATH}/dissimilar_all_images",
                f"{BASE_PATH}/similar_all_images",
                f"{BASE_PATH}/tmp",
            ]
        )
    )
)

l, g = 0, 0

random.shuffle(folders_list)
for i in glob.glob(folder_path):
    if i in [
        f"{BASE_PATH}/dissimilar_all_images",
        f"{BASE_PATH}/similar_all_images",
        f"{BASE_PATH}/tmp",
    ]:
        continue

    file_name = i.split("\\")[-1].split("-")[1]
    picked_files = random.sample(get_file_names(i), 6)

    copy_files(i, picked_files, tmp)

    for m in range(3):
        rename_file(
            os.path.join(tmp, picked_files[m * 2]), "similar_" + str(g) + "_first.jpg"
        )
        rename_file(
            os.path.join(tmp, picked_files[m * 2 + 1]),
            "similar_" + str(g) + "_second.jpg",
        )
        g += 1
    move_files(tmp, op_path_similar)
    choice_one, choice_two = random.choice(range(len(folders_list))), random.choice(
        range(len(folders_list))
    )

    picked_dissimilar_one = random.sample(get_file_names(folders_list[choice_one]), 3)
    picked_dissimilar_two = random.sample(get_file_names(folders_list[choice_two]), 3)

    copy_files(folders_list[choice_one], picked_dissimilar_one, tmp)
    copy_files(folders_list[choice_two], picked_dissimilar_two, tmp)
    picked_files_dissimilar = picked_dissimilar_one + picked_dissimilar_two

    for m in range(3):
        rename_file(
            os.path.join(tmp, picked_files_dissimilar[m]),
            "dissimilar_" + str(l) + "_first.jpg",
        )
        rename_file(
            os.path.join(tmp, picked_files_dissimilar[m + 3]),
            "dissimilar_" + str(l) + "_second.jpg",
        )
        l += 1
    move_files(tmp, op_path_dissimilar)
