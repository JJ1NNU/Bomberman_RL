import sys
import os
import pickle
import numpy as np

def load(files, dir):
    matrices = []

    for file_name in files:
        print(file_name)

        with open(f"./agent_code/{dir}/mp/data/{file_name}", "rb") as file:
            matrices.append(pickle.load(file))

    print(f"loaded files: {len(matrices)}")
    return matrices

def merge(matrices):

    mats = len(matrices)
    cols = len(matrices[0])
    rows = len(matrices[0][0])

    print(f"mats {mats}")
    print(f"cols {cols}")
    print(f"rows {rows}")

    sum = np.zeros((cols, rows))
    for i in range(mats):
        sum += matrices[i]
    mean_matrix = sum / mats

    print(type(matrices))
    print(type(matrices[0]))
    print(type(matrices[0][0]))
    print(type(matrices[0][0][0]))
    
    return mean_matrix

def save(matrix, dir):
    with open(f"./agent_code/{dir}/my-saved-model.pt", "wb") as file:
        pickle.dump(matrix, file)

def list_files(directory):
    try:
        # list of files and dirs
        all_files_and_dirs = os.listdir(directory)

        # list all files
        files = [f for f in all_files_and_dirs if os.path.isfile(os.path.join(directory, f))]

        return files
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied to access '{directory}'.")
        return []

def main(argv):
    # parse argv
    dir = argv[0]

    # load q matrices
    matrices = load(list_files(f"./agent_code/{dir}/mp/data"), dir)

    # merge q matrices
    matrix = merge(matrices)

    # save merged q matrix
    save(matrix, dir)

if __name__ == '__main__':
    main(sys.argv[1:])
