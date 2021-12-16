import csv
import numpy as np


def fetch_database(dirs, debug=False):
    Database = []
    for dir in dirs:
        with open(dir, newline='') as handle:
            reader = csv.reader(handle, delimiter=';', quotechar='"')
            labels = reader.__next__()
            for row in reader:
                Database.append(row)
        if debug:
            print(f'fetched file {dir}')

    return labels, np.array(Database)


if __name__ == "__main__":
    fetch_database(['winequality-red.csv', 'winequality-white.csv'], True)
