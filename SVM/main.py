# Dlaczego po prostu nie zrobić
# import sklearn.svm?

from SVMIO import fetch_database
import pandas as pd


class model(object):
    def __init__(self, train: pd.DataFrame, verify: pd.DataFrame) -> None:
        self.verification = verify
        self.training = train


class SVM(object):
    def __init__(self, data_dirs=[], debug=False) -> None:
        labels, data = fetch_database(data_dirs, debug)
        self.DATA = pd.DataFrame(data, columns=labels)
        self.debug = debug
        if debug:
            print('Successfully created database object...')
            print(self.DATA)

    def update_class_separation(self, point):
        def fun(row):
            if row[11] > point:
                return 1            # wine is good
            else:
                return -1           # wine is bad
        self.DATA['discriminator'] = [fun(row) for row in self.DATA.values]
        if self.debug:
            print('Successfully qualified data...')
            print(self.DATA)

    def create_model(self, t_vs_v=4):
        self.DATA = self.DATA.sample(frac=1)    # shuffle rows
        slicing = len(self.DATA.values) // (t_vs_v + 1)
        # to verification upper, everything else to testing

        md = model(
            self.DATA.loc[range(slicing, len(self.DATA.values), 1)],
            self.DATA.loc[range(0, slicing, 1)]
        )

        if self.debug:
            print("Successfully build training & verification sets...")

        return md


if __name__ == "__main__":
    m = SVM(['winequality-red.csv'], True)
    print(m.DATA['quality'].value_counts().sort_index())
    choice = input(
        'Choose the separation point (0, S] + (S, INF).\n'
        + 'Mind, that the algorithm will do the best, if in both sets'
        + ' the amount were about equal:\n')

    m.update_class_separation(choice)

    print("Successfully classified the database...\n")
    print(m.DATA['discriminator'].value_counts())

    m.create_model()
