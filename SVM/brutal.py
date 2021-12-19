from SVMIO import fetch_database
import pandas as pd
from sklearn.svm import SVC


class SVM(object):
    def __init__(self, data_dirs=[], debug=False) -> None:
        labels, data = fetch_database(data_dirs, debug)
        self.DATA = pd.DataFrame(data, columns=labels)
        self.debug = debug
        self.learning = None
        self.verification = None
        if debug:
            print('Successfully created database object...')

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
            print(self.DATA['discriminator'].value_counts())

    def create_model(self, t_vs_v=4):
        self.DATA = self.DATA.sample(frac=1)    # shuffle rows
        slicing = len(self.DATA.values) // (t_vs_v + 1)
        # to verification upper, everything else to testing

        self.verification = self.DATA.loc[range(slicing, len(self.DATA.values))],
        self.learning = self.DATA.loc[range(0, slicing, 1)]

        if self.debug:
            print("Successfully build training & verification sets...")

    def prepare_data(self):
        print(self.DATA['quality'].value_counts().sort_index())
        choice = input(
            'Choose the separation point (0, S] + (S, INF).\n'
            + 'Mind, that the algorithm will do the best, if in both sets'
            + ' the amount were about equal:\n')

        self.update_class_separation(choice)

        self.create_model()
        m.verification = m.verification[0]


if __name__ == "__main__":
    m = SVM(['winequality-white.csv', 'winequality-red.csv'], True)
    m.prepare_data()

    kernels = ['linear', 'rbf']     # dimension transformation
    constants = [0.1, 1, 50]     # penalty
    gammas = [0.001, 0.1, 1]        # scouted coeff.

    for kernel in kernels:
        results = []
        for C in constants:
            rg = []
            for gamma in gammas:
                model = SVC(kernel=kernel, C=C, gamma=gamma)
                model.fit(
                    X=m.learning[m.learning.columns.difference(
                        ['quality', 'discriminator'])].values,
                    y=m.learning['discriminator'].values
                )
                predictions = model.predict(
                    m.verification[m.verification.columns.difference(
                        ['quality', 'discriminator'])].values
                )
                positive = sum([x == y for x, y in zip(predictions, m.verification['discriminator'].values)])
                rg.append(positive / len(predictions))
                print(f'Accuracy for {kernel} with C={C} and gamma={gamma}: {rg[-1]}')
            results.append(rg)
        print(f"Accuracy using kernel: {kernel}\n")
        results = pd.DataFrame(
            results,
            columns=[f'Gamma={g}' for g in gammas],
            index=[f'Const={c}' for c in constants]
        )
        print(results)
