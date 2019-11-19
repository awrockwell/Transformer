import pandas as pd
import scipy.stats

class funcHolding:
    def __init__(self):
        pass


class BoxCox(funcHolding):
    starters = [-5, .0001, 5]
    rotations = 15

    def equation(self, Xs, Lambda):
        return (Xs ** Lambda - 1) / Lambda


class Inverse(funcHolding):
    starters = [-10000, 10000, 0]
    rotations = 15

    def equation(self, Xs, Lambda):
        return 1 / (Xs + Lambda)


class Normalize01(funcHolding):
    starters = []
    rotations = 1

    def equation(self, Xs, Lambda):
        return (Xs - Xs.min()) / (Xs.max() - Xs.min())


class NormalizeStdDev(funcHolding):
    starters = []
    rotations = 1

    def equation(self, Xs, Lambda):
        return (Xs - Xs.mean()) / Xs.std()

class NormalDistCDF(funcHolding):
    starters = []
    rotations = 1

    def equation(self, Xs, Lambda):
        columnSave = Xs.columns
        npArray = scipy.stats.norm(Xs.mean(), Xs.std()).cdf(Xs)
        return pd.DataFrame(data=npArray[0:, 0:], columns=columnSave)



