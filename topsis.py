from numpy import sqrt
from pandas import DataFrame


class TOPSIS:
    def __init__(self, KPIsToAnalyze: list, benefit_attributes):
        self.KPIsToAnalyze = KPIsToAnalyze
        self.benefit_attributes = benefit_attributes

    def execute(self, data: DataFrame, weights: dict, scenario) -> int:
        # Wherever appears 'i' refers to a solution id (row)
        #                  'j' refers to an attribute  (column)

        r = DataFrame()
        v = DataFrame()
        v_plus = list()
        v_neg = list()
        for column in data[self.KPIsToAnalyze]:

            # Step 1. r_{ij} = x_{ij}/sqrt(sum_{i=1 to m} x_{ij}^2)
            r[column] = data[column].div(sqrt((data[column] ** 2).sum())).fillna(0).copy()

            # Step 2. v_{ij} = w_{j} r_{ij}
            v[column] = r[column].multiply(weights[column]).copy()

            # Step 3. V^+ = {v^+_j} and V^- = {v^-_j}
            max_val = v[column].max()
            min_val = v[column].min()
            if column in self.benefit_attributes:
                v_plus.append(max_val)
                v_neg.append(min_val)
            else:
                v_plus.append(min_val)
                v_neg.append(max_val)

        # Step 4. S^+ = {s_i^+}, S^- = {s_i^-}
        # Step 5. C = {s_i^-/(s_i^+ + s_i^-)}
        cols = v.columns
        idx = v.index.values
        c = list()
        for i in idx:
            s_i_plus = sqrt(sum([(v[c][i] - v_plus[j]) ** 2 for j, c in enumerate(cols)]))
            s_i_minus = sqrt(sum([(v[c][i] - v_neg[j]) ** 2 for j, c in enumerate(cols)]))
            c.append(s_i_minus / (s_i_plus + s_i_minus))

        # Step 6. Creating output dataframe sorted by rating column
        data_rated = data.copy()
        data_rated['RATING'] = c
        data_rated['SCENARIO'] = [scenario] * data_rated.shape[0]
        data_rated.sort_values(by=['RATING'], ascending=False, inplace=True)

        print('Best solution: {}'.format(data_rated.index[0] + 1))

        return data_rated.index[0]
