import random

import pandas

from BackTrace import BackTrace


class Model:
    def __init__(self, dataset: pandas.DataFrame):
        self.dataset = dataset
        self.parameters: pandas.Series = pandas.Series({
            "const": random.Random().randint(0, 100),
        })
        self.dataset_of_point = pandas.DataFrame({})

    def add_constant(self, predictor: pandas.Series):
        self.parameters[predictor.name] = random.Random().randint(0, 100)
        self.dataset_of_point[predictor.name] = predictor
        return self.dataset_of_point

    def fit_OLS(self, dependent_variable: pandas.Series):
        self.dataset_of_point["DEPENDENT"] = dependent_variable

        #TODO отдавать 80%

        self.parameters = BackTrace(self.parameters, self.dataset_of_point).set_exponential_decay(0.01,1).set_full_output().start_back_trace()
        return self.parameters
