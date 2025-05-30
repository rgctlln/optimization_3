import pandas

from BackTrace import BackTrace


class Model:
    def __init__(self):
        self.parameters: pandas.Series = pandas.Series({
            "const": 0,
        })
        self._education80Percent = False
        self.dataset_of_point = pandas.DataFrame({})

    def add_constant(self, predictor: pandas.Series):
        self.parameters[predictor.name] = 0
        self.dataset_of_point[predictor.name] = predictor

    def fit_OLS(self, dependent_variable: pandas.Series, use80forEducation=False, full_output=False,
                batch_size: int = 1) -> pandas.Series:
        self.dataset_of_point["DEPENDENT"] = dependent_variable
        self._education80Percent = use80forEducation

        dataset_for_education = self.dataset_of_point
        if self._education80Percent:
            dataset_for_education = dataset_for_education.iloc[:int(len(dataset_for_education) * 0.8)]

        self.parameters = (lambda bt: bt.set_full_output() if full_output else bt)(
            BackTrace(self.parameters, dataset_for_education, batch_size=batch_size).set_piecewise_constant(
                0.01)).start_back_trace()

        return self.parameters

    def predict(self, predictor: pandas.Series | list[pandas.Series] | pandas.DataFrame,
                predict_for_test=False) -> pandas.Series:
        const = self.parameters["const"]
        temp_parameters = self.parameters.drop(index=["const"])

        if type(predictor) == pandas.Series and not len(self.parameters) == 2:
            raise Exception("The number of parameters entered does not match the number of predictors entered")

        dataset = pandas.DataFrame(predictor).transpose() if type(predictor) != pandas.DataFrame else predictor

        if predict_for_test and self._education80Percent:
            dataset = dataset.iloc[int(len(dataset) * 0.8):]

        y_pred = dataset.dot(temp_parameters) + const
        return y_pred

    def cnt_mistakes(self) -> pandas.DataFrame:
        y_pred = self.predict(self.dataset_of_point.drop(columns="DEPENDENT"), True)
        y_real = self.dataset_of_point["DEPENDENT"]

        if self._education80Percent:
            y_real = y_real.iloc[int(len(y_real) * 0.8):]

        MSE = ((y_pred - y_real) ** 2).mean()
        MAE = (y_pred - y_real).abs().mean()

        metrics = {
            "MSE": [MSE],
            "MAE": [MAE]
        }

        df_metrics = pandas.DataFrame(metrics)
        return df_metrics

    def view_2d_graphic(self):
        if len(self.parameters) != 2:
            raise Exception("There should be only one predictor, otherwise nothing will be visible on the graph.")

        import matplotlib.pyplot as plt

        feature_name = next(col for col in self.dataset_of_point.columns if col != "DEPENDENT")

        X = self.dataset_of_point[feature_name]
        Y_real = self.dataset_of_point["DEPENDENT"]

        feature_name = next(col for col in self.dataset_of_point.columns if col != "DEPENDENT")

        line_X = self.dataset_of_point[feature_name].sort_values()
        line_Y = self.parameters["const"] + self.parameters[feature_name] * line_X

        # plt.figure(figsize=(8, 6))
        plt.scatter(X, Y_real, label='Точки (X, Y)', alpha=0.7)
        plt.plot(line_X, line_Y, color='red', label=f'Предсказанная модель', linewidth=2)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("График точек и линейной регрессии")
        plt.legend()
        plt.grid(True)
