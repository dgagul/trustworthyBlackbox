import csv
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import skewnorm
import sklearn


class Generator(object):
    def __init__(self):
        self.ranges = []

    def generate_dataset_must(self, f_names, d_type, path, min_, max_, unique):
        """ Global function used for MUST dataset generation.

        Args:
            :param f_names: [String] names of the features
            :param d_type: [List:Strings] data_types of the features
            :param path: [bool] used for file naming
            :param min_, max_, unique:  [List] of size n

        Returns:
            :return         [DataFrame] synthetic dataset
        """
        self.ranges = []
        self._specify_ranges_must(d_type, min_, max_, unique)
        print("Specified ranges by for MUST scenario:")
        print(self.ranges)
        return self._generate_dataset(path, f_names, True)

    def generate_dataset_may(self, f_names, d_type, path, min_, max_, unique, mean, std, skew):
        """ Global function used for MAY dataset generation.

        Args:
            :param f_names: [String] names of the features
            :param d_type: [List:Strings] data_types of the features
            :param path: [bool] used for file naming
            :param min_, max_, unique:  [List] of size n
            :param mean, std, skew:  [List:float] of size n
                            NaN for categorical features

        Returns:
            :return         [DataFrame] synthetic dataset
        """
        self.ranges = []
        self._specify_ranges_may(d_type, min_, max_, unique, mean, std, skew)
        print("Specified ranges by for MAY scenario:")
        print(self.ranges)
        return self._generate_dataset(path, f_names, False)

    def _specify_ranges_must(self, d_type, min_, max_, unique):
        """ Iterates over the number of features (n) and
        specifies a [List] for each feature containing
        possible values for the later dataset generation.
        _private method, called by the generate_dataset____() function

        Args:
            :param d_type:  [List:Strings] of size n
            :param min_, max_, unique:  [List] of size n

        Sets:
            self.ranges:  [NestedList] attribute of the
            Generator object with n Lists of different size
        """
        for idx, _ in enumerate(d_type):
            if d_type[idx].__eq__("categorical"):
                # ordinal or nominal
                self.ranges.append(unique[idx])

            # discrete (finite options within a defined range)
            if d_type[idx].__eq__("discrete"):
                self.ranges.append(range(int(min_[idx]), int(max_[idx]) + 1, np.maximum(1, int(max_[idx] - min_[idx]) // 10)))

            # continuous (infinite options)
            if d_type[idx].__eq__("continuous"):
                self.ranges.append(np.arange(int(min_[idx]), int(max_[idx]) + 1, int(max_[idx] - min_[idx]) / 10))

    def _specify_ranges_may(self, d_type, min_, max_, unique, mean, std, skew):
        """ Similar to _specify_ranges_must() except that it
        takes additional properties as input, used for a variation
        in the range specification for discrete and continuous
        features (described in the report)

        Args:
            :param d_type:  [List:Strings] of size n
            :param min_, max_, unique:  [List] of size n
            :param mean, std, skew:  [List:float] of size n
                            NaN for categorical features

        Sets:
            self.ranges:  [NestedList] attribute of the
            Generator object with n Lists of different size
        """
        for idx, _ in enumerate(d_type):
            if d_type[idx].__eq__("categorical"):
                # ordinal or nominal
                self.ranges.append(unique[idx])
                continue

            # draw from distribution with statistical properties
            X = skewnorm(skew[idx], loc=mean[idx], scale=std[idx]).rvs(10000)
            # replace drawn samples out of bound
            X[X < min_[idx]] = min_[idx]
            X[X > max_[idx]] = max_[idx]
            X.sort()
            c_prop = np.arange(1, len(X) + 1) / len(X)
            r = []


            for threshold in np.arange(0, 1.1, 1 / 9):
                i = (np.abs(c_prop - threshold)).argmin()
                # discrete (finite options)
                if d_type[idx].__eq__("discrete"):
                    r.append(int(X[i]))
                # continuous (infinite options)
                elif d_type[idx].__eq__("continuous"):
                    r.append(X[i])
                else:
                    raise AttributeError("Unsupported data type provided.")

            r = list(set(r))
            r.sort()
            if int(min_[idx]) not in r: r.append(int(min_[idx]))
            if int(max_[idx]) not in r: r.append(int(max_[idx]))
            self.ranges.append(r)



    def _generate_dataset(self, path, f_names, must):
        """ Generates a dataset and saves it to a CSV
            file at the provided path destination.
            This function uses the ranges saved in the
            self.ranges attribute.

        Requires:
            Before calling this function, the ranges
            have to be specified by calling the
            _specify_ranges() function.

        Args:
            :param path:    [String] path location where
                             the file will be stored
            :param f_names: [List:Strings] names of the
                            features in a list for
            :param must:    [bool] used for file naming

        Returns:
            :return         [DataFrame] synthetic dataset
        """
        mustormay = "must" if must is True else "may"
        full_filename = f"{path}/synDS_{mustormay}.csv"
        with open(full_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(f_names)
            # Write all possible combinations to the CSV file
            for element in itertools.product(*self.ranges):
                writer.writerow(element)
        return pd.read_csv(full_filename)

    def label_synDS(self, model, dataset, categorical, target_name):
        """ labels the provided dataset with the specified model
        label is added at last column of the dataset

        Args:
            :param model: ML model (sklearn.RF, Keras sequential supported)
            :param dataset: [DataFrame] synthetic dataset
            :param categorical: [List:Strings] names of the categorical columns
            :param target_name: [String] name of the column with the labels

        Returns:
            :return [DataFrame] labeled synthetic dataset
        """
        for cat in categorical:
            # Encode the categorical data using sequential numbers
            feature = list(dataset[cat].unique())
            dataset[cat].replace(feature, [i for i, _ in enumerate(feature)], inplace=True)

        if isinstance(model, sklearn.ensemble.RandomForestClassifier):
            np_data = np.array(dataset)
            y_pred = model.predict(np_data)
        elif isinstance(model, tf.keras.Sequential):
            y_pred = model.predict(dataset)
            y_pred = y_pred.argmax(axis=1)
        else:
            raise Exception("Model not supported for labeling")

        dataset[target_name] = list(y_pred)
        return dataset