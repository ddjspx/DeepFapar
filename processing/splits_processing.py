import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DatasetProcessor:
    def __init__(self, splits_params):
        """
        Initialize the DatasetProcessor using a dictionary of parameters.
        :param splits_params: Dictionary containing all configuration parameters for data processing.
        """
        # Extract parameters from splits_params
        self.dataset_path = splits_params["dataset_path"]
        self.features = splits_params["features"]
        self.target = splits_params["target"]
        self.trials = splits_params["trials"]
        self.training_size = splits_params.get("training_size", 0.7)
        self.test_size = splits_params.get("test_size", 0.15)
        self.normalize = splits_params.get("normalize", True)
        self.output_path = splits_params.get("output_path", "splits")

        # Load dataset
        self.data = pd.read_excel(self.dataset_path)
        self.splits = None
        self.min_max_values = None

    def _normalize_difference(self, df, columns, min_values, max_values):
        """
        Apply min-max normalization to specified columns.
        :param df: DataFrame to normalize.
        :param columns: Columns to normalize.
        :param min_values: Series of minimum values for each column.
        :param max_values: Series of maximum values for each column.
        :return: Normalized DataFrame.
        """
        normalized_df = df.copy()
        for col in columns:
            xmin = min_values[col]
            xmax = max_values[col]
            normalized_df[col] = (df[col] - xmin) / (xmax - xmin)
        return normalized_df

    def normalize_splits(self):
        """
        Splits the dataset into train, validation, and test sets for each trial.
        Normalizes feature values if required.
        """
        # Filter data by trial conditions
        trials_data = {name: self.data.query(condition) for name, condition in self.trials.items()}

        if self.normalize:
            # Compute min and max values for normalization
            min_values = self.data[self.features].min()
            max_values = self.data[self.features].max()
            self.min_max_values = pd.DataFrame({
                'variable': min_values.index,
                'min': min_values.values,
                'max': max_values.values
            })
            # Normalize data per trial
            trials_data = {name: self._normalize_difference(df[self.features], self.features, min_values, max_values)
                           for name, df in trials_data.items()}
        else:
            trials_data = {name: df[self.features] for name, df in trials_data.items()}

        # Split data into train, validation, and test
        splits = {}
        for trial_name, trial_X in trials_data.items():
            trial_y = self.data.query(self.trials[trial_name])[self.target].to_numpy()
            X_train, X_test_val, y_train, y_test_val = train_test_split(
                trial_X, trial_y, train_size=self.training_size
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_test_val, y_test_val, train_size=self.test_size / (1 - self.training_size)
            )
            splits[trial_name] = {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test),
            }

        self.splits = splits

    def save_splits(self):
        """
        Save train, validation, and test splits to Excel files.
        Also returns the saved splits and normalization parameters.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)

        # Initialize variables to store combined data for return
        combined_X = {'train': [], 'val': [], 'test': []}
        combined_y = {'train': [], 'val': [], 'test': []}

        # Combine splits for all trials
        for split_name in ['train', 'val', 'test']:
            # Initialize temporary lists to store split data for this iteration
            temp_X = []
            temp_y = []

            for trial in self.splits:
                trial_X, trial_y = self.splits[trial][split_name]
                temp_X.append(trial_X)
                temp_y.append(trial_y)

            # Now combine and save the data for this split
            X_combined = np.vstack(temp_X)
            y_combined = np.hstack(temp_y)

            X_df = pd.DataFrame(X_combined, columns=self.features)
            y_df = pd.DataFrame(y_combined, columns=[self.target])

            # Save to Excel
            X_df.to_excel(f'{self.output_path}/X_{split_name}.xlsx', index=False)
            y_df.to_excel(f'{self.output_path}/y_{split_name}.xlsx', index=False)

            # Store combined data for return
            combined_X[split_name] = X_combined
            combined_y[split_name] = y_combined

        # Save min-max normalization values if normalization is applied
        if self.normalize:
            self.min_max_values.to_excel(f'{self.output_path}/min_max_values.xlsx', index=False)

        # Return combined splits and normalization parameters
        return combined_X, combined_y, self.min_max_values
