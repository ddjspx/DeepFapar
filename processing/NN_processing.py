import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
from utils.metrics import error_metrics

class NNProcessor:
    def __init__(self, name, params, predict_config=None):
        """
        Initializes the Neural Network Processor with necessary configurations.

        Parameters:
        - name : str : The name of the model trial.
        - params : dict : Dictionary containing parameters such as depth, neurons, activation, optimizer, epochs, etc.
        - predict_config : dict : Optional dictionary for prediction configurations (feature selection, normalization, etc.)
        """
        self.name = name
        self.depth = params.get('depth', 1)
        self.neurons = params.get('neurons', (64,))
        self.activation = params.get('activation', 'relu')
        self.optimizer = params.get('optimizer', 'adam')
        self.epochs = params.get('epochs', 100)
        self.predict_config = predict_config if predict_config else {}
        self.original_dir = os.getcwd()

        os.makedirs(self.name, exist_ok=True)
        os.chdir(self.name)

    def train(self, X_train, y_train, X_val, y_val, print=True):
        """Train the neural network."""
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        if self.depth == 1:
            model.add(Dense(units=self.neurons[0], activation=self.activation))
        else:
            for i in range(self.depth):
                model.add(Dense(units=self.neurons[i], activation=self.activation))
        model.add(Dense(units=1))
        model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])

        checkpoint_callback = ModelCheckpoint(filepath="epoch_{epoch:02d}.keras", save_freq="epoch",
                                              save_weights_only=False, verbose=1)

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.epochs,
                            callbacks=[checkpoint_callback])

        # Save metrics and hyperparameters
        metrics_df = pd.DataFrame({
            'Epoch': range(1, self.epochs + 1),
            'Training Loss': history.history['loss'],
            'Validation Loss': history.history['val_loss'],
            'Training MAE': history.history['mae'],
            'Validation MAE': history.history['val_mae']
        })

        hyperparams_df = pd.DataFrame({
            'Parameter': ['Depth', 'Neurons', 'Activation', 'Optimizer'],
            'Value': [self.depth, self.neurons, self.activation, self.optimizer]
        })

        if print:
            # Plot loss
            plt.plot(range(0, self.epochs), history.history['loss'], '-', label='training loss')
            plt.plot(range(0, self.epochs), history.history['val_loss'], label='validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{self.name}_loss_LAI_NN.png')
            plt.show()

            # Plot MAE
            plt.plot(range(0, self.epochs), history.history['mae'], 'y', label='Training MAE')
            plt.plot(range(0, self.epochs), history.history['val_mae'], 'r', label='Validation MAE')
            plt.title('Training and validation MAE')
            plt.xlabel('Epochs')
            plt.ylabel('Mean Absolute Error (MAE)')
            plt.legend()
            plt.savefig(f'{self.name}_mae_LAI_NN.png')
            plt.show()

        # Find best model (min validation loss)
        val_loss = history.history['val_loss']
        min_val_epoch = val_loss.index(min(val_loss)) + 1
        model_min_val_epoch = load_model(f"epoch_{min_val_epoch:02d}.keras")

        predictions_test = model_min_val_epoch.predict(X_val)
        r, r2, mse, rmse, rrmse, mae, nash = error_metrics(y_val, predictions_test)

        # Save final metrics
        metrics_final_model = pd.DataFrame({
            'Metric': ['r', 'r^2', 'MSE', 'RMSE', 'RRMSE', 'MAE', 'Nash-Sutcliffe', 'Epoch'],
            'Value': [r, r2, mse, rmse, rrmse, mae, nash, min_val_epoch]
        })

        with pd.ExcelWriter(f"{self.name}_validation_metrics.xlsx") as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            hyperparams_df.to_excel(writer, sheet_name='Hyperparameters', index=False)
            metrics_final_model.to_excel(writer, sheet_name='Metrics_Final', index=False)

        model_min_val_epoch.save(f"{self.name}_model.keras")
        return model_min_val_epoch

    def test(self, X_test, y_test, model):
        """Test the trained model."""

        os.makedirs(f'test', exist_ok=True)
        os.chdir(f'test')

        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
        elif isinstance(X_test, pd.DataFrame):
            pass
        else:
            raise TypeError("X_test must be an ndarray or DataFrame.")

        predictions_test = model.predict(X_test)

        r, r2, mse, rmse, rrmse, mae, nash = error_metrics(y_test, predictions_test)
        print(f"Testing NN RMSE: {rmse}")

        textstr = '\n'.join((r'$R^2=%.2f$' % (r2,),
                             r'$RMSE=%.2f$' % (rmse,),
                             r'$RRMSE=%.2f\%%$' % (rrmse,),
                             r'$MAE=%.2f$' % (mae,),
                             r'$NASH=%.2f$' % (nash,)))

        plt.figure(figsize=(8, 6))
        plt.scatter(predictions_test, y_test)
        plt.plot([predictions_test.min(), predictions_test.max()], [predictions_test.min(), predictions_test.max()],
                 '--', color='gray')

        m, b = np.polyfit(predictions_test.flatten(), y_test, 1)
        plt.plot(predictions_test, m * predictions_test + b, color='grey', lw=2)

        plt.xlabel('Predicted Value NN', fontsize=16)
        plt.ylabel('Measured Value', fontsize=16)
        plt.legend()
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=16, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5))
        plt.tight_layout()
        plt.savefig(f'{self.name}_inference_nn.png')
        plt.show()

    def predict(self, model, min_max_values):

        os.chdir(self.original_dir)
        dataset = pd.read_excel(self.predict_config['dataset'])
        normalize = self.predict_config['normalize']
        features = self.predict_config['features']
        target = self.predict_config['target']

        dataset_inputs = dataset[features]

        if normalize and min_max_values is not None:
            def normalize_difference(df, columns, min_values=None, max_values=None):
                normalized_df = df.copy()
                for col in columns:
                    xmin = min_values[col] if min_values is not None else df[col].min()
                    xmax = max_values[col] if max_values is not None else df[col].max()
                    normalized_df[col] = (df[col] - xmin) / (xmax - xmin)
                return normalized_df

            min_values = min_max_values.set_index("variable")["min"].to_dict()
            max_values = min_max_values.set_index("variable")["max"].to_dict()
            inputs_array = normalize_difference(dataset_inputs, features, min_values=min_values, max_values=max_values)
        else:
            inputs_array = dataset_inputs.values

        start_time = time.time()
        predictions = model.predict(inputs_array)
        elapsed_time = time.time() - start_time
        print(f"Time taken for prediction: {elapsed_time} seconds")

        dataset_inputs[target] = predictions
        dataset[target] = predictions

        path_save = rf'{self.original_dir}/{self.name}'
        dataset.to_excel(f'{path_save}/outputs_{self.name}.xlsx', index=False)

        return dataset
