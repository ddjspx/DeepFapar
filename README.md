# DeepFapar

Neural Network (Keras/TensorFlow) based pipeline for estimating biophysical variables (fAPAR, LAI, etc.) from UAV-derived data.

## Project Structure

```
DeepFapar/
├── estimation.py              # Entry point
├── config/
│   ├── splits_params.py       # Data splitting & normalization config
│   └── NN_params.py           # Neural Network hyperparameters & prediction config
├── processing/
│   ├── splits_processing.py   # Dataset splitting & normalization
│   └── NN_processing.py       # NN training, testing, inference
└── utils/
    ├── metrics.py             # Error metrics (R2, RMSE, MAE, NASH, etc.)
    ├── example_dataset_train_val_test.xlsx   # Training/validation/test dataset
    └── example_dataset_predict.xlsx          # Prediction dataset
```

## Installation

Option 1: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate deepfapar
pip install .
```

Option 2: pip
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras openpyxl
```

## Configuration

### Data splitting (`config/splits_params.py`)

| Parameter | Description | Default |
|---|---|---|
| `dataset_path` | Path to training dataset Excel file | `utils/example_dataset_train_val_test.xlsx` |
| `features` | List of input feature column names | `['logpre','logvpd','chi','logppfd','gtmp','AI','loggpp','soc','logtp','logtn']` |
| `target` | Target variable column name | `'fapar'` |
| `trials` | Query conditions to filter data | `{'all_data': 'index == index'}` |
| `training_size` | Fraction of data for training | `0.7` |
| `test_size` | Fraction of data for testing | `0.15` |
| `normalize` | Whether to apply min-max normalization | `True` |
| `output_path` | Directory for saving split data | `output_splits` |

### Neural Network hyperparameters (`config/NN_params.py`)

| Parameter | Description | Default |
|---|---|---|
| `depth` | Number of hidden layers | `2` |
| `neurons` | Neurons per hidden layer (tuple) | `(64, 32)` |
| `activation` | Activation function | `'relu'` |
| `optimizer` | Optimizer for training | `'adam'` |
| `epochs` | Number of training epochs | `100` |

### Prediction config (`config/NN_params.py` - `NN_predict`)

| Parameter | Description | Default |
|---|---|---|
| `dataset` | Path to prediction dataset | `utils/example_dataset_predict.xlsx` |
| `features` | Input features (must match training) | same as above |
| `target` | Target variable name | `'fapar'` |
| `normalize` | Whether to normalize prediction data | `True` |

## Usage

```bash
cd DeepFapar
python estimation.py
```

## Pipeline

1. **Data Splitting** - Loads the training dataset, applies min-max normalization, splits into train/validation/test sets (70/15/15), and saves to `output_splits/`.
2. **NN Training** - Builds and trains a feedforward neural network, saves model checkpoints per epoch, selects the best model based on minimum validation loss, and saves training curves and metrics.
3. **NN Testing** - Evaluates the best model on the test set, generates scatter plot and error metrics.
4. **NN Prediction** - Loads the prediction dataset, applies normalization using training min/max values, runs inference, and saves results.

## Outputs

After running, the following outputs are generated:

- `output_splits/` - Normalized train/val/test splits and min-max values
- `trial1_NN/` - Model checkpoints (*.keras), best model, training loss/MAE curves, validation metrics
- `trial1_NN/test/` - Test scatter plot and metrics
- `trial1_NN/outputs_trial1_NN.xlsx` - Prediction results

## Model Selection

The pipeline automatically selects the epoch with the minimum validation loss as the best model. All epoch checkpoints are saved during training for further analysis.

## Dependencies

- Python >= 3.8, < 3.12
- numpy >= 1.23
- pandas >= 1.4
- matplotlib >= 3.5
- scikit-learn >= 1.1
- tensorflow >= 2.11
- keras >= 2.11
- openpyxl >= 3.0
