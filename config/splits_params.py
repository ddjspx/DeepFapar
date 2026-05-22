
# splits_processing

splits_params = {
    "dataset_path": "utils/example_dataset_train_val_test.xlsx",
    "features": ['logpre','logvpd',	'chi', 'logppfd', 'gtmp', 'AI', 'loggpp', 'soc', 'logtp','logtn'],
    "target": 'fapar',
    "trials": {
        'all_data': 'index == index',
    },
    "training_size": 0.7,
    "test_size": 0.15,
    "normalize": True,
    "output_path": "outputs/output_splits"
}
