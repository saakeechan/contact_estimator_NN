
## Quick Start

### 1. Start Docker Container
```bash
docker start contact_estimator_NN && docker exec -it contact_estimator_NN /bin/bash
```

### 2. Process CSV Data
Convert CSV files to numpy format with run boundaries:
```bash
python3 utils/csv2numpy.py --config_name config/network_params.yaml
```

### 3. Train Model
Train the contact estimation network:
```bash
python3 src/train.py --config_name config/network_params.yaml
```

### 4. Test Model
Evaluate on test set:
```bash
python3 src/test.py --config_name config/network_params.yaml
```

## Configuration

All parameters are centralized in `config/network_params.yaml`:
- Data paths and CSV processing settings
- Model architecture (`num_features`, `window_size`)
- Training hyperparameters (`batch_size`, `learning_rate`, `num_epochs`)
- Train/val/test split ratios
- Regularization settings
- Test and evaluation parameters