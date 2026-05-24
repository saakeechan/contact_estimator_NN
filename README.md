
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

Each training run creates a timestamped directory in `logs/run_YYYY-MM-DD_HH-MM-SS/` containing:
- **network_params.yaml** - Copy of configuration used
- **training_summary.txt/json** - Final metrics, training time, best results
- **training_validation_loss.png** - Loss curves (auto-generated)
- **training_validation_mae.png** - MAE curves (auto-generated)
- **tensorboard/** - TensorBoard event logs
- **model_*.pt** - Best and final model checkpoints

### 4. Test Model
Evaluate on test set:
```bash
python3 src/test.py --config_name config/network_params.yaml
```

### 5. Plot Loss
Plot training and validation loss/MAE from TensorBoard logs (saves to `results/` directory):
```bash
python3 utils/plot_loss.py
```

**Note:** Plots are now automatically generated at the end of training and saved to the run directory. 
This manual command is only needed if you want to regenerate plots from existing logs.

This creates:
- `results/training_validation_loss.png`
- `results/training_validation_mae.png`

Optional: Specify custom output directory or metric tags:
```bash
python3 utils/plot_loss.py --output-dir plots \
  --train-loss-tag "training/total_loss" \
  --val-loss-tag "validation/total_loss"
```

### 6. View Training Results
To view TensorBoard logs for a specific run:
```bash
tensorboard --logdir=logs/run_YYYY-MM-DD_HH-MM-SS/tensorboard
```

Or view all runs together:
```bash
tensorboard --logdir=logs
```

## Configuration

All parameters are centralized in `config/network_params.yaml`:
