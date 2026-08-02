
## Quick Start

### 1. Start Docker Container
```bash
./docker.sh run
```

### 2. Process CSV Data
Convert CSV files to numpy format with run boundaries:
```bash
python3 utils/csv2numpyV1.py
```

### 3. Train Encoder
Train the configured DAE or VAE encoder:
```bash
python3 src/trainEncoder.py
```

### 4. Train Model
Train the contact estimation network:
```bash
python3 src/train.py
```

### 5. Test Model
Evaluate on test set:
```bash
python3 src/test.py
```

### 6. Test One Trajectory
Run the single-trajectory evaluation:
```bash
python3 src/testSingle.py
```

### 7. Run the Seed Sweep
Evaluate `testSingle.py` across the default seed range and write its CSV report:
```bash
python3 utils/run_testsingle_seeds.py
```

### 8. Run the Seed Sweep for Encoder
Evaluate `testSingleDecoupledInput.py` across the default seed range and write its CSV report:
```bash
python3 utils/run_testsingle_seedsDecoupled.py
```

Each training run creates a timestamped directory in `logs/run_YYYY-MM-DD_HH-MM-SS/` containing:
- **network_params.yaml** - Copy of configuration used
- **training_summary.txt/json** - Final metrics, training time, best results
- **training_validation_loss.png** - Loss curves (auto-generated)
- **training_validation_mae.png** - MAE curves (auto-generated)
- **tensorboard/** - TensorBoard event logs
- **model_*.pt** - Best and final model checkpoints

## Configuration

General settings are in `config/network_params.yaml`; NatPN settings are in
`config/NatPN_params.yaml`. The utility, training, and test commands load both.
