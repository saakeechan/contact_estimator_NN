
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
Train the configured DAE or VAE encoder. When `train_task_after_encoder: true`, this automatically starts NatPN task training with `src/trainNatPN.py`.
```bash
python3 src/trainEncoder.py
```

### 4. Train NatPN Model
Train the NatPN contact-estimation network manually:
```bash
python3 src/trainNatPN.py
```

### 5. Train DER Model
Train the deep evidential-regression version:
```bash
python3 src/trainDER.py
```

### 6. Train Standard Model
Train the non-evidential baseline:
```bash
python3 src/train.py
```

### 7. Test Model
Evaluate on test set:
```bash
python3 src/test.py
```

### 8. Test One Trajectory
Run the single-trajectory evaluation:
```bash
python3 src/testSingleNatPN.py
```

### 9. Run a Seed Sweep
```bash
python3 utils/testSeries.py
```

Each seed sweep writes a CSV, PDF table, and aggregate plot under `testResults/`.
Select the model, seed range, OOD feature, and command-velocity/environment windows
at the top of `utils/testSeries.py`.
The default CSV names include the network and inclusive seed range, such as
`natpn_seeds_500-600.csv`, `der_seeds_500-600.csv`, and `encoder_seeds_2800-2851.csv`.

Training runs create timestamped directories in `logs/`, `logsNatPN/`, `logsDER/`, or `logsEncoder/`, containing:
- **network_params.yaml** - Copy of configuration used
- **training_summary.txt/json** - Final metrics, training time, best results
- **training_validation_loss.png** - Loss curves (auto-generated)
- **training_validation_mae.png** - MAE curves (auto-generated)
- **tensorboard/** - TensorBoard event logs
- **model_*.pt** - Best and final model checkpoints

## Configuration

General settings are in `config/network_params.yaml`; NatPN settings are in
`config/NatPN_params.yaml`. The utility, training, and test commands load both.



## TO-DO
1. Learn alternating optimization for bayesian while flow is frozen and flow while task encoder is frozen
2. Reformulate the whole bayesian structure because you dont really have prior over mean. Look into IG distributions and get lambda from normalizing flow
3. Look at the 3 losses, chatpgt gave vs papers, and understand the difference
