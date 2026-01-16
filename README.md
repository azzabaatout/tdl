# Trustworthy Distributed Learning
Implementation of the paper "Can We Trust the Similarity Measurement in Federated Learning?" Wang et al. 2023

---

## Quick Start - Run ALL Experiments

```bash
# Make script executable
chmod +x RUN_TABLE_EXPERIMENTS.sh

# Run all 20 experiments
./RUN_TABLE_EXPERIMENTS.sh
```

---

## Individual Commands (Run these one by one)

### Baseline (FedAvg) - No Defense

```bash
# No Attack
python src/experiments.py --config configs/table/baseline_none.yaml

# LA Attack
python src/experiments.py --config configs/table/baseline_la.yaml

# MB Attack
python src/experiments.py --config configs/table/baseline_mb.yaml

# Faker Attack
python src/experiments.py --config configs/table/baseline_faker.yaml
```

---

### FLTrust Defense

```bash
# No Attack
python src/experiments.py --config configs/table/fltrust_none.yaml

# LA Attack
python src/experiments.py --config configs/table/fltrust_la.yaml

# MB Attack
python src/experiments.py --config configs/table/fltrust_mb.yaml

# Faker Attack
python src/experiments.py --config configs/table/fltrust_faker.yaml
```

---

### Norm-clipping Defense

```bash
# No Attack
python src/experiments.py --config configs/table/normclip_none.yaml

# LA Attack
python src/experiments.py --config configs/table/normclip_la.yaml

# MB Attack
python src/experiments.py --config configs/table/normclip_mb.yaml

# Faker Attack
python src/experiments.py --config configs/table/normclip_faker.yaml
```

---

### SPP Defense

```bash
# No Attack
python src/experiments.py --config configs/table/spp_none.yaml

# LA Attack
python src/experiments.py --config configs/table/spp_la.yaml

# MB Attack
python src/experiments.py --config configs/table/spp_mb.yaml

# Faker Attack
python src/experiments.py --config configs/table/spp_faker.yaml
```

---

### Krum Defense

```bash
# No Attack
python src/experiments.py --config configs/table/krum_none.yaml

# LA Attack
python src/experiments.py --config configs/table/krum_la.yaml

# MB Attack
python src/experiments.py --config configs/table/krum_mb.yaml

# Faker Attack
python src/experiments.py --config configs/table/krum_faker.yaml
```

---
