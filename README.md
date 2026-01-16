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

### ROW 1: Baseline (FedAvg) - No Defense

```bash
# Column 1: No Attack
python src/experiments.py --config configs/table/baseline_none.yaml

# Column 2: LA Attack
python src/experiments.py --config configs/table/baseline_la.yaml

# Column 3: MB Attack
python src/experiments.py --config configs/table/baseline_mb.yaml

# Column 4: Faker Attack
python src/experiments.py --config configs/table/baseline_faker.yaml
```

---

### ROW 2: FLTrust Defense

```bash
# Column 1: No Attack
python src/experiments.py --config configs/table/fltrust_none.yaml

# Column 2: LA Attack
python src/experiments.py --config configs/table/fltrust_la.yaml

# Column 3: MB Attack
python src/experiments.py --config configs/table/fltrust_mb.yaml

# Column 4: Faker Attack (MOST IMPORTANT)
python src/experiments.py --config configs/table/fltrust_faker.yaml
```

---

### ROW 3: Norm-clipping Defense

```bash
# Column 1: No Attack
python src/experiments.py --config configs/table/normclip_none.yaml

# Column 2: LA Attack
python src/experiments.py --config configs/table/normclip_la.yaml

# Column 3: MB Attack
python src/experiments.py --config configs/table/normclip_mb.yaml

# Column 4: Faker Attack
python src/experiments.py --config configs/table/normclip_faker.yaml
```

---

### ROW 4: SPP Defense

```bash
# Column 1: No Attack
python src/experiments.py --config configs/table/spp_none.yaml

# Column 2: LA Attack
python src/experiments.py --config configs/table/spp_la.yaml

# Column 3: MB Attack
python src/experiments.py --config configs/table/spp_mb.yaml

# Column 4: Faker Attack (KEY RESULT!)
python src/experiments.py --config configs/table/spp_faker.yaml
```

---

### ROW 5: Krum Defense

```bash
# Column 1: No Attack
python src/experiments.py --config configs/table/krum_none.yaml

# Column 2: LA Attack
python src/experiments.py --config configs/table/krum_la.yaml

# Column 3: MB Attack
python src/experiments.py --config configs/table/krum_mb.yaml

# Column 4: Faker Attack
python src/experiments.py --config configs/table/krum_faker.yaml
```

---
