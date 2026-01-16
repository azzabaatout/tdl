#!/bin/bash
# Script to run all experiments for the comparison table

echo "=========================================="
echo "Running ALL Experiments"
echo "Total: 20 experiments (5 defenses × 4 attacks)"
echo "=========================================="
echo ""

# Row 1: Baseline (FedAvg) - No Defense
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ROW 1: Baseline (FedAvg)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/experiments.py --config configs/table/baseline_none.yaml
python src/experiments.py --config configs/table/baseline_la.yaml
python src/experiments.py --config configs/table/baseline_mb.yaml
python src/experiments.py --config configs/table/baseline_faker.yaml

# Row 2: FLTrust Defense
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ROW 2: FLTrust Defense"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/experiments.py --config configs/table/fltrust_none.yaml
python src/experiments.py --config configs/table/fltrust_la.yaml
python src/experiments.py --config configs/table/fltrust_mb.yaml
python src/experiments.py --config configs/table/fltrust_faker.yaml

# Row 3: Norm-clipping Defense
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ROW 3: Norm-clipping Defense"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/experiments.py --config configs/table/normclip_none.yaml
python src/experiments.py --config configs/table/normclip_la.yaml
python src/experiments.py --config configs/table/normclip_mb.yaml
python src/experiments.py --config configs/table/normclip_faker.yaml

# Row 4: SPP Defense
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ROW 4: SPP Defense"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/experiments.py --config configs/table/spp_none.yaml
python src/experiments.py --config configs/table/spp_la.yaml
python src/experiments.py --config configs/table/spp_mb.yaml
python src/experiments.py --config configs/table/spp_faker.yaml

# Row 5: Krum Defense
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ROW 5: Krum Defense"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/experiments.py --config configs/table/krum_none.yaml
python src/experiments.py --config configs/table/krum_la.yaml
python src/experiments.py --config configs/table/krum_mb.yaml
python src/experiments.py --config configs/table/krum_faker.yaml
