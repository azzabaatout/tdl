

import json
from pathlib import Path
from collections import defaultdict

def extract_results():
    """Extract error rates from all experiment logs."""
    # Dictionary to store results
    results = defaultdict(dict)

    attacks = {
        'none': 'No Attack',
        'la': 'LA',
        'mb': 'MB',
        'faker': 'Faker'
    }

    defense_names = {
        'baseline': 'Baseline (FA)',
        'fltrust': 'FLTrust',
        'normclip': 'Norm-clipping',
        'spp': 'SPP',
        'krum': 'Krum'
    }

    log_base = Path('logs')
    if not log_base.exists():
        print("Error - logs/ directory not found. Run experiments first!")
        return

    found_experiments = []
    for log_dir in log_base.glob('*'):
        if not log_dir.is_dir():
            continue

        exp_name_parts = log_dir.name.split('_')

        if len(exp_name_parts) >= 2:
            defense = exp_name_parts[0]
            attack = exp_name_parts[1]

            if defense in defense_names and attack in attacks:
                metrics_file = log_dir / 'metrics.json'
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)

                        # Extract final accuracy
                        conv_metrics = metrics.get('convergence_metrics', {})
                        final_acc = conv_metrics.get('final_accuracy', 0)

                        error_rate = 1 - (final_acc / 100.0)

                        results[defense][attacks[attack]] = error_rate
                        found_experiments.append(f"{defense}_{attack}")

                        print(f"✓ Extracted: {defense:10} vs {attack:6} -> Error: {error_rate:.4f}")
                    except Exception as e:
                        print(f"✗ Error reading {log_dir.name}: {e}")

    if not found_experiments:
        print("\nNo experiment results found!")
        print("Make sure to run experiments first.")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(found_experiments)} completed experiments")
    print(f"{'='*60}\n")

    # Print comparison table
    print_table(results, defense_names, attacks)

    # Print statistics
    print_statistics(results, defense_names, attacks)

    return results

def print_table(results, defense_names, attacks):


    for defense_key in ['baseline', 'fltrust', 'normclip', 'spp', 'krum']:
        label = defense_names[defense_key]
        row = f"║ {label:13} ║"

        for attack in ['No Attack', 'LA', 'MB', 'Faker']:
            if attack in results[defense_key]:
                val = f"{results[defense_key][attack]:.2f}"
            else:
                val = "  ???  "
            row += f" {val:^9} ║"
        print(row)


def print_statistics(results, defense_names, attacks):
    """Print additional statistics and insights."""
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    faker_results = []
    for defense in ['baseline', 'fltrust', 'normclip', 'spp', 'krum']:
        if 'Faker' in results[defense]:
            faker_results.append((defense, results[defense]['Faker']))

    if faker_results:
        faker_results.sort(key=lambda x: x[1])
        best_defense, best_error = faker_results[0]
        worst_defense, worst_error = faker_results[-1]

        print(f"\n🎯 FAKER ATTACK DEFENSE RANKING:")
        for i, (defense, error) in enumerate(faker_results, 1):
            label = defense_names[defense]
            marker = "⭐" if i == 1 else "  "
            print(f"  {marker} {i}. {label:15} - Error: {error:.4f}")

        if best_defense == 'spp':
            print(f"\n✨ SPP (your contribution) is the BEST defense against Faker!")
            print(f"   Error rate improvement over worst: {(worst_error - best_error):.4f}")

    print(f"\n📊 DEFENSE OVERHEAD (vs No Attack baseline):")
    baseline_no_attack = results['baseline'].get('No Attack', None)

    for defense in ['fltrust', 'normclip', 'spp', 'krum']:
        no_attack_err = results[defense].get('No Attack', None)
        if baseline_no_attack is not None and no_attack_err is not None:
            overhead = no_attack_err - baseline_no_attack
            label = defense_names[defense]
            print(f"  {label:15} - Overhead: {overhead:+.4f}")

    print(f"\n⚔️  ATTACK SEVERITY (average error increase):")
    for attack in ['LA', 'MB', 'Faker']:
        errors = []
        for defense in ['baseline', 'fltrust', 'normclip', 'spp', 'krum']:
            no_attack = results[defense].get('No Attack')
            with_attack = results[defense].get(attack)
            if no_attack is not None and with_attack is not None:
                errors.append(with_attack - no_attack)

        if errors:
            avg_increase = sum(errors) / len(errors)
            print(f"  {attack:6} - Avg error increase: {avg_increase:+.4f}")

def main():
    """Main function."""
    print("\n" + "="*60)
    print("EXTRACTING EXPERIMENT RESULTS")
    print("="*60 + "\n")

    results = extract_results()

    if results:
        print("\n" + "="*60)
        print("Results saved! Copy the table above for your paper.")
        print("="*60 + "\n")

if __name__ == '__main__':
    main()