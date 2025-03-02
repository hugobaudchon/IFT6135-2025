import json
import os
import time
from pathlib import Path
import random

if __name__ == "__main__":
    output_dir = 'output/p4_q4'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    patch_sizes = [2, 4, 8, 16]
    max_epochs = 60
    patience = 10

    best_patch_size = None
    best_run_test_acc = 0
    best_run_valid_acc = 0

    for i in range(len(patch_sizes)):
        a = time.time()
        patch_size = patch_sizes[i]

        print(f"Running {i} patch_size={patch_size}...")

        config_path = Path('./model_configs/mlpmixer_p4_q4.json')

        mlp_config = json.load(open(config_path, 'r'))
        mlp_config['patch_size'] = patch_size
        with open(config_path, 'w') as f:
            f.write(json.dumps(mlp_config, indent=4))

        run_logdir = f"{output_dir}/run_{i}"
        command = f"python main.py --model mlpmixer --model_config {str(config_path)} --patience {patience} --epochs {max_epochs} --logdir {run_logdir} > {output_dir}/log.txt 2>&1"
        os.system(command)

        # load results.json in logdir
        run_results = json.load(open(Path(run_logdir) / 'results.json', 'r'))
        test_acc = run_results['test_acc']
        best_valid_acc = max(run_results['valid_accs'])

        if best_valid_acc > best_run_valid_acc:
            best_run_valid_acc = best_valid_acc
            best_run_test_acc = test_acc
            best_patch_size = patch_size

        run_results['patience'] = patience
        run_results['patch_size'] = patch_size
        with open(Path(run_logdir) / 'results.json', 'w') as f:
            f.write(json.dumps(run_results, indent=4))

        b = time.time()
        print(f"Time taken for patch_size={patch_size}: {b-a}")

    print(f"Best patch_size: {best_patch_size}, with best checkpoint validation accuracy: {best_run_valid_acc} and test accuracy: {best_run_test_acc}")
