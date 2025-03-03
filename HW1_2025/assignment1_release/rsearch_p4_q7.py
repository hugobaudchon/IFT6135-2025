import json
import os
from pathlib import Path
import random

if __name__ == "__main__":
    output_dir = 'output/p4_q7'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    embed_dim_list = [
        64,
        128,
        256,
        384,
        512,
        1024
    ]
    max_epochs = 100
    patience = 10

    best_embed_dim = None
    best_run_test_acc = 0
    best_run_valid_acc = 0

    for i in range(len(embed_dim_list)):
        embed_dim = embed_dim_list[i]

        print(f"Running random search iteration {i} embed_dim={embed_dim}...")

        config_path = Path('./model_configs/mlpmixer_p4_q7.json')

        mlp_config = json.load(open(config_path, 'r'))
        mlp_config['embed_dim'] = embed_dim
        with open(config_path, 'w') as f:
            f.write(json.dumps(mlp_config, indent=4))

        run_logdir = f"{output_dir}/run_{i}"
        command = f"python main.py --model mlpmixer --model_config {str(config_path)} --epochs {max_epochs} --logdir {run_logdir} > {output_dir}/log.txt 2>&1"
        os.system(command)

        # load results.json in logdir
        run_results = json.load(open(Path(run_logdir) / 'results.json', 'r'))
        test_acc = run_results['test_acc']
        best_valid_acc = max(run_results['valid_accs'])

        if best_valid_acc > best_run_valid_acc:
            best_run_valid_acc = best_valid_acc
            best_run_test_acc = test_acc
            best_embed_dim = embed_dim

        run_results['patience'] = patience
        run_results['embed_dim'] = embed_dim
        with open(Path(run_logdir) / 'results.json', 'w') as f:
            f.write(json.dumps(run_results, indent=4))

    print(f"Best embed_dim: {best_embed_dim}, with best checkpoint validation accuracy: {best_run_valid_acc} and test accuracy: {best_run_test_acc}")
