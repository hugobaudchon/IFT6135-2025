import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json


def generate_plots(list_of_dirs, legend_names, save_path):
    """ Generate plots according to log 
    :param list_of_dirs: List of paths to log directories
    :param legend_names: List of legend names
    :param save_path: Path to save the figs
    """
    assert len(list_of_dirs) == len(legend_names), "Names and log directories must have same length"
    data = {}
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        assert os.path.exists(os.path.join(logdir, 'results.json')), f"No json file in {logdir}"
        with open(json_path, 'r') as f:
            data[name] = json.load(f)
    
    for yaxis in ['train_accs', 'valid_accs', 'train_losses', 'valid_losses']:
        fig, ax = plt.subplots()
        for name in data:
            ax.plot(data[name][yaxis], label=name)
        ax.legend()
        ax.set_xlabel('epochs')
        ax.set_ylabel(yaxis.replace('_', ' '))
        fig.savefig(os.path.join(save_path, f'{yaxis}.png'))


def generate_gradient_layer_plot(log_dir, save_path, max_epoch, title):
    json_path = os.path.join(log_dir, 'results.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No results.json found in {log_dir}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    train_gradients = data.get("train_gradients", [])
    if not train_gradients:
        raise ValueError("No gradient data found in results.json")

    # layer_avg will store: { layer_name: [avg_grad_epoch1, avg_grad_epoch2, ...] }
    layer_avg = {}

    for epoch_grad in train_gradients:
        if not epoch_grad:
            continue
        epoch_layer_avg = {}
        for layer in epoch_grad[0].keys():
            batch_vals = [batch.get(layer, 0.0) for batch in epoch_grad]
            epoch_layer_avg[layer] = sum(batch_vals) / len(batch_vals)
        for layer, avg in epoch_layer_avg.items():
            layer_avg.setdefault(layer, []).append(avg)

    fig, ax = plt.subplots()

    epochs_range = range(1, max_epoch + 1)
    for layer, values in layer_avg.items():
        if len(values) < max_epoch:
            values_extended = values + [np.nan] * (max_epoch - len(values))
        else:
            values_extended = values[:max_epoch]
        ax.plot(epochs_range, values_extended, label=layer)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Gradient Norm")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(1, max_epoch)
    fig.savefig(os.path.join(save_path, "gradient_evolution.png"))
    plt.close(fig)
        

def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability.

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    elif isinstance(tensors, list):
        return list(
            (to_device(tensors[0], device), to_device(tensors[1], device)))
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor):
    """ Return the mean loss for this batch
    :param logits: [batch_size, num_class]
    :param labels: [batch_size]
    :return loss 
    """
    batch_size = logits.shape[0]
    one_hot = torch.zeros_like(logits)
    one_hot[torch.arange(batch_size), labels] = 1
    softmax = torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)
    loss = -torch.sum(one_hot * torch.log(softmax)) / batch_size
    return loss


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """ Compute the accuracy of the batch """
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return acc
