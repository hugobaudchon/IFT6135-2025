import os
from pathlib import Path

from HW1_2025.assignment1_release.utils import generate_gradient_layer_plot

if __name__ == "__main__":
    root_dir = Path('/home/hugo/PycharmProjects/IFT6135-2025/HW1_2025/assignment1_release/output')
    name = 'p4_q8_gradients_mlpmixer'
    title = 'Gradient norm of MLPMixer'
    max_epoch = 60
    log_dir = root_dir / name

    output_dir = f'figs_{name}'
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    generate_gradient_layer_plot(log_dir, output_dir, max_epoch=max_epoch, title=title)
