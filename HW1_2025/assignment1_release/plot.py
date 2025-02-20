from pathlib import Path

from HW1_2025.assignment1_release.utils import generate_plots

if __name__ == "__main__":
    root_dir = Path('C:/Users/Hugo/PycharmProjects/IFT6135-2025/output')
    list_of_dirs = [
        root_dir / 'p4_q2_relu',
        root_dir / 'p4_q2_tanh',
        root_dir / 'p4_q2_sigmoid'
    ]

    legend_names = [
        'relu',
        'tanh',
        'sigmoid'
    ]

    generate_plots(list_of_dirs, legend_names, 'figs')
