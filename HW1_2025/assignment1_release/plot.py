import os
from pathlib import Path

from HW1_2025.assignment1_release.utils import generate_plots

if __name__ == "__main__":
    root_dir = Path('/home/hugo/PycharmProjects/IFT6135-2025/HW1_2025/assignment1_release/output')
    list_of_dirs = [
        # root_dir / 'p4_q2_relu',
        # root_dir / 'p4_q2_tanh',
        # root_dir / 'p4_q2_sigmoid'

        # root_dir / 'p4_q3_lr0p1',
        # root_dir / 'p4_q3_lr0p01',
        # root_dir / 'p4_q3_lr0p001',
        # root_dir / 'p4_q3_lr0p0001',
        # root_dir / 'p4_q3_lr0p00001'

        root_dir / 'p4_q4_patchsize2',
        root_dir / 'p4_q4_patchsize4',
        root_dir / 'p4_q4_patchsize8',
        root_dir / 'p4_q4_patchsize16',
    ]

    legend_names = [
        # 'relu',
        # 'tanh',
        # 'sigmoid'

        # 'lr=0.1',
        # 'lr=0.01',
        # 'lr=0.001',
        # 'lr=0.0001',
        # 'lr=0.00001'

        'patchsize=2',
        'patchsize=4',
        'patchsize=8',
        'patchsize=16'
    ]

    output_dir = 'figs_p4_q4'
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    generate_plots(list_of_dirs, legend_names, output_dir)
