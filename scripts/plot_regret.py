
from pathlib import Path
import seaborn as sns
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import argparse

from jumbo.util import read_yaml


DEFAULT_LABEL_SIZE = 20
DEFAULT_TICK_SIZE = 15


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='yaml config file')
    return parser.parse_args()


def plt_regret(reg_t: np.ndarray):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    avg_reg = np.cumsum(reg_t) / (np.arange(len(reg_t)) + 1)[None, :]
    simp_reg = np.minimum.accumulate(reg_t)[None, :]

    df = pd.DataFrame(simp_reg).melt()
    sns.lineplot(x='variable', y='value', data=df, ax=ax1)
    ax1.set_ylabel('Simple Regret')
    ax1.set_ylabel('iter')

    df = pd.DataFrame(list(avg_reg)).melt()
    sns.lineplot(x='variable', y="value", data=df, ax=ax2)

    ax2.set_ylabel('Cumulative Regret')
    ax2.set_xlabel('iter')
    return f

def main(config):
    exclude_list = config.get('exclude_list', [])
    props = config['props']
    content = config['content']

    subplot_kwargs = props['subplot']
    plt_config = props['plt']
    cum_regret = props['content']['cum_regret']
    timed = props['content']['timed']
    legend_cf = props['legend']
    yrange = props.get('yrange', None)
    xrange = props.get('xrange', None)
    index_max = props.get('index_max', None)
    legend_ax_id = legend_cf.pop('ax_id', None)

    # fontsize
    fontsize = props.get('fontsize', {})
    ax_label_size = fontsize.get('labels', DEFAULT_LABEL_SIZE)
    tick_label_size = fontsize.get('ticks', DEFAULT_TICK_SIZE)


    f, axes = plt.subplots(**subplot_kwargs)
    legend_handles, legend_labels = [], []

    if isinstance(axes, plt.Axes):
        axes = np.array([axes])
    for idx, (content_item, ax) in enumerate(zip(content, axes.flatten())):
        ax.grid()
        title = content_item.get('title', None)
        lb = content_item.get('lower_bound', None)
        lines = content_item['lines']

        for label, path in lines.items():
            if label in exclude_list:
                continue
            path = Path(path)
            runtime = None
            try:
                res = read_yaml(path / 'res.yaml')
                reg_t = np.array(res['y'])
                if index_max is not None:
                    reg_t = reg_t[:index_max]

                if lb is not None:
                    reg_t = reg_t - lb
                avg_reg = np.cumsum(reg_t) / (np.arange(len(reg_t)) + 1)[None, :]
                simp_reg = np.minimum.accumulate(reg_t)[None, :]
                if timed:
                    try:
                        runtime = np.array(res['runtime'])
                    except KeyError:
                        raise ValueError('The plot should be against time but runtime is not '
                                         'provided.')
            except FileNotFoundError:
                # it has multiple seeds
                avg_reg, simp_reg, seed_arr = [], [], []
                runtime = []
                seed_id = 0
                seed_list = []
                for seed_path in Path(path).iterdir():
                    if seed_path.is_dir():
                        res = read_yaml(seed_path / 'res.yaml')
                        reg_t = np.array(res['y'])
                        if index_max is not None:
                            reg_t = reg_t[:index_max]
                        if lb is not None:
                            reg_t = reg_t - lb
                        avg_reg.append(np.cumsum(reg_t) / (np.arange(len(reg_t)) + 1))
                        simp_reg.append(np.minimum.accumulate(reg_t))

                        if 'runtime' in res:
                            runtime.append(np.array(res['runtime']).flatten())
                        seed_arr.append(np.array([seed_id] * len(reg_t)))
                        seed_id += 1
                        seed_list.append(seed_path.stem)

            if timed and runtime:
                com_len = min(x.shape[0] for x in simp_reg)
                simp_reg = np.array([x[:com_len] for x in simp_reg])
                avg_reg = np.array([x[:com_len] for x in avg_reg])
                runtime = np.array([x[:com_len] for x in runtime])
                nseed = avg_reg.shape[0]

                avg_reg = np.array(avg_reg)

                simp_reg_mean = np.mean(simp_reg, axis=0)
                simp_reg_std = np.std(simp_reg, axis=0)
                yerr = simp_reg_std / np.sqrt(nseed)

                runtime_mean = np.mean(runtime, axis=0)
                runtime_std = np.std(runtime, axis=0)

                ax.plot(runtime_mean, simp_reg_mean, label=label, **plt_config[label])
                ax.fill_between(x=runtime_mean,
                                y1=simp_reg_mean - yerr,
                                y2=simp_reg_mean + yerr,
                                color=plt_config[label]['color'], alpha=0.4)

                ax.set_xscale('log')
                if yrange:
                    ax.set_ylim(*[10**y for y in yrange])
            else:
                df = pd.DataFrame(simp_reg).melt()
                sns.lineplot(x='variable', y='value', data=df, label=label, ax=ax,
                             **plt_config[label])
                df = pd.DataFrame(list(avg_reg)).melt()

        ax.set_ylabel('Simple Regret', fontsize=ax_label_size)

        handles, labels = ax.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            # all handles with the same label should be identical as per code's assumption
            if label not in legend_labels:
                legend_labels.append(label)
                legend_handles.append(handle)

        if lb is not None:
            ax.set_yscale('log')

        last_row = len(content) - 1 - axes.shape[-1] < idx and idx <= len(content) - 1
        if last_row:
            if timed:
                ax.set_xlabel('runtime (seconds)', fontsize=ax_label_size)
            else:
                ax.set_xlabel('iteration', fontsize=ax_label_size)

            if xrange:
                ax.set_xlim(*xrange)

        if title:
            ax.set_title(title, fontsize=ax_label_size)

        # set tick label fontsize
        ax.tick_params(axis='x', labelsize=tick_label_size)
        ax.tick_params(axis='y', labelsize=tick_label_size)

        if idx == len(content) - 1:
            if legend_ax_id is not None:
                ax = axes.flatten()[legend_ax_id]
            ax.legend(handles=legend_handles, labels=legend_labels, **legend_cf)


    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    _args = _get_args()
    config = read_yaml(_args.config)
    main(config)
