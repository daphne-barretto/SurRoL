"""
USAGE:
python plotMany.py <path to folder containing a progrss.csv folder with 'epoch' and 'test/success_rate' columns>
e.g. python plotMany.py C:\\Users\\megan\Downloads\\logs_0\\logs_0\\her\\NeedleReach-1e5_0 C:\\Users\\megan\\Downloads\\logs_1\\logs_1\\her\\NeedleReach-1e5_0 
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns; sns.set()


def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result

parser = argparse.ArgumentParser()
parser.add_argument('dirs', nargs='+', type=str, help='List of directories containing progress.csv to plot')
parser.add_argument('--smooth', type=int, default=1)
parser.add_argument('--hide_ddpg', action='store_true', default=False, help='Hide ddpg red line')
parser.add_argument('--plot_separate', action='store_true', default=False, help='Plot individual input lines instead of mean/std/error')
args = parser.parse_args()

plt.figure(figsize=(10, 6))

all_success_rates = []
all_epochs = []

for dir_path in args.dirs:
    csv_path = os.path.join(dir_path, 'progress.csv')
    results = load_results(csv_path)
    if not results:
        print(f'Skipping {csv_path} (no data)')
        continue

    success_rate = np.array(results['test/success_rate'])
    epoch = np.array(results['epoch']) + 1

    if args.smooth:
        epoch, success_rate = smooth_reward_curve(epoch, success_rate)

    all_epochs.append(epoch)
    all_success_rates.append(success_rate)

min_length = min(len(ep) for ep in all_epochs)
trim_epochs = np.array([ep[:min_length] for ep in all_epochs])
trim_success_rates = np.array([sr[:min_length] for sr in all_success_rates])

mean = np.mean(trim_success_rates, axis=0)
std = np.std(trim_success_rates, axis=0)
stderr = std / np.sqrt(len(trim_success_rates))

if not args.plot_separate:
    alg = os.path.normpath(args.dirs[0]).split(os.sep)[-2]
    plt.plot(trim_epochs[0], mean, color='blue', label=alg)
    plt.fill_between(trim_epochs[0], mean - std, mean + std, color='blue', alpha=0.2)
    plt.fill_between(trim_epochs[0], mean - stderr, mean + stderr, color='blue', alpha=0.4)
else:
    for dir_path, epoch, success_rate in zip(args.dirs, trim_epochs, trim_success_rates):
        label = os.path.normpath(dir_path).split(os.sep)[-3]
        plt.plot(epoch, success_rate, alpha=0.5, label=label)
      
if not args.hide_ddpg:
    plt.plot(trim_epochs[0], np.zeros_like(trim_epochs[0]), color='red', label='ddpg')

plt.xlabel('Epoch')
plt.ylabel('Success Rate')
plot_title = os.path.basename(os.path.normpath(args.dirs[0]))
plt.title(plot_title)
plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.25))
plt.grid(True)
plt.tight_layout()
plt.show()
