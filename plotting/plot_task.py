import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

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

def calculate_success_rate(data, task):
    # rollout/return - (-1 * rollout/episode_steps) / rollout/episode_steps
    success_rate = np.array(data['rollout/return']) - (-1 * np.array(data['rollout/episode_steps'])) / np.array(data['rollout/episode_steps'])
    success_rate = np.clip(success_rate, 0, 1)
    return success_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logs_dir', type=str, help='logs directory containing algorithms with task folders of training progress to plot')
    parser.add_argument('task', type=str, help='task name')
    parser.add_argument('--plot_separate', action='store_true', default=False, help='Plot individual input lines instead of mean/std/error')
    args = parser.parse_args()

    plt.figure(figsize=(10, 6))

    print(f'Plotting {args.task} from {args.logs_dir}')

    color_map = {
        'ddpg': 'red',
        'her': 'green',
        'herdemo': 'blue',
        'ddpgcl': 'orange',
        'hercl': 'purple',
        'herdemocl': 'pink',
    }

    # go through all directories inside logs_dir
    # filter by directories that start with the task name
    all_results = {}
    for alg_dir in os.listdir(args.logs_dir):
        print(f'Processing {alg_dir}')
        alg_name = str(alg_dir)
        for run_dir in os.listdir(os.path.join(args.logs_dir, alg_dir)):
            if run_dir.startswith(args.task):
                print(f'Processing {run_dir}')
                csv_path = os.path.join(args.logs_dir, alg_dir, run_dir, 'progress.csv')
                results = load_results(csv_path)
                if not results:
                    print(f'Skipping {csv_path} (no data)')
                    continue

                if 'test/success_rate' not in results:
                    results['test/success_rate'] = calculate_success_rate(results, args.task)
                success_rate = np.array(results['test/success_rate'])

                if 'epoch' not in results:
                    results['epoch'] = results['total/epochs']
                epoch = np.array(results['epoch']) + 1

                epoch, success_rate = smooth_reward_curve(epoch, success_rate)

                if alg_name not in all_results:
                    all_results[alg_name] = {'epochs': [], 'success_rates': []}
                all_results[alg_name]['epochs'].append(epoch)
                all_results[alg_name]['success_rates'].append(success_rate)


    for alg_name, data in all_results.items():
        min_length = min(len(ep) for ep in data['epochs'])
        trim_epochs = np.array([ep[:min_length] for ep in data['epochs']])
        trim_success_rates = np.array([sr[:min_length] for sr in data['success_rates']])

        mean = np.mean(trim_success_rates, axis=0)
        std = np.std(trim_success_rates, axis=0)
        stderr = std / np.sqrt(len(trim_success_rates))

        color = color_map[alg_name]

        if args.plot_separate:
            for i, (epoch, success_rate) in enumerate(zip(trim_epochs, trim_success_rates)):
                label = f'{alg_name} run {i+1}'
                plt.plot(epoch, success_rate, alpha=0.5, label=label, color=color)
        else:
            plt.plot(trim_epochs[0], mean, label=alg_name, color=color)
            plt.fill_between(trim_epochs[0], mean - std, mean + std, alpha=0.2, color=color)
            plt.fill_between(trim_epochs[0], mean - stderr, mean + stderr, alpha=0.4, color=color)
            plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.25))  

    if args.task == "NeedleReach":
        plt.xlim(0, 20)

    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1.05)
    plt.title(args.task)
    plt.grid(True)
    plt.tight_layout()

    if args.plot_separate:
        plt.savefig("./plotting/pngs/" + args.task + "_separate.png", dpi=300)
    else:
        plt.savefig("./plotting/pngs/" + args.task + ".png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()