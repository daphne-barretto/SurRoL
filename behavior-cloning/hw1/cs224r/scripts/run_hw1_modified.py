"""
Modified version of run_hw1.py to include ConvNetPolicySL for image-based observations
"""

import os
import time
import argparse
import pickle

from cs224r.infrastructure.bc_trainer import BCTrainer
from cs224r.agents.bc_agent import BCAgent
from cs224r.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from cs224r.policies.MLP_policy import MLPPolicySL
from cs224r.policies.ConvNet_policy import ConvNetPolicySL  # Import the new policy
from cs224r.infrastructure.image_data_processor import load_and_process_data
from cs224r.infrastructure.utils import MJ_ENV_KWARGS, MJ_ENV_NAMES


def run_bc(params):
    """
    Runs behavior cloning with the specified parameters

    Args:
        params: experiment parameters
    """

    #######################
    ## AGENT PARAMS
    #######################

    agent_params = {
        'n_layers': params['n_layers'],
        'size': params['size'],
        'learning_rate': params['learning_rate'],
        'max_replay_buffer_size': params['max_replay_buffer_size'],
    }
    
    # Add image-specific parameters if using ConvNetPolicySL
    if params['use_images']:
        agent_params.update({
            'image_shape': params['image_shape'],
            'n_conv_layers': params['n_conv_layers'],
            'conv_channels': params['conv_channels'],
            'conv_kernel_sizes': params['conv_kernel_sizes'],
            'conv_strides': params['conv_strides'],
            'n_fc_layers': params['n_fc_layers'],
            'fc_size': params['fc_size'],
        })
    
    params['agent_class'] = BCAgent
    params['agent_params'] = agent_params

    #######################
    ## ENVIRONMENT PARAMS
    #######################

    params["env_kwargs"] = MJ_ENV_KWARGS[params['env_name']]

    #######################
    ## LOAD EXPERT POLICY
    #######################

    print('Loading expert policy from...', params['expert_policy_file'])
    loaded_expert_policy = LoadedGaussianPolicy(
        params['expert_policy_file'])
    print('Done restoring expert policy...')
    
    #######################
    ## PREPROCESS DATA
    #######################
    
    # If we are using images, we want to preprocess our expert data
    if params['use_images'] and params['preprocess_data']:
        print(f"Preprocessing expert data from {params['expert_data']}...")
        processed_data_path = params['expert_data'].replace('.pkl', '_processed.pkl')
        
        # Check if processed data already exists
        if os.path.exists(processed_data_path) and not params['force_preprocess']:
            print(f"Using existing processed data at {processed_data_path}")
            params['expert_data'] = processed_data_path
        else:
            print(f"Processing expert data and saving to {processed_data_path}")
            paths = load_and_process_data(params['expert_data'], use_images=True)
            
            # Save processed data
            with open(processed_data_path, 'wb') as f:
                pickle.dump(paths, f)
            
            params['expert_data'] = processed_data_path
            print(f"Preprocessing complete. Processed data saved to {processed_data_path}")

    ###################
    ### RUN TRAINING
    ###################

    trainer = BCTrainer(params)
    
    # If using images, modify the policy class in the agent to use ConvNetPolicySL
    if params['use_images']:
        # Replace MLPPolicySL with ConvNetPolicySL
        trainer.agent.actor = ConvNetPolicySL(
            trainer.agent.agent_params['ac_dim'],
            trainer.agent.agent_params['ob_dim'],
            image_shape=params['image_shape'],
            n_conv_layers=params['n_conv_layers'],
            conv_channels=params['conv_channels'],
            conv_kernel_sizes=params['conv_kernel_sizes'],
            conv_strides=params['conv_strides'],
            n_fc_layers=params['n_fc_layers'],
            fc_size=params['fc_size'],
            learning_rate=trainer.agent.agent_params['learning_rate'],
        )
    
    trainer.run_training_loop(
        n_iter=params['n_iter'],
        initial_expertdata=params['expert_data'],
        collect_policy=trainer.agent.actor,
        eval_policy=trainer.agent.actor,
        relabel_with_expert=params['do_dagger'],
        expert_policy=loaded_expert_policy,
    )


def main():
    """
    Parses arguments, creates logger, and runs behavior cloning
    """

    parser = argparse.ArgumentParser()
    # NOTE: The file path is relative to where you're running this script from
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)
    parser.add_argument('--expert_data', '-ed', type=str, required=True)
    parser.add_argument('--env_name', '-env', type=str,
        help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    parser.add_argument('--exp_name', '-exp', type=str,
        default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int)

    # Sets the number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int,
        default=1000)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    # Amount of training data collected (in the env) during each iteration
    parser.add_argument('--batch_size', type=int, default=1000)
    # Amount of evaluation data collected (in the env) for logging metrics
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    # Number of sampled data points to be used per gradient/train step
    parser.add_argument('--train_batch_size', type=int, default=100)

    # Depth of the policy to be learned
    parser.add_argument('--n_layers', type=int, default=2)
    # Width of each layer of the policy to be learned
    parser.add_argument('--size', type=int, default=64)
    # Learning rate for supervised learning
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)

    # NEW: Image-based observation parameters
    parser.add_argument('--use_images', action='store_true',
        help='Whether to use image-based observations')
    parser.add_argument('--image_shape', nargs='+', type=int, default=[3, 480, 640],
        help='Shape of image observations (channels, height, width)')
    parser.add_argument('--n_conv_layers', type=int, default=3,
        help='Number of convolutional layers')
    parser.add_argument('--conv_channels', nargs='+', type=int, default=[16, 32, 64],
        help='Number of channels in each convolutional layer')
    parser.add_argument('--conv_kernel_sizes', nargs='+', type=int, default=[5, 3, 3],
        help='Kernel sizes in each convolutional layer')
    parser.add_argument('--conv_strides', nargs='+', type=int, default=[2, 2, 2],
        help='Strides in each convolutional layer')
    parser.add_argument('--n_fc_layers', type=int, default=2,
        help='Number of fully connected layers')
    parser.add_argument('--fc_size', type=int, default=64,
        help='Size of fully connected layers')
    parser.add_argument('--preprocess_data', action='store_true',
        help='Whether to preprocess the expert data')
    parser.add_argument('--force_preprocess', action='store_true',
        help='Force preprocessing even if processed data exists')

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # Convert arguments to dictionary for easy reference
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if args.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q2_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) \
            of training, to iteratively query the expert and train \
            (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data \
            just once (n_iter=1)')

    # If using images, add a suffix to the log directory
    if args.use_images:
        logdir_prefix += 'img_'

    # Directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        '../../data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + \
        time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    run_bc(params)


if __name__ == "__main__":
    main()