import sys
import os
import os.path as osp
import argparse
import numpy as np
import tensorflow as tf

from baselines.common.vec_env import VecFrameStack, VecNormalize
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import make_vec_env, make_env
from baselines.common import tf_util
import baselines.her.experiment.config as config

from baselines import logger
from importlib import import_module

def get_env_type(env_name, env_type=None):
    """Determine environment type from environment name"""
    import gym
    from collections import defaultdict
    
    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        env_type_parsed = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type_parsed].add(env.id)

    if env_type is not None:
        return env_type, env_name

    env_type = None
    for g, e in _game_envs.items():
        if env_name in e:
            env_type = g
            break
    
    if env_type is None and ':' in env_name:
        env_type = env_name.split(':')[0]
    
    assert env_type is not None, f'env_id {env_name} is not recognized'
    return env_type, env_name


def build_env_for_video(env_name, env_type=None, num_env=1, seed=42, reward_scale=1.0):
    """Build environment similar to the training setup"""
    env_type, env_id = get_env_type(env_name, env_type)
    
    if env_type in {'atari', 'retro'}:
        frame_stack_size = 4
        env = make_vec_env(env_id, env_type, num_env, seed, reward_scale=reward_scale)
        env = VecFrameStack(env, frame_stack_size)
    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        tf_util.get_session(config=config)

        # For HER, don't flatten dict observations
        flatten_dict_observations = False
        env = make_vec_env(env_id, env_type, num_env, seed, 
                          reward_scale=reward_scale, 
                          flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def load_model(model_path, env, env_name):
    """Load the saved model"""
    params = {"max_u": 1.0, "layers": 3, "hidden": 256, "network_class": "baselines.her.actor_critic:ActorCritic", "Q_lr": 0.001, "pi_lr": 0.001, "buffer_size": 1000000, "polyak": 0.95, "action_l2": 1.0, "clip_obs": 200.0, "scope": "ddpg", "relative_goals": False, "n_cycles": 50, "rollout_batch_size": 2, "n_batches": 40, "batch_size": 256, "n_test_rollouts": 10, "test_with_polyak": False, "random_eps": 0.3, "noise_eps": 0.2, "replay_strategy": "future", "replay_k": 4, "norm_eps": 0.01, "norm_clip": 5, "bc_loss": 0, "q_filter": 0, "num_demo": 100, "demo_batch_size": 128, "prm_loss_weight": 0.001, "aux_loss_weight": 0.0078, "env_name": env_name}
    params = config.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs

    params['bc_loss'] = 1
    kwargs = {'bc_loss': 1, 'q_filter': 1, 'num_demo': 1000, 'demo_batch_size': 128, 'prm_loss_weight': 0.001, 'aux_loss_weight': 0.0078, 'n_cycles': 20, 'batch_size': 1024, 'random_eps': 0.1, 'noise_eps': 0.1}
    params.update(kwargs)

    # config.log_params(params, logger=logger)
    dims = config.configure_dims(params)

    model = config.configure_ddpg(dims=dims, params=params, clip_return=1)
    if model_path is not None:
        tf_util.load_variables(model_path)
    return model


def generate_videos(model_path, env_name, video_dir='./videos', 
                   num_episodes=5, video_length=200, env_type=None, 
                   seed=42):
    """
    Generate videos using a saved policy
    
    Args:
        model_path: Path to saved model file
        env_name: Name of the environment
        video_dir: Directory to save videos
        num_episodes: Number of episodes to record
        video_length: Maximum length of each video
        env_type: Environment type (auto-detected if None)
        seed: Random seed
    """
    
    logger.configure(dir=video_dir)

    # Build environment
    logger.info(f"Building environment: {env_name}")
    env = build_env_for_video(env_name, env_type, num_env=1, seed=seed)
    
    # Wrap with video recorder and record every episode
    record_trigger = lambda episode_id: episode_id % 1 == 0
    env = VecVideoRecorder(env, video_dir, 
                          record_video_trigger=record_trigger,
                          video_length=video_length)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path, env, env_name)

    logger.info(f"Starting video generation for {num_episodes} episodes")
    logger.info(f"Videos will be saved to: {video_dir}")
    
    # Run episodes
    obs = env.reset()
    episode_count = 0
    episode_rewards = []
    current_episode_reward = 0
    
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))
    
    step_count = 0
    max_steps_per_episode = video_length * 2
    
    while episode_count < num_episodes:
        # Get action from model
        if state is not None:
            action, _, state, _ = model.step(obs, S=state, M=dones)
        else:
            action, _, _, _ = model.step(obs)
        
        # Take step in environment
        obs, reward, done, info = env.step(action)
        current_episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        step_count += 1
        
        episode_done = done[0] if isinstance(done, np.ndarray) else done
        if episode_done or step_count >= max_steps_per_episode:
            episode_count += 1
            episode_rewards.append(current_episode_reward)
            logger.info(f"Episode {episode_count}/{num_episodes} completed. Reward: {current_episode_reward:.2f}. Is_Success: {info[0]['is_success']}")
            
            current_episode_reward = 0
            step_count = 0
            
            if episode_count < num_episodes:
                obs = env.reset()
                if state is not None:
                    state = model.initial_state
                dones = np.zeros((1,))
    
    env.close()
    
    logger.info(f"\nVideo generation completed!")
    logger.info(f"Generated {num_episodes} episodes")
    logger.info(f"Average reward: {np.mean(episode_rewards):.2f}")
    logger.info(f"Videos saved to: {video_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate videos from saved HER+DDPG policy')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to saved model file')
    parser.add_argument('--env', type=str, required=True,
                       help='Environment name')
    parser.add_argument('--env-type', type=str, default=None,
                       help='Environment type (auto-detected if not specified)')
    parser.add_argument('--video-dir', type=str, default='./videos',
                       help='Directory to save videos')
    parser.add_argument('--num-episodes', type=int, default=5,
                       help='Number of episodes to record')
    parser.add_argument('--video-length', type=int, default=200,
                       help='Maximum length of each video')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create video directory if it doesn't exist
    os.makedirs(args.video_dir, exist_ok=True)
    
    generate_videos(
        model_path=args.model_path,
        env_name=args.env,
        video_dir=args.video_dir,
        num_episodes=args.num_episodes,
        video_length=args.video_length,
        env_type=args.env_type,
        seed=args.seed
    )


if __name__ == '__main__':
    main()