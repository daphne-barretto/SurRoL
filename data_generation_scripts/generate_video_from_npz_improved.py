#!/usr/bin/env python3

"""
IMPROVED Video Generation Script for .npz Demonstration Data

This version generates NEW demonstration episodes using the oracle policy
to ensure successful task completion, rather than replaying stored actions
which may not work due to state differences.

Usage:
    DISPLAY=:99 PYBULLET_EGL=1 xvfb-run -a python generate_video_from_npz_improved.py --env_name PegTransferTwoBlocksTargetBlockOnlyNoColor-v0 --num_episodes 5
"""

import os
import sys
import numpy as np
import imageio
import argparse
from datetime import datetime

# Add SurRoL to path
project_root = os.path.dirname(os.path.abspath(__file__))
surrol_path = os.path.join(project_root, 'SurRoL')
if surrol_path not in sys.path:
    sys.path.insert(0, surrol_path)

import gymnasium as gym
import surrol.gym

def generate_fresh_demo_videos(env_name, output_dir=None, num_episodes=5, 
                              max_steps=50, fps=20, verbose=True):
    """
    Generate fresh demonstration videos using oracle policy
    
    Args:
        env_name: Environment name
        output_dir: Directory to save videos (auto-generated if None)
        num_episodes: Number of episodes to generate
        max_steps: Maximum steps per episode
        fps: Frames per second for video
        verbose: Whether to print progress
    """
    
    if verbose:
        print(f"🎬 GENERATING FRESH DEMONSTRATION VIDEOS")
        print("="*60)
        print(f"   • Environment: {env_name}")
        print(f"   • Episodes: {num_episodes}")
        print(f"   • Max steps: {max_steps}")
        print(f"   • FPS: {fps}")
    
    # Create environment with fallback render mode detection
    env = None
    render_modes_to_try = [None, 'rgb_array', 'human']
    
    for render_mode in render_modes_to_try:
        try:
            if render_mode is None:
                env = gym.make(env_name)
            else:
                env = gym.make(env_name, render_mode=render_mode)
            
            # Test rendering
            env.reset()
            test_img = env.render()
            if hasattr(test_img, 'shape') and len(test_img.shape) == 3 and test_img.shape[0] > 0:
                if verbose:
                    print(f"   ✅ Environment created with render_mode='{render_mode}'")
                    print(f"   ✅ Render test: {test_img.shape}")
                break
            else:
                env.close()
                env = None
                
        except Exception as e:
            if env:
                env.close()
            env = None
    
    if env is None:
        print(f"   ❌ Could not create working environment")
        return None
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"/home/ubuntu/project/SurRoL/surrol/data/video/fresh_demos_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"   • Output directory: {output_dir}")
    
    # Get access to the underlying environment for oracle actions
    actual_env = env
    while hasattr(actual_env, '_env') or hasattr(actual_env, 'env'):
        if hasattr(actual_env, '_env'):
            actual_env = actual_env._env
        elif hasattr(actual_env, 'env'):
            actual_env = actual_env.env
        else:
            break
    
    # Generate episodes
    successful_videos = 0
    all_episode_images = []
    episode_results = []
    
    for episode in range(num_episodes):
        try:
            if verbose:
                print(f"   🎥 Generating episode {episode + 1}/{num_episodes}...")
            
            # Reset environment
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            
            episode_images = []
            episode_success = False
            episode_steps = 0
            
            # Capture initial frame
            img = env.render()
            if img is not None and hasattr(img, 'shape') and len(img.shape) == 3:
                episode_images.append(img)
            
            # Run episode using oracle policy
            for step in range(max_steps):
                try:
                    # Get oracle action
                    if hasattr(actual_env, 'get_oracle_action'):
                        action = actual_env.get_oracle_action(obs)
                    else:
                        # Fallback to random action
                        action = env.action_space.sample()
                    
                    # Take action
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        obs, reward, done, truncated, info = step_result
                    else:
                        obs, reward, done, info = step_result
                        truncated = False
                    
                    episode_steps += 1
                    
                    # Capture frame
                    img = env.render()
                    if img is not None and hasattr(img, 'shape') and len(img.shape) == 3:
                        episode_images.append(img)
                    
                    # Check success
                    if info.get('is_success', False):
                        episode_success = True
                        if verbose:
                            print(f"      ✅ Episode {episode + 1} succeeded at step {step + 1}")
                        break
                    
                    # Check termination
                    if done or truncated:
                        break
                        
                except Exception as step_error:
                    if verbose:
                        print(f"      ⚠️  Step error: {step_error}")
                    break
            
            # Store episode data
            episode_info = {
                'episode': episode + 1,
                'success': episode_success,
                'steps': episode_steps,
                'frames': len(episode_images)
            }
            episode_results.append(episode_info)
            
            if len(episode_images) > 0:
                all_episode_images.append(episode_images)
                successful_videos += 1
                
                if verbose:
                    status = "✅ SUCCESS" if episode_success else "❌ FAILED"
                    print(f"      {status} | Steps: {episode_steps} | Frames: {len(episode_images)}")
            
        except Exception as e:
            if verbose:
                print(f"      ❌ Episode {episode + 1} failed: {e}")
            continue
    
    env.close()
    
    if verbose:
        print(f"   📊 Generated: {successful_videos}/{num_episodes} episodes")
        success_count = sum(1 for ep in episode_results if ep['success'])
        print(f"   🎯 Success rate: {success_count}/{len(episode_results)} ({success_count/len(episode_results)*100:.1f}%)")
    
    # Generate videos
    if successful_videos == 0:
        print("   ❌ No episodes generated for video creation")
        return None
    
    if verbose:
        print(f"   🎬 Creating video files...")
    
    # Individual episode videos
    video_paths = []
    for i, episode_images in enumerate(all_episode_images):
        ep_result = episode_results[i] if i < len(episode_results) else {'success': False}
        success_marker = "_SUCCESS" if ep_result.get('success', False) else "_FAILED"
        video_name = f"episode_{i+1:03d}{success_marker}.mp4"
        video_path = os.path.join(output_dir, video_name)
        
        try:
            writer = imageio.get_writer(video_path, fps=fps)
            for img in episode_images:
                if hasattr(img, 'shape') and len(img.shape) == 3:
                    # Ensure proper format
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    writer.append_data(img)
            writer.close()
            video_paths.append(video_path)
            
            if verbose:
                print(f"      ✅ {video_name} ({len(episode_images)} frames)")
                
        except Exception as e:
            if verbose:
                print(f"      ❌ Failed to create {video_name}: {e}")
    
    # Combined video (successful episodes only)
    successful_episodes = [i for i, ep in enumerate(episode_results) if ep.get('success', False)]
    if successful_episodes and len(successful_episodes) > 0:
        combined_video_name = f"successful_episodes_combined.mp4"
        combined_video_path = os.path.join(output_dir, combined_video_name)
        
        try:
            writer = imageio.get_writer(combined_video_path, fps=fps)
            total_frames = 0
            
            for ep_idx in successful_episodes:
                if ep_idx < len(all_episode_images):
                    episode_images = all_episode_images[ep_idx]
                    for img in episode_images:
                        if hasattr(img, 'shape') and len(img.shape) == 3:
                            if img.dtype != np.uint8:
                                if img.max() <= 1.0:
                                    img = (img * 255).astype(np.uint8)
                                else:
                                    img = img.astype(np.uint8)
                            writer.append_data(img)
                            total_frames += 1
                    
                    # Add separator frames
                    if len(episode_images) > 0 and ep_idx < successful_episodes[-1]:
                        black_frame = np.zeros_like(episode_images[0], dtype=np.uint8)
                        for _ in range(fps//2):  # 0.5 second gap
                            writer.append_data(black_frame)
                            total_frames += 1
            
            writer.close()
            
            if verbose:
                print(f"      ✅ {combined_video_name} ({total_frames} frames, {len(successful_episodes)} episodes)")
        
        except Exception as e:
            if verbose:
                print(f"      ❌ Failed to create combined video: {e}")
    
    # Summary
    if verbose:
        print(f"\n📊 VIDEO GENERATION SUMMARY:")
        print(f"   • Environment: {env_name}")
        print(f"   • Episodes generated: {successful_videos}")
        print(f"   • Successful episodes: {success_count}")
        print(f"   • Individual videos: {len(video_paths)}")
        print(f"   • Output directory: {output_dir}")
        print(f"   • Video FPS: {fps}")
    
    return {
        'output_dir': output_dir,
        'individual_videos': video_paths,
        'episode_results': episode_results,
        'successful_episodes': success_count,
        'total_episodes': successful_videos
    }

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Generate fresh demonstration videos using oracle policy')
    parser.add_argument('--env_name', type=str, required=True, help='Environment name')
    parser.add_argument('--output_dir', type=str, help='Output directory (auto-generated if not provided)')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to generate')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum steps per episode')
    parser.add_argument('--fps', type=int, default=20, help='Video frames per second')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    result = generate_fresh_demo_videos(
        env_name=args.env_name,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        verbose=not args.quiet
    )
    
    if result:
        print(f"\n✅ Video generation completed successfully!")
        return 0
    else:
        print(f"\n❌ Video generation failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 