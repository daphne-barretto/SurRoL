#!/usr/bin/env python3

"""
Video Generation Script for .npz Demonstration Data

This script takes a .npz file containing demonstration data and generates
videos by replaying the stored actions in the environment.

Usage:
    python generate_video_from_npz.py --data_path path/to/data.npz --env_name PegTransferTwoBlocksTargetBlockOnlyNoColor-v0
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

def generate_video_from_npz(data_path, env_name=None, output_dir=None, 
                           max_episodes=10, fps=20, verbose=True):
    """
    Generate video from stored demonstration data
    
    Args:
        data_path: Path to .npz file containing demonstration data
        env_name: Environment name (auto-detected if None)
        output_dir: Directory to save videos (auto-generated if None)
        max_episodes: Maximum number of episodes to convert to video
        fps: Frames per second for video
        verbose: Whether to print progress
    """
    
    if verbose:
        print(f"🎬 GENERATING VIDEO FROM DEMONSTRATION DATA")
        print("="*60)
        print(f"   • Data file: {data_path}")
        print(f"   • Max episodes: {max_episodes}")
        print(f"   • FPS: {fps}")
    
    # Load data
    try:
        data = np.load(data_path, allow_pickle=True)
        obs_data = data['obs']
        acs_data = data['acs']
        
        if verbose:
            print(f"   • Episodes in data: {len(obs_data)}")
            print(f"   • Will process: {min(max_episodes, len(obs_data))} episodes")
    except Exception as e:
        print(f"   ❌ Failed to load data: {e}")
        return None
    
    # Auto-detect environment name if not provided
    if env_name is None:
        filename = os.path.basename(data_path)
        if 'TargetBlockOnlyNoColor' in filename:
            env_name = 'PegTransferTwoBlocksTargetBlockOnlyNoColor-v0'
        elif 'TargetBlockOnly' in filename:
            env_name = 'PegTransferTwoBlocksTargetBlockOnly-v0'
        elif 'TwoBlocks' in filename:
            # Try to guess the right environment
            env_name = 'PegTransferTwoBlocksNoColor-v0'  # Default fallback
        else:
            env_name = 'PegTransfer-v0'  # Ultimate fallback
        
        if verbose:
            print(f"   • Auto-detected environment: {env_name}")
    
    # Create environment
    try:
        # Try different render modes to find one that works
        env = None
        render_modes_to_try = ['rgb_array', 'human', None]
        
        for render_mode in render_modes_to_try:
            try:
                if render_mode is None:
                    env = gym.make(env_name)
                else:
                    env = gym.make(env_name, render_mode=render_mode)
                
                # Test if rendering works
                test_obs = env.reset()
                if isinstance(test_obs, tuple):
                    test_obs = test_obs[0]
                
                test_img = env.render()
                if hasattr(test_img, 'shape') and len(test_img.shape) == 3 and test_img.shape[0] > 0:
                    if verbose:
                        print(f"   ✅ Environment created successfully with render_mode='{render_mode}'")
                        print(f"   ✅ Render test successful: {test_img.shape}")
                    break
                else:
                    env.close()
                    env = None
                    
            except Exception as e:
                if env:
                    env.close()
                env = None
                if verbose and render_mode == render_modes_to_try[-1]:
                    print(f"   ⚠️  Render mode '{render_mode}' failed: {e}")
        
        if env is None:
            # Fallback: create env without render mode and try different render calls
            env = gym.make(env_name)
            if verbose:
                print(f"   ⚠️  Using fallback environment creation")
        
    except Exception as e:
        print(f"   ❌ Failed to create environment: {e}")
        return None
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"/home/ubuntu/project/SurRoL/surrol/data/video/from_npz_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"   • Output directory: {output_dir}")
    
    # Process episodes
    successful_videos = 0
    all_episode_images = []
    
    num_episodes_to_process = min(max_episodes, len(obs_data))
    
    for episode_idx in range(num_episodes_to_process):
        try:
            if verbose:
                print(f"   🎥 Processing episode {episode_idx + 1}/{num_episodes_to_process}...")
            
            episode_obs = obs_data[episode_idx]
            episode_acs = acs_data[episode_idx]
            
            # Reset environment
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                env_obs, _ = reset_result  # New gymnasium API
            else:
                env_obs = reset_result  # Old gym API
            episode_images = []
            
            # Get the actual environment for state setting if needed
            actual_env = env
            while hasattr(actual_env, '_env') or hasattr(actual_env, 'env'):
                if hasattr(actual_env, '_env'):
                    actual_env = actual_env._env
                elif hasattr(actual_env, 'env'):
                    actual_env = actual_env.env
                else:
                    break
            
            # Capture initial frame
            img = env.render()
            if verbose and episode_idx == 0:  # Debug first episode
                print(f"      🔧 Render result type: {type(img)}")
                if img is not None:
                    if hasattr(img, 'shape'):
                        print(f"      🔧 Image shape: {img.shape}")
                    print(f"      🔧 Image dtype: {getattr(img, 'dtype', 'unknown')}")
            
            # Try alternative rendering methods if the first one fails
            if img is None or (hasattr(img, 'shape') and img.shape == (0,)):
                # Try different render methods
                alternative_methods = [
                    lambda: env.render('rgb_array'),
                    lambda: getattr(env, 'render', lambda: None)('rgb_array'),
                    lambda: getattr(env.unwrapped, 'render', lambda: None)('rgb_array') if hasattr(env, 'unwrapped') else None,
                ]
                
                for method in alternative_methods:
                    try:
                        alt_img = method()
                        if alt_img is not None and hasattr(alt_img, 'shape') and len(alt_img.shape) == 3:
                            img = alt_img
                            break
                    except:
                        continue
            
            if img is not None:
                episode_images.append(img)
            
            # Replay stored actions
            for step, action in enumerate(episode_acs):
                # Take action
                step_result = env.step(action)
                if len(step_result) == 5:
                    # New gymnasium API: obs, reward, done, truncated, info
                    env_obs, reward, done, truncated, info = step_result
                else:
                    # Old gym API: obs, reward, done, info
                    env_obs, reward, done, info = step_result
                    truncated = False
                
                # Capture frame
                img = env.render()
                
                # Try alternative rendering methods if the first one fails
                if img is None or (hasattr(img, 'shape') and img.shape == (0,)):
                    # Try different render methods
                    alternative_methods = [
                        lambda: env.render('rgb_array'),
                        lambda: getattr(env, 'render', lambda: None)('rgb_array'),
                        lambda: getattr(env.unwrapped, 'render', lambda: None)('rgb_array') if hasattr(env, 'unwrapped') else None,
                    ]
                    
                    for method in alternative_methods:
                        try:
                            alt_img = method()
                            if alt_img is not None and hasattr(alt_img, 'shape') and len(alt_img.shape) == 3:
                                img = alt_img
                                break
                        except:
                            continue
                
                if img is not None:
                    episode_images.append(img)
                
                # Check if episode ended
                if done or truncated:
                    if verbose and info.get('is_success', False):
                        print(f"      ✅ Episode {episode_idx + 1} completed successfully at step {step + 1}")
                    break
            
            # Store episode images
            if len(episode_images) > 0:
                all_episode_images.append(episode_images)
                successful_videos += 1
            
        except Exception as e:
            if verbose:
                print(f"      ❌ Episode {episode_idx + 1} failed: {e}")
            continue
    
    env.close()
    
    if verbose:
        print(f"   📊 Successfully processed: {successful_videos}/{num_episodes_to_process} episodes")
    
    # Generate videos
    if successful_videos == 0:
        print("   ❌ No episodes successfully processed for video generation")
        return None
    
    if verbose:
        print(f"   🎬 Generating videos...")
    
    # Individual episode videos
    video_paths = []
    for i, episode_images in enumerate(all_episode_images):
        video_name = f"episode_{i+1:03d}.mp4"
        video_path = os.path.join(output_dir, video_name)
        
        try:
            if verbose:
                print(f"      🎬 Creating video for episode {i+1}: {len(episode_images)} frames")
            
            writer = imageio.get_writer(video_path, fps=fps)
            for frame_idx, img in enumerate(episode_images):
                try:
                    # Ensure image is in the right format for imageio
                    if hasattr(img, 'shape') and len(img.shape) == 3:
                        # Convert to uint8 if needed
                        if img.dtype != np.uint8:
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)
                        writer.append_data(img)
                    else:
                        if verbose:
                            print(f"        ⚠️  Skipping frame {frame_idx}: invalid shape {getattr(img, 'shape', 'unknown')}")
                except Exception as frame_error:
                    if verbose:
                        print(f"        ❌ Frame {frame_idx} error: {frame_error}")
                        print(f"        🔧 Frame type: {type(img)}, shape: {getattr(img, 'shape', 'unknown')}")
                    continue
            writer.close()
            video_paths.append(video_path)
            
            if verbose:
                print(f"      ✅ Episode {i+1} video: {video_name} ({len(episode_images)} frames)")
                
        except Exception as e:
            if verbose:
                print(f"      ❌ Failed to create video for episode {i+1}: {e}")
                import traceback
                traceback.print_exc()
    
    # Combined video of all episodes
    combined_video_name = f"all_episodes_combined.mp4"
    combined_video_path = os.path.join(output_dir, combined_video_name)
    
    try:
        if verbose:
            print(f"      🎬 Creating combined video...")
        
        writer = imageio.get_writer(combined_video_path, fps=fps)
        total_frames = 0
        for ep_idx, episode_images in enumerate(all_episode_images):
            for frame_idx, img in enumerate(episode_images):
                try:
                    # Ensure image is in the right format for imageio
                    if hasattr(img, 'shape') and len(img.shape) == 3:
                        # Convert to uint8 if needed
                        if img.dtype != np.uint8:
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)
                        writer.append_data(img)
                        total_frames += 1
                    else:
                        if verbose:
                            print(f"        ⚠️  Skipping episode {ep_idx+1} frame {frame_idx}: invalid shape {getattr(img, 'shape', 'unknown')}")
                except Exception as frame_error:
                    if verbose:
                        print(f"        ❌ Episode {ep_idx+1} Frame {frame_idx} error: {frame_error}")
                    continue
            # Add a few black frames between episodes
            if len(episode_images) > 0:
                try:
                    # Create black frame with same dimensions as first valid frame
                    valid_frames = [img for img in episode_images if hasattr(img, 'shape') and len(img.shape) == 3]
                    if valid_frames:
                        black_frame = np.zeros_like(valid_frames[0], dtype=np.uint8)
                        for _ in range(10):  # 0.5 second gap at 20fps
                            writer.append_data(black_frame)
                            total_frames += 1
                except Exception as black_frame_error:
                    if verbose:
                        print(f"        ⚠️  Skipping black frames: {black_frame_error}")
        writer.close()
        
        if verbose:
            print(f"      ✅ Combined video: {combined_video_name} ({total_frames} frames)")
    
    except Exception as e:
        if verbose:
            print(f"      ❌ Failed to create combined video: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if verbose:
        print(f"\n📊 VIDEO GENERATION SUMMARY:")
        print(f"   • Input data: {data_path}")
        print(f"   • Environment: {env_name}")
        print(f"   • Episodes processed: {successful_videos}")
        print(f"   • Individual videos: {len(video_paths)}")
        print(f"   • Combined video: {'✅' if os.path.exists(combined_video_path) else '❌'}")
        print(f"   • Output directory: {output_dir}")
        print(f"   • Video FPS: {fps}")
    
    return {
        'output_dir': output_dir,
        'individual_videos': video_paths,
        'combined_video': combined_video_path if os.path.exists(combined_video_path) else None,
        'episodes_processed': successful_videos,
        'total_episodes': len(obs_data)
    }

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Generate videos from .npz demonstration data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to .npz data file')
    parser.add_argument('--env_name', type=str, help='Environment name (auto-detected if not provided)')
    parser.add_argument('--output_dir', type=str, help='Output directory (auto-generated if not provided)')
    parser.add_argument('--max_episodes', type=int, default=10, help='Maximum episodes to convert to video')
    parser.add_argument('--fps', type=int, default=20, help='Video frames per second')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return 1
    
    result = generate_video_from_npz(
        data_path=args.data_path,
        env_name=args.env_name,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
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