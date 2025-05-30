import os
import time
import numpy as np

import pybullet as p

import argparse
import time

from surrol.tasks.psm_env import PsmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle
)
from surrol.const import ASSET_DIR_PATH

import gym


class SurgicalRoboticsDiscretize(gym.Wrapper):
    '''Maps discrete actions to continuous robot movements.
    '''
    def __init__(self, env, action_step_size=0.01, render_every_step=False):
        super().__init__(env)
        self.action_step_size = action_step_size
        self._render_every_step = render_every_step
        
        # It's 5 for Peg Transfer.
        # print(f"Environment action space dimension: {env.action_space.shape[0]}")
        
        self.action_mapping = {
            0: [self.action_step_size, 0, 0, 0, 0],      # Move +X
            1: [-self.action_step_size, 0, 0, 0, 0],     # Move -X
            2: [0, self.action_step_size, 0, 0, 0],      # Move +Y
            3: [0, -self.action_step_size, 0, 0, 0],     # Move -Y
            4: [0, 0, self.action_step_size, 0, 0],      # Move +Z
            5: [0, 0, -self.action_step_size, 0, 0],     # Move -Z
            6: [0, 0, 0, 0, 1.0],                        # Close gripper
            7: [0, 0, 0, 0, -1.0],                       # Open gripper
        }
        
        # Update action space to discrete
        self.action_space = gym.spaces.Discrete(len(self.action_mapping))

    def reset(self):
        """
        Reset the environment and return block-conditioned observation.
        """
        self.env.reset()
        return self.env.get_block_conditioned_observation()

    def step(self, action):
        """
        Takes an action in the SurRoL environments.

        Args:
            action (int): discrete action in [0, NUM_ACTIONS-1]

        Returns:
            next_state (ndarray): block-conditioned observation
            reward (float): reward from the environment
            done (bool): whether the episode is done
            info (dict): additional info from the environment
        """
        assert action in self.action_mapping
        
        # maps discrete action to continuous action
        action_continuous = np.array(self.action_mapping[action], dtype=np.float32)
        assert len(action_continuous) == self.env.action_space.shape[0]
        
        # take the action
        ob, reward, done, info = self.env.step(action_continuous)
        
        # if rendering is turned on, render the environment
        if self._render_every_step:
            self.env.render()
        
        next_state = self.env.get_block_conditioned_observation()
        
        return next_state, reward, done, info

    def set_target_block(self, block_id):
        """Set the block that the agent should target."""
        return self.env.set_target_block(block_id)

    def get_block_conditioned_observation(self):
        return self.env.get_block_conditioned_observation()

    @property
    def num_actions(self):
        return len(self.action_mapping)


class BlockConditionedPegTransfer(PsmEnv):
    POSE_BOARD = ((0.55, 0, 0.6861), (0, 0, 0))  # 0.675 + 0.011 + 0.001
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.686, 0.745))
    SCALING = 5.

    # TODO: grasp is sometimes not stable; check how to fix it

    def __init__(self, render_mode=None, num_blocks=4):
        self.num_blocks = num_blocks
        self.current_target_block_id = 0  # which block is targeted (0 to num_blocks - 1)
        self.block_colors = [
            (1.0, 0.0, 0.0, 1.0),    # Red - Block 0
            (0.0, 1.0, 0.0, 1.0),    # Green - Block 1
            (0.0, 0.0, 1.0, 1.0),    # Blue - Block 2
            (1.0, 1.0, 0.0, 1.0),    # Yellow - Block 3
        ]
        self.block_ids = []  # store pybullet IDs of blocks
        self.block_positions = []  # store initial positions of blocks
        self.target_peg_positions = []  # store target peg positions for each block

        super().__init__(render_mode=render_mode)
        
    def _env_setup(self):
        super(BlockConditionedPegTransfer, self)._env_setup()
        self.has_object = True

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False

        # peg board
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'peg_board/peg_board.urdf'),
                            np.array(self.POSE_BOARD[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_BOARD[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)
        
        # setup the pegs: target pegs (0-5) and source pegs (6-11)
        self._pegs = np.arange(12)
        source_pegs = self._pegs[6:6+self.num_blocks]  # pegs where blocks start
        target_pegs = self._pegs[0:self.num_blocks]    # pegs where blocks should go aka goals
        
        self.block_ids = []
        self.block_positions = []
        self.target_peg_positions = []

        # create blocks on the source pegs
        for i in range(self.num_blocks):
            source_peg_idx = source_pegs[i]
            target_peg_idx = target_pegs[i]

            pos, orn = get_link_pose(self.obj_ids['fixed'][1], source_peg_idx)
            yaw = (np.random.rand() - 0.5) * np.deg2rad(60)
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'block/block.urdf'),
                                np.array(pos) + np.array([0, 0, 0.03]),
                                p.getQuaternionFromEuler((0, 0, yaw)),
                                useFixedBase=False,
                                globalScaling=self.SCALING)
            self.obj_ids['rigid'].append(obj_id)
            
            # set block color
            color = self.block_colors[i % len(self.block_colors)]
            p.changeVisualShape(obj_id, -1, rgbaColor=color)
            
            self.block_ids.append(obj_id)
            self.block_positions.append(np.array(pos))
            
            # save target position / goal for this block
            target_pos = get_link_pose(self.obj_ids['fixed'][1], target_peg_idx)[0]
            self.target_peg_positions.append(np.array(target_pos))
        
        # set the initial target block to block 0
        self.current_target_block_id = 0
        self.obj_id = self.block_ids[self.current_target_block_id]
        self.obj_link1 = 1
        self.goal = self.target_peg_positions[self.current_target_block_id].copy()

    def set_target_block(self, block_id):
        """Set which block the agent should pick up (0 to num_blocks-1)."""
        assert block_id >= 0 and block_id <= self.num_blocks
        self.current_target_block_id = block_id
        
        # update the goal to be the target peg for this specific block
        if len(self.target_peg_positions) > block_id:
            self.goal = self.target_peg_positions[block_id].copy()

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        return np.logical_and(
            goal_distance(achieved_goal[..., :2], desired_goal[..., :2]) < 5e-3 * self.SCALING,
            np.abs(achieved_goal[..., -1] - desired_goal[..., -1]) < 4e-3 * self.SCALING
        ).astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        """Return the target peg position for the current target block."""
        return self.target_peg_positions[self.current_target_block_id].copy()

    def _get_current_position_of_target_block(self):
        """Get the current position of the target block."""
        assert self.current_target_block_id < len(self.block_ids)
        pos, _ = get_link_pose(self.block_ids[self.current_target_block_id], -1)
        return np.array(pos)

    def get_block_conditioned_observation(self):
        """Get observation with block info concatenated."""
        base_obs = self._get_obs()
        
        # one-hot encoding of target block
        block_encoding = np.zeros(self.num_blocks)
        block_encoding[self.current_target_block_id] = 1.0
        
        # Combine all features
        full_obs = np.concatenate([
            base_obs['observation'],      # Base robot state
            block_encoding               # One-hot encoding of target block
        ], dtype=np.float32)
        
        # print(f"full_obs.shape: {full_obs.shape}")
        return full_obs

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
            
        target_block_id = self.block_ids[self.current_target_block_id]
        
        self._waypoints = [None, None, None, None, None, None]  # six waypoints
        pos_obj, orn_obj = get_link_pose(target_block_id, self.obj_link1)
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, 0.5])  # above object
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, -0.5])  # lift up

        # move to target peg for this specific block
        goal = self.target_peg_positions[self.current_target_block_id]
        self._waypoints[4] = np.array([goal[0], goal[1], 
                                       goal[2] + 0.045 * self.SCALING, yaw, -0.5])  # above goal
        self._waypoints[5] = np.array([goal[0], goal[1], 
                                       goal[2] + 0.015 * self.SCALING, yaw, 0.5])  # release

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        pose = get_link_pose(self.obj_id, -1)
        return pose[0][2] > self.goal[2] + 0.01 * self.SCALING

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # six waypoints executed in sequential order
        action = np.zeros(5)
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.7
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 2e-3 and np.abs(delta_yaw) < np.deg2rad(2.):
                self._waypoints[i] = None
            break

        return action
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Block-Conditioned Peg Transfer in Surgical Robotics')
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--action_step_size', type=float, default=0.01, help='Step size for discrete actions')
    args = parser.parse_args()

    env = BlockConditionedPegTransfer(render_mode='human', num_blocks=args.num_blocks)
    env = SurgicalRoboticsDiscretize(env, action_step_size=args.action_step_size)
    
    env.test()
    env.close()
    time.sleep(2)
