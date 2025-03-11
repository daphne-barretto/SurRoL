import os
import time
import numpy as np
import pandas as pd

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
)
from surrol.const import ASSET_DIR_PATH


class NeedleReachCurriculumLearning(PsmEnv):
    """
    Refer to Gym FetchReach
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/reach.py
    """
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.

    def _env_setup(self):
        super(NeedleReachCurriculumLearning, self)._env_setup()
        self.has_object = False

        workspace_limits = self.workspace_limits1

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(10, 10, 10))
        self.obj_ids['fixed'].append(obj_id)  # 1

        # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        needle_pos = (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,
                        workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                        workspace_limits[2][0] + 0.01)
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            needle_pos,
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

        # robot
        final_initial_robot_pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        # set robot position (start state) based on previous evaluation data
        # ================================================
        alg = 'hercl' # 'ddpgcl' or 'hercl'
        if alg == 'ddpgcl':
            file_path = './logs/ddpgcl/NeedleReachCurriculumLearning-1e5_0/progress.csv'
            try:
                data = pd.read_csv(file_path)
                if data.empty:
                    # set success_rate to data frame with 0.0
                    most_recent_success_rate = 0.0
                else:
                    success_rate = data['eval/return']
                    most_recent_success_rate = success_rate.iloc[-1]
            except pd.errors.EmptyDataError:
                most_recent_success_rate = 0.0
        elif alg == 'hercl':
            file_path = './logs/hercl/NeedleReachCurriculumLearning-1e5_0/progress.csv'
            try:
                data = pd.read_csv(file_path)
                print("Reading data from file:", file_path)
                if data.empty:
                    epoch = 0
                    print('Empty data frame')
                else:
                    data_epoch = data['epoch']
                    epoch = data_epoch.iloc[-1]
                print('Epoch:', epoch)
            except pd.errors.EmptyDataError:
                epoch = 0
        total_epochs = 50
        training_progress = epoch * 1.0 / total_epochs
        # set robot position to be between final_initial_pos and needle_pos based on training progress
        # so that the robot position moves from close to the needle to far away from the needle as training progresses
        robot_pos = np.array(final_initial_robot_pos) * training_progress + np.array(needle_pos) * (1 - training_progress)
        print('final_initial_robot_pos:', final_initial_robot_pos)
        print('needle_pos:', needle_pos)
        print('robot_pos:', robot_pos)
        # ================================================
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((robot_pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = True

    def _set_action(self, action: np.ndarray):
        action[3] = 0  # no yaw change
        super(NeedleReachCurriculumLearning, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        goal = np.array([pos[0], pos[1], pos[2] + 0.005 * self.SCALING])
        return goal.copy()

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        delta_pos = (obs['desired_goal'] - obs['achieved_goal']) / 0.01
        if np.linalg.norm(delta_pos) < 1.5:
            delta_pos.fill(0)
        if np.abs(delta_pos).max() > 1:
            delta_pos /= np.abs(delta_pos).max()
        delta_pos *= 0.3

        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0., 0.])
        return action


if __name__ == "__main__":
    env = NeedleReachCurriculumLearning(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
