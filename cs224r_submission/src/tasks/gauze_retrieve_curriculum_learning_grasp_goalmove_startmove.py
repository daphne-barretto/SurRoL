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


class GauzeRetrieveCurriculumLearningGraspGoalMoveStartMove(PsmEnv):
    """
    Refer to Gym FetchPickAndPlace
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/pick_and_place.py
    """
    POSE_TRAY = ((0.55, 0, 0.6781), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.

    # TODO: grasp is sometimes not stable; check how to fix it

    def _env_setup(self):
        super(GauzeRetrieveCurriculumLearningGraspGoalMoveStartMove, self)._env_setup()
        self.has_object = True
        self._waypoint_goal = True
        # self._contact_approx = True  # mimic the dVRL setting, prove nothing?

        workspace_limits = self.workspace_limits1

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1
        p.changeVisualShape(obj_id, -1, rgbaColor=(225 / 255, 225 / 255, 225 / 255, 1))

        gauze_pos = (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,
                     workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                     workspace_limits[2][0] + 0.01)
        final_initial_robot_pos = (workspace_limits[0][0],
                                    workspace_limits[1][1],
                                    (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        # goal_pos = self._sample_goal()
        
        # set robot position (start state) based on previous evaluation data
        # ================================================
        alg = 'hercl' # 'ddpgcl' or 'hercl'
        if alg == 'ddpgcl':
            file_path = './logs/ddpgcl/GauzeRetrieveCurriculumLearningGraspGoalMoveStartMove-1e5_0/progress.csv'
            try:
                data = pd.read_csv(file_path)
                if data.empty:
                    epoch = 0
                    train_success = 0
                else:
                    data_epoch = data['total/epochs'] + 1
                    epoch = data_epoch.iloc[-1]
                    data_train_success = data['train/success_rate']
                    train_success = data_train_success.iloc[-1]
            except pd.errors.EmptyDataError:
                epoch = 0
                train_success = 0
        elif alg == 'hercl':
            file_path = './logs/hercl/GauzeRetrieveCurriculumLearningGraspGoalMoveStartMove-1e5_0/progress.csv'
            try:
                data = pd.read_csv(file_path)
                if data.empty:
                    epoch = 0
                    train_success = 0
                else:
                    data_epoch = data['epoch'] + 1
                    epoch = data_epoch.iloc[-1]
                    data_train_success = data['train/success_rate']
                    train_success = data_train_success.iloc[-1]
            except pd.errors.EmptyDataError:
                epoch = 0
                train_success =0 
        total_epochs = 50
        training_progress = (epoch * 1.0 / total_epochs) #* train_success
        self.training_progress = training_progress
        # print("training progress is ", training_progress)
        
        grasp_curriculum_hyperparam = 0.5 # how long to train with gauze already in the psm's jaw
        self.grasp_curriculum_hyperparam = grasp_curriculum_hyperparam
        if training_progress < grasp_curriculum_hyperparam:
            grasp_progress = training_progress / grasp_curriculum_hyperparam
            # robot_pos = final_initial_robot_pos # np.array(gauze_pos) * grasp_progress + np.array(goal_pos) * (1 - grasp_progress)
            robot_pos =  np.array(gauze_pos) #* grasp_progress + np.array(final_initial_robot_pos) * (1 - grasp_progress)
            # place the gauze in the psm's jaw
            gauze_pos = (robot_pos[0], robot_pos[1], robot_pos[2] - (-0.0007 + 0.0102) * self.SCALING)
        else:
            non_grasp_progress = (training_progress - grasp_curriculum_hyperparam) / (1 - grasp_curriculum_hyperparam)
            robot_pos = np.array(final_initial_robot_pos) * non_grasp_progress + np.array(gauze_pos) * (1 - non_grasp_progress)
        self.gauze_pos = gauze_pos
        # print('final_initial_robot_pos:', final_initial_robot_pos)
        # print('gauze_pos:', gauze_pos)
        # print('robot_pos:', robot_pos)
        # ================================================
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((robot_pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)

        # gauze
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                            gauze_pos,
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(0, 0, 0))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1

        if training_progress < grasp_curriculum_hyperparam:
            self.psm1.close_jaw()

    def _set_action(self, action: np.ndarray):
        action[3] = 0  # no yaw change
        # if action[4] > 0:
        #     print("JAW OPEN")
        super(GauzeRetrieveCurriculumLearningGraspGoalMoveStartMove, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.02 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.02 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.03 * self.SCALING])
        if self.training_progress < self.grasp_curriculum_hyperparam:
            rat = self.training_progress/self.grasp_curriculum_hyperparam
        else:
            rat = self.training_progress-self.grasp_curriculum_hyperparam/(1-self.grasp_curriculum_hyperparam)
        # final_goal_pos = np.array(goal) * rat + np.array(self.robot_pos) * (1 - rat)
        final_goal_pos = np.array(goal) * rat + np.array(self.gauze_pos) * (1 - rat)
        # print ("gauze_pos is,", self.gauze_pos)
        # print ("final_goal_pos is,", final_goal_pos)
        return final_goal_pos.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None, None]  # five waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, 0., 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., -0.5])  # grasp
        self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, 0., -0.5])  # grasp
        self._waypoints[4] = np.array([self.goal[0], self.goal[1],
                                       self.goal[2] + 0.0102 * self.SCALING, 0., -0.5])  # lift up

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped object to make it stable
        pose = get_link_pose(self.obj_id, self.obj_link1)
        return pose[0][2] > self._waypoint_z_init + 0.0025 * self.SCALING
        # return True  # mimic the dVRL setting

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.6
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0., waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4:
                self._waypoints[i] = None
            break

        return action


if __name__ == "__main__":
    env = GauzeRetrieveCurriculumLearningGraspGoalMoveStartMove(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
