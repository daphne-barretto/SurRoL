import os
import time
import numpy as np
import pandas as pd

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
    step
)
from surrol.utils.robotics import get_matrix_from_pose_2d
from surrol.const import ASSET_DIR_PATH


class GauzeRetrieveCurriculumLearningGraspGoalMove(PsmEnv):
    """
    Refer to Gym FetchPickAndPlace
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/pick_and_place.py
    """
    POSE_TRAY = ((0.55, 0, 0.6781), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.

    # TODO: grasp is sometimes not stable; check how to fix it

    def _env_setup(self):
        super(GauzeRetrieveCurriculumLearningGraspGoalMove, self)._env_setup()
        self.has_object = True
        self._waypoint_goal = True
        # self._contact_approx = True  # mimic the dVRL setting, prove nothing?

       
        # robot
        psm = self.psm1
        workspace_limits = self.workspace_limits1

        pos = (workspace_limits[0].mean(),
               workspace_limits[1].mean(),
               workspace_limits[2].mean())
        # orn = p.getQuaternionFromEuler(np.deg2rad([0, np.random.uniform(-45, -135), -90]))
        orn = p.getQuaternionFromEuler(np.deg2rad([0, -90, -90]))  # reduce difficulty

        # psm.reset_joint(self.QPOS_PSM1)
        joint_positions = psm.inverse_kinematics((pos, orn), psm.EEF_LINK_INDEX)
        psm.reset_joint(joint_positions)

        self.block_gripper = False  # set the constraint
        
        workspace_limits = self.workspace_limits1

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1
        p.changeVisualShape(obj_id, -1, rgbaColor=(225 / 255, 225 / 255, 225 / 255, 1))

        # gauze
        # gauze_pos = (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,
        #              workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
        #              workspace_limits[2][0] + 0.01)
        # self.gauze_pos =gauze_pos
        limits_span = (workspace_limits[:, 1] - workspace_limits[:, 0]) / 3
        sample_space = workspace_limits.copy()
        sample_space[:, 0] += limits_span
        sample_space[:, 1] -= limits_span
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                            (0.01 * self.SCALING, 0, 0),
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(0, 0, 0))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1

        # get eopch from saved data
        # ================================================
        alg = 'hercl' # 'ddpgcl' or 'hercl'
        if alg == 'ddpgcl':
            file_path = './logs/ddpgcl/GauzeRetrieveCurriculumLearningSmarter-1e5_0/progress.csv'
            try:
                data = pd.read_csv(file_path)
                if data.empty:
                   epoch = 0
                else:
                    data_epoch = data['total/epochs']
                    epoch = data_epoch.iloc[-1]
            except pd.errors.EmptyDataError:
                epoch = 0
        elif alg == 'hercl':
            file_path = './logs/hercl/GauzeRetrieveCurriculumLearningSmarter-1e5_0/progress.csv'
            try:
                data = pd.read_csv(file_path)
                if data.empty:
                    epoch = 0
                else:
                    data_epoch = data['epoch']
                    epoch = data_epoch.iloc[-1]
            except pd.errors.EmptyDataError:
                epoch = 0
        total_epochs = 50
        training_progress = epoch * 1.0 / total_epochs
        self.training_progress = training_progress
        # ================================================
        # robot
        robot_pos = (workspace_limits[0][0],
                                    workspace_limits[1][1],
                                    (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        # set robot position to be between final_initial_pos and needle_pos based on training progress
        # final_initial_robot_pos = (workspace_limits[0][0],
        #                             workspace_limits[1][1],
        #                             (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        # set robot position to be between final_initial_pos and needle_pos based on training progress
        # so that the robot position moves from close to the needle to far away from the needle as training progresses
        # gauze_pos = self.obj_ids['rigid'][0]
        # robot_pos = np.array(final_initial_robot_pos) * training_progress + np.array(gauze_pos) * (1 - training_progress)
        # robot_pos[2]+= 0.055
        # self.robot_pos = robot_pos
        # print('final_initial_robot_pos:', final_initial_robot_pos)
        # print('needle_pos:', gauze_pos)
        # print('robot_pos:', robot_pos)
        # ================================================
        
        psm = self.psm1
        orn = (0.5, 0.5, -0.5, -0.5)
        #grasp gauze at start
        self.grasp_curriculum_hyperparam = 0.5 # how long to train with gauze already in the psm's jaw
        if training_progress < self.grasp_curriculum_hyperparam:
            # grasp_progress = training_progress / self.grasp_curriculum_hyperparam
            # robot_pos = np.array(gauze_pos) * grasp_progress + np.array(goal_pos) * (1 - grasp_progress)
            while True:
                # open the jaw
                psm.open_jaw()
                # TODO: strange thing that if we use --num_env=1 with openai baselines, the qs vary before and after step!
                step(0.5)

                # set the position until the psm can grasp it
                gauze_pos = np.random.uniform(low=sample_space[:, 0], high=sample_space[:, 1])
                pitch = np.random.uniform(low=-105., high=-75.)  # reduce difficulty
                orn_needle = p.getQuaternionFromEuler(np.deg2rad([-90, pitch, 90]))
                p.resetBasePositionAndOrientation(obj_id, gauze_pos, orn_needle)

                # record the needle pose and move the psm to grasp the needle
                pos_waypoint, orn_waypoint = get_link_pose(obj_id, self.obj_link1)  # the right side waypoint
                self._waypoint_z_init = pos_waypoint[2]
                orn_waypoint = np.rad2deg(p.getEulerFromQuaternion(orn_waypoint))
                p.resetBasePositionAndOrientation(obj_id, (0, 0, 0.01 * self.SCALING), (0, 0, 0, 1))

                # get the eef pose according to the needle pose
                orn_tip = p.getQuaternionFromEuler(np.deg2rad([90, -90 - orn_waypoint[1], 90]))
                pose_tip = [pos_waypoint + np.array([0.0015 * self.SCALING, 0, 0]), orn_tip]
                pose_eef = psm.pose_tip2eef(pose_tip)

                
                # # gauze_pos = (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,
                # #             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                # #             workspace_limits[2][0] + 0.01)
                # # self.gauze_pos =gauze_pos

                # gauze_pos = (robot_pos[0], robot_pos[1], robot_pos[2] - (-0.0007 + 0.0102) * self.SCALING)
                # self.gauze_pos =gauze_pos
                # # pos_waypoint, orn_waypoint = get_link_pose(obj_id, self.obj_link2)  # the right side waypoint
                # # p.resetBasePositionAndOrientation(obj_id, (0, 0, 0.01 * self.SCALING), (0, 0, 0, 1))
                # pose_tip = [gauze_pos + np.array([0.0015 * self.SCALING, 0, 0]), orn]
                # pose_eef = psm.pose_tip2eef(pose_tip)

                # move the psm
                pose_world = get_matrix_from_pose_2d(pose_eef)
                action_rcm = psm.pose_world2rcm(pose_world)
                success = psm.move(action_rcm)
                if success is False:
                    continue
                step(1)
                # place the gauze in the psm's jaw
                # gauze_pos = (robot_pos[0], robot_pos[1], robot_pos[2] - (-0.0007 + 0.0102) * self.SCALING)
                #             p.resetBasePositionAndOrientation(obj_id, pos_needle, orn_needle)
                cid = p.createConstraint(obj_id, -1, -1, -1,
                                        p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], gauze_pos,
                                        childFrameOrientation=orn)
                psm.close_jaw()
                step(0.5)
                p.removeConstraint(cid)
                self._activate(0)
                self._step_callback()
                step(1)
                self._step_callback()
                self.robot_pos = pose_tip
                if self._activated >= 0:
                    break


        else:
            final_initial_robot_pos = (workspace_limits[0][0],
                                        workspace_limits[1][1],
                                        (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
            non_grasp_progress = (training_progress - self.grasp_curriculum_hyperparam) / (1 - self.grasp_curriculum_hyperparam)
            robot_pos = np.array(final_initial_robot_pos) * non_grasp_progress + np.array(gauze_pos) * (1 - non_grasp_progress)
            self.robot_pos = robot_pos
            # ================================================

            orn = (0.5, 0.5, -0.5, -0.5)
            joint_positions = self.psm1.inverse_kinematics((robot_pos, orn), self.psm1.EEF_LINK_INDEX)
            self.psm1.reset_joint(joint_positions)
            self.block_gripper = False

        
    


    def _set_action(self, action: np.ndarray):
        action[3] = 0  # no yaw change
        super(GauzeRetrieveCurriculumLearningGraspGoalMove, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.02 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.02 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.03 * self.SCALING])
        
        # gauze_pos = self.obj_ids['rigid'][0]
        final_goal_pos = np.array(goal) * self.training_progress + np.array(self.robot_pos) * (1 - self.training_progress)
        # final_goal_pos = np.array(goal) * self.training_progress + np.array(self.gauze_pos) * (1 - self.training_progress)
        # print ("gauze_pos is,", self.gauze_pos)
        print ("final_goal_pos is,", final_goal_pos)
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
    env = GauzeRetrieveCurriculumLearningGraspGoalMove(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
