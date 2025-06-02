from gym.envs.registration import register


# PSM Env
register(
    id='NeedleReach-v0',
    entry_point='surrol.tasks.needle_reach:NeedleReach',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieve-v0',
    entry_point='surrol.tasks.gauze_retrieve:GauzeRetrieve',
    max_episode_steps=50,
)

register(
    id='NeedlePick-v0',
    entry_point='surrol.tasks.needle_pick:NeedlePick',
    max_episode_steps=50,
)

register(
    id='PegTransfer-v0',
    entry_point='surrol.tasks.peg_transfer:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTargetBlock-v0',
    entry_point='surrol.tasks.peg_transfer_target_block:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTargetBlockTargetPeg-v0',
    entry_point='surrol.tasks.peg_transfer_target_block_and_target_peg:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferOneHotTargetPeg-v0',
    entry_point='surrol.tasks.peg_transfer_onehot_and_target_peg:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferAllBlocksFourTuple-v0',
    entry_point='surrol.tasks.peg_transfer_four_tuple:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferColor-v0',
    entry_point='surrol.tasks.peg_transfer_with_all_blocks_colored:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferColorTargetBlock-v0',
    entry_point='surrol.tasks.peg_transfer_with_all_blocks_colored_target_block:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferColorTargetBlockTargetPeg-v0',
    entry_point='surrol.tasks.peg_transfer_with_all_blocks_colored_target_block_peg:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferColorOneHotTargetPeg-v0',
    entry_point='surrol.tasks.peg_transfer_with_all_blocks_colored_one_hot_target_peg:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTBO-v0',
    entry_point='surrol.tasks.peg_transfer_with_only_target_block:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTBOTwoBlock-v0',
    entry_point='surrol.tasks.peg_transfer-two_blocks-with_only_target_block:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferFourTuple-v0',
    entry_point='surrol.tasks.peg_transfer_with_all_blocks_colored_four_tuple:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferColorLanguage-v0',
    entry_point='surrol.tasks.peg_transfer_with_all_blocks_colored_color_language:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTwoBlocksOneHot-v0',
    entry_point='surrol.tasks.peg_transfer-two_blocks-with_all_blocks_colored-no_obs:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTwoBlocksFourTuple-v0',
    entry_point='surrol.tasks.peg_transfer-two_blocks-with_all_blocks_colored-no_obs_four_tuple:PegTransfer',
    max_episode_steps=50,
)

register(
    id='NeedleReachRandom-v0',
    entry_point='surrol.tasks.needle_reach_random:NeedleReachRandom',
    max_episode_steps=50,
)

# PSM Curriculum Learning Env
register(
    id='NeedleReachCurriculumLearning-v0',
    entry_point='surrol.tasks.needle_reach_curriculum_learning:NeedleReachCurriculumLearning',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieveCurriculumLearning-v0',
    entry_point='surrol.tasks.gauze_retrieve_curriculum_learning:GauzeRetrieveCurriculumLearning',
    max_episode_steps=50,
)

register(
    id='NeedlePickCurriculumLearning-v0',
    entry_point='surrol.tasks.needle_pick_curriculum_learning:NeedlePickCurriculumLearning',
    max_episode_steps=50,
)

register(
    id='PegTransferCurriculumLearning-v0',
    entry_point='surrol.tasks.peg_transfer_curriculum_learning:PegTransferCurriculumLearning',
    max_episode_steps=50,
)

#PSM CL Smarters
register(
    id='GauzeRetrieveCurriculumLearningSmarter-v0',
    entry_point='surrol.tasks.gauze_retrieve_curriculum_learning_goalmove:GauzeRetrieveCurriculumLearningSmarter',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieveCurriculumLearningGraspGoalMove-v0',
    entry_point='surrol.tasks.gauze_retrieve_curriculum_learning_grasp_goalmove:GauzeRetrieveCurriculumLearningGraspGoalMove',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieveCurriculumLearningGraspGoalMoveStartMove-v0',
    entry_point='surrol.tasks.gauze_retrieve_curriculum_learning_grasp_goalmove_startmove:GauzeRetrieveCurriculumLearningGraspGoalMoveStartMove',
    max_episode_steps=50,
)

# PSM Contact Approx Tasks
register(
    id='GauzeRetrieveContactApprox-v0',
    entry_point='surrol.tasks.gauze_retrieve_contact_approx:GauzeRetrieveContactApprox',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieveContactApproxCurriculumLearning-v0',
    entry_point='surrol.tasks.gauze_retrieve_contact_approx_curriculum_learning:GauzeRetrieveContactApproxCurriculumLearning',
    max_episode_steps=50,
)

# PSM Auto Grasp Tasks
register(
    id='GauzeRetrieveAutoGrasp-v0',
    entry_point='surrol.tasks.gauze_retrieve_auto_grasp:GauzeRetrieveAutoGrasp',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieveAutoGraspCurriculumLearning-v0',
    entry_point='surrol.tasks.gauze_retrieve_auto_grasp_curriculum_learning:GauzeRetrieveAutoGraspCurriculumLearning',
    max_episode_steps=50,
)