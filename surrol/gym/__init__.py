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
    entry_point='surrol.tasks.gauze_retrieve_curriculum_learning_smarter:GauzeRetrieveCurriculumLearningSmarter',
    max_episode_steps=50,
)

register(
    id='GauzeRetrieveCurriculumLearningGraspGoalMove-v0',
    entry_point='surrol.tasks.gauze_retrieve_curriculum_learning_grasp_goalmove:GauzeRetrieveCurriculumLearningGraspGoalMove',
    max_episode_steps=50,
)

# Bimanual PSM Env
register(
    id='NeedleRegrasp-v0',
    entry_point='surrol.tasks.needle_regrasp_bimanual:NeedleRegrasp',
    max_episode_steps=50,
)

register(
    id='BiPegTransfer-v0',
    entry_point='surrol.tasks.peg_transfer_bimanual:BiPegTransfer',
    max_episode_steps=50,
)

# ECM Env
register(
    id='ECMReach-v0',
    entry_point='surrol.tasks.ecm_reach:ECMReach',
    max_episode_steps=50,
)

register(
    id='MisOrient-v0',
    entry_point='surrol.tasks.ecm_misorient:MisOrient',
    max_episode_steps=50,
)

register(
    id='StaticTrack-v0',
    entry_point='surrol.tasks.ecm_static_track:StaticTrack',
    max_episode_steps=50,
)

register(
    id='ActiveTrack-v0',
    entry_point='surrol.tasks.ecm_active_track:ActiveTrack',
    max_episode_steps=500,
)
