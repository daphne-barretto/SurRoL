from gym.envs.registration import register


##### SurRoL Patient-Side Manipulator Environments #####

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

##### Goal-Conditioned Peg Transfer Environments #####

# Baselines: Target Block Position

register(
    id='PegTransferTBOTwoBlock-v0',
    entry_point='surrol.tasks.peg_transfer-two_blocks-with_only_target_block:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTBO-v0',
    entry_point='surrol.tasks.peg_transfer_with_only_target_block:PegTransfer',
    max_episode_steps=50,
)

# 2 Blocks with All block positions

register(
    id='PegTransferTwoBlocksAllBlocks-v0',
    entry_point='surrol.tasks.peg_transfer-two_blocks-with_all_blocks-no_obs:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTwoBlocksTargetBlock-v0',
    entry_point='surrol.tasks.peg_transfer-two_blocks-with_all_blocks-no_obs_target_block:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTwoBlocksOneHotTargetPeg-v0',
    entry_point='surrol.tasks.peg_transfer-two_blocks-with_all_blocks-no_obs_one_hot_target_peg:PegTransfer',
    max_episode_steps=50,
)

# 2 Blocks with All block positions and colors

register(
    id='PegTransferTwoBlocksOneHot-v0',
    entry_point='surrol.tasks.peg_transfer-two_blocks-with_all_blocks_colored-no_obs:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTwoBlocksColored-v0',
    entry_point='surrol.tasks.peg_transfer-two_blocks-with_all_blocks_colored-no_obs:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTwoBlocksFourTuple-v0',
    entry_point='surrol.tasks.peg_transfer-two_blocks-with_all_blocks_colored-no_obs_four_tuple:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTwoBlocksColorTargetBlock-v0',
    entry_point='surrol.tasks.peg_transfer-two_blocks-with_all_blocks_colored-targetblock:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferTwoBlocksColorOneHotTargetPeg-v0',
    entry_point='surrol.tasks.peg_transfer-two_blocks-with_all_blocks_colored-onehottargetpeg:PegTransfer',
    max_episode_steps=50,
)

# 4 Blocks with All block positions

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
    id='PegTransferAllBlocksFourTuple-v0',
    entry_point='surrol.tasks.peg_transfer_four_tuple:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferOneHotTargetPeg-v0',
    entry_point='surrol.tasks.peg_transfer_onehot_and_target_peg:PegTransfer',
    max_episode_steps=50,
)

# 4 Blocks with All block positions and colors

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
    id='PegTransferFourTuple-v0',
    entry_point='surrol.tasks.peg_transfer_with_all_blocks_colored_four_tuple:PegTransfer',
    max_episode_steps=50,
)

register(
    id='PegTransferColorLanguage-v0',
    entry_point='surrol.tasks.peg_transfer_with_all_blocks_colored_color_language:PegTransfer',
    max_episode_steps=50,
)