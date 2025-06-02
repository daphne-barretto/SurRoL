python -m baselines.run \
--num_env=2 --alg=her --env=PegTransferTwoBlocksOneHotTargetPeg-v0 --num_timesteps=2e5 --policy_save_interval=5 \
--demo_file=../surrol/data/two_blocks_color/data_PegTransfer-v0_random_1000_2025-06-01_10-06-51_onehottargetpeg.npz \
--save_path=../policies/her/PegTransferTwoBlocks-onehottargetpeg-1000demo-2e5_0_d3 \
--bc_loss=1 --q_filter=1 --num_demo=1000 --demo_batch_size=128 --prm_loss_weight=0.001 --aux_loss_weight=0.0078 --n_cycles=20 --batch_size=1024 --random_eps=0.1 --noise_eps=0.1 \
--log_path=../logs/her/PegTransferTwoBlocks-onehottargetpeg-1000demo-2e5_0_d3

## save and log path suffix r0 = run #0, r1 = run #1, r2 = run #2