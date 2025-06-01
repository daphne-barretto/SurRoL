## Using with_target_block demo data
## herdemo_pegtransfer_onlywithtargetblock.sh
python -m baselines.run \
--num_env=2 --alg=her --env=PegTransferTBO-v0 --num_timesteps=2e5 --policy_save_interval=5 \
--demo_file=../surrol/data/with_target_block/data_PegTransfer-v0_random_1000_2025-05-31_16-49-17.npz \
--save_path=../policies/her/PegTransferTBO-1000demo-2e5_0 \
--bc_loss=1 --q_filter=1 --num_demo=1000 --demo_batch_size=128 --prm_loss_weight=0.001 --aux_loss_weight=0.0078 --n_cycles=20 --batch_size=1024 --random_eps=0.1 --noise_eps=0.1 \
--log_path=../logs/her/PegTransferTBO-1000demo-2e5_0 \
--save_video_interval=1