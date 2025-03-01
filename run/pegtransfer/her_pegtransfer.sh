python -m baselines.run \
--num_env=2 --alg=her --env=PegTransfer-v0 \
--num_timesteps=1e5 \
--save_path=./policies/her/PegTransfer-1e5_0 \
--log_path=./logs/her/PegTransfer-1e5_0 \
--n_cycles=20