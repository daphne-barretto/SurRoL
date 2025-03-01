python -m baselines.run \
--num_env=2 --alg=ddpg --env=PegTransfer-v0 \
--num_timesteps=1e5 \
--log_path=./logs/ddpg/PegTransfer-1e5_0