python -m baselines.run \
--num_env=2 --alg=ddpg --env=GauzeRetrieve-v0 \
--num_timesteps=1e5 \
--log_path=./logs/ddpg/GauzeRetrieve-1e5_0