python -m baselines.run \
--num_env=2 --alg=ddpg --env=NeedleReach-v0 \
--num_timesteps=1e5 \
--save_video_interval=30 \
--save_path=./policies/ddpg/NeedleReach-1e5_0 \
--log_path=./logs/ddpg/NeedleReach-1e5_0