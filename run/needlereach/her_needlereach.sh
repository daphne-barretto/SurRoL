python -m baselines.run \
--num_env=2 --alg=her --env=NeedleReach-v0 \
--num_timesteps=1e5 \
--save_path=../policies/her/NeedleReach-1e5_0 \
--log_path=../logs/her/NeedleReach-1e5_0 \
--n_cycles=20