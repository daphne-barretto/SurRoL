python -m baselines.run \
--num_env=2 --alg=her --env=GauzeRetrieveContactApprox-v0 \
--eval_env=GauzeRetrieveContactApprox-v0 \
--num_timesteps=1e5 \
--save_path=./policies/her/GauzeRetrieveContactApprox-1e5_0 \
--log_path=./logs/her/GauzeRetrieveContactApprox-1e5_0 \
--n_cycles=20