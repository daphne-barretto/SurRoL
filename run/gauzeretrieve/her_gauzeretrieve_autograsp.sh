python -m baselines.run \
--num_env=2 --alg=her --env=GauzeRetrieveAutoGrasp-v0 \
--eval_env=GauzeRetrieveAutoGrasp-v0 \
--num_timesteps=1e5 \
--save_path=./policies/her/GauzeRetrieveAutoGrasp-1e5_1 \
--log_path=./logs/her/GauzeRetrieveAutoGrasp-1e5_1 \
--n_cycles=20