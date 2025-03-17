python -m baselines.run \
--num_env=2 --alg=her --env=GauzeRetrieveContactApproxCurriculumLearning-v0 \
--eval_env=GauzeRetrieveContactApprox-v0 \
--num_timesteps=1e5 \
--save_path=./policies/hercl/GauzeRetrieveContactApproxCurriculumLearning-1e5_0 \
--log_path=./logs/hercl/GauzeRetrieveContactApproxCurriculumLearning-1e5_0 \
--n_cycles=20