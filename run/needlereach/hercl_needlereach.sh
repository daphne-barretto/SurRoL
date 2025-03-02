python -m baselines.run \
--num_env=2 --alg=her --env=NeedleReachCurriculumLearning-v0 \
--eval_env=NeedleReach-v0 \
--num_timesteps=1e5 \
--save_path=./policies/cl/NeedleReachCurriculumLearning-1e5_0 \
--log_path=./logs/cl/NeedleReachCurriculumLearning-1e5_0 \
--n_cycles=20