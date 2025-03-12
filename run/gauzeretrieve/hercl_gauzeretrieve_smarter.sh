python -m baselines.run \
--num_env=2 --alg=her --env=GauzeRetrieveCurriculumLearningSmarter-v0 \
--eval_env=GauzeRetrieve-v0 \
--num_timesteps=1e5 \
--save_path=./policies/hercl/GauzeRetrieveCurriculumLearningSmarter-1e5_0 \
--log_path=./logs/hercl/GauzeRetrieveCurriculumLearningSmarter-1e5_0 \
--n_cycles=20 