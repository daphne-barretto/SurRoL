python -m baselines.run \
--num_env=2 --alg=her --env=GauzeRetrieveCurriculumLearning-v0 \
--eval_env=GauzeRetrieve-v0 \
--num_timesteps=1e5 \
--save_path=./policies/hercl/GauzeRetrieveCurriculumLearning-1e5_0 \
--log_path=./logs/hercl/GauzeRetrieveCurriculumLearning-1e5_0 \
--n_cycles=20 \
--save_video_interval=1