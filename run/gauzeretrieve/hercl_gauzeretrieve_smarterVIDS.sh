python -m baselines.run \
--num_env=2 --alg=her --env=GauzeRetrieveCurriculumLearningSmarter-v0 \
--eval_env=GauzeRetrieve-v0 \
--num_timesteps=1e5 \
--log_path=./logs/ddpgcl/GauzeRetrieveCurriculumLearningSmarter-1e5_0 \
--save_video_interval=1