python -m baselines.run \
--num_env=2 --alg=ddpg --env=GauzeRetrieveCurriculumLearning-v0 \
--eval_env=GauzeRetrieve-v0 \
--num_timesteps=1e5 \
--log_path=./logs/ddpgcl/GauzeRetrieveCurriculumLearning-1e5_0