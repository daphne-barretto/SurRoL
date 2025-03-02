python -m baselines.run \
--num_env=2 --alg=ddpg --env=NeedleReachCurriculumLearning-v0 \
--eval_env=NeedleReach-v0 \
--num_timesteps=1e5 \
--log_path=./logs/ddpg/NeedleReachCurriculumLearning-1e5_0