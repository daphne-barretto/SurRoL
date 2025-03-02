python -m baselines.run \
--num_env=2 --alg=ddpg --env=NeedlePickCurriculumLearning-v0 \
--eval_env=NeedlePick-v0 \
--num_timesteps=1e5 \
--log_path=./logs/ddpg/NeedlePickCurriculumLearning-1e5_0