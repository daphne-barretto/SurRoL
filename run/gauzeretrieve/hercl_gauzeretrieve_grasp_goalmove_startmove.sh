python -m baselines.run \
--num_env=2 --alg=her --env=GauzeRetrieveCurriculumLearningGraspGoalMoveStartMove-v0 \
--eval_env=GauzeRetrieve-v0 \
--num_timesteps=1e5 \
--log_path=./logs/hercl/GauzeRetrieveCurriculumLearningGraspGoalMoveStartMove-1e5_0 \
--n_cycles=20 