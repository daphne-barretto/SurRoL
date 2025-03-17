python -m baselines.run \
--num_env=2 --alg=her --env=GauzeRetrieveAutoGraspCurriculumLearning-v0 \
--eval_env=GauzeRetrieveAutoGrasp-v0 \
--num_timesteps=1e5 \
--save_path=./policies/hercl/GauzeRetrieveAutoGraspCurriculumLearning-1e5_1 \
--log_path=./logs/hercl/GauzeRetrieveAutoGraspCurriculumLearning-1e5_1 \
--n_cycles=20