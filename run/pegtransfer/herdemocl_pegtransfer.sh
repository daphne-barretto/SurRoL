python -m baselines.run \
--num_env=2 --alg=her --env=PegTransferCurriculumLearning-v0 \
--eval_env=PegTransfer-v0 \
--num_timesteps=1e5 \
--save_path=./policies/herdemocl/PegTransferCurriculumLearning-1e5_0 \
--log_path=./logs/herdemocl/PegTransferCurriculumLearning-1e5_0 \
--n_cycles=20 \
--demo_file=./surrol/data/demo/data_PegTransfer-v0_random_100.npz \
--bc_loss=1 --q_filter=1 --num_demo=100 --demo_batch_size=128 --prm_loss_weight=0.001 --aux_loss_weight=0.0078 --n_cycles=20 --batch_size=1024 --random_eps=0.1 --noise_eps=0.1