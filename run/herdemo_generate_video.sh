#/home/alycialee/SurRoL/policies/her/PegTransferTwoBlocks-fourtuple-1000demo-2e5_0_fixed_r2

# 


python -m baselines.generate_videos \
--model-path ./../policies/her/PegTransferTwoBlocksNoColor-fourtuple-1000demo-2e5_0_fixed_r0 \
--env PegTransferTwoBlocksFourTuple-v0 \
--video-dir ./evaluation_videos/PegTransferTwoBlocksNoColor-fourtuple-1000demo-2e5_0_fixed_r0 \
--num-episodes 10 \
--video-length 200

python -m baselines.generate_videos \
--model-path ./../policies/her/PegTransferTwoBlocks-onehot-1000demo-2e5_0_fixed_r0 \
--env PegTransferTwoBlocksOneHot-v0 \
--video-dir ./evaluation_videos/PegTransferTwoBlocks-onehot-1000demo-2e5_0_fixed_r0 \
--num-episodes 10 \
--video-length 200

python -m baselines.generate_videos \
--model-path ./../policies/her/PegTransferTwoBlocks-fourtuple-1000demo-2e5_0_fixed_r2 \
--env PegTransferTwoBlocksFourTuple-v0 \
--video-dir ./evaluation_videos/PegTransferTwoBlocksColor-fourtuple-1000demo-2e5_0_fixed_r2 \
--num-episodes 10 \
--video-length 200

python -m baselines.generate_videos \
--model-path ./../policies/her/PegTransferColor-fourtuple-1000demo-2e5_0_r1 \
--env PegTransferFourTuple-v0 \
--video-dir ./evaluation_videos/PegTransferColor-fourtuple-1000demo-2e5_0_r1 \
--num-episodes 10 \
--video-length 200


python -m baselines.generate_videos \
--model-path ./../policies/her/PegTransferColor-onehot-1000demo-2e5_0 \
--env PegTransferColor-v0 \
--video-dir ./evaluation_videos/PegTransferColor-onehot-1000demo-2e5_0 \
--num-episodes 10 \
--video-length 200


# python -m baselines.generate_videos \
# --model-path ./../policies/her/PegTransferColor-14_onehottargetpeg-1000demo-2e5_0_r2 \
# --env PegTransferColorOneHotTargetPeg-v0 \
# --video-dir ./evaluation_videos/PegTransferColor-14_onehottargetpeg-1000demo-2e5_0_r2 \
# --num-episodes 10 \
# --video-length 200

# python -m baselines.generate_videos \
# --model-path ./../policies/her/PegTransfer-targetblock-1000demo-2e5_0 \
# --env PegTransferTargetBlock-v0 \
# --video-dir ./evaluation_videos/PegTransfer-targetblock-1000demo-2e5_0 \
# --num-episodes 10 \
# --video-length 200