# python surrol/data/data_generation_visual_obs.py --env NeedleReach-v0 --num_itr 10000 &
# python surrol/data/data_generation_visual_obs.py --env GauzeRetrieve-v0 --num_itr 10000 &
# python surrol/data/data_generation_visual_obs.py --env NeedlePick-v0 --num_itr 10000 &
# python surrol/data/data_generation_visual_obs.py --env PegTransfer-v0 --num_itr 10000 &
python surrol/data/data_generation.py --env NeedlePick-v0 --num_itr 1000 &
python surrol/data/data_generation.py --env NeedlePick-v0 --num_itr 100 &
python surrol/data/data_generation.py --env GauzeRetrieve-v0 --num_itr 1000 &
python surrol/data/data_generation.py --env GauzeRetrieve-v0 --num_itr 100 &
python surrol/data/data_generation.py --env PegTransfer-v0 --num_itr 1000 &
python surrol/data/data_generation.py --env PegTransfer-v0 --num_itr 100 &
python surrol/data/data_generation_visual_obs.py --env NeedlePick-v0 --num_itr 1000 &
python surrol/data/data_generation_visual_obs.py --env NeedlePick-v0 --num_itr 100 &
python surrol/data/data_generation_visual_obs.py --env GauzeRetrieve-v0 --num_itr 1000 &
python surrol/data/data_generation_visual_obs.py --env GauzeRetrieve-v0 --num_itr 100 &
python surrol/data/data_generation_visual_obs.py --env PegTransfer-v0 --num_itr 1000 &
python surrol/data/data_generation_visual_obs.py --env PegTransfer-v0 --num_itr 100