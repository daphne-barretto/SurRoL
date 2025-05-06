## Setup

Create conda environment
1. conda create -n surrol python=3.7 -y
2. conda activate surrol

Clone SurRoL repo and install SurRoL with all packages into conda environment
1. git clone https://github.com/daphne-barretto/SurRoL.git
2. cd SurRoL
3. pip install -e .[all]

Clone baselines repo and install baselines into conda environment
1. cd ..
2. git clone https://github.com/openai/baselines.git
3. cd baselines
4. pip install -e .

Downgrade protobuf
1. pip install protobuf==3.20

Change mpi4py installation (can be OS + hardware specific)
1. pip uninstall mpi4py -y
2. conda install mpi4py -y

Following the above instructures produces the environment list in requirements.txt

## Tests

Jupyter Notebook for PSM
1. jupyter notebook
2. go to local host site and navigate to .\tests\test_psm.ipynb
3. run blocks, and the last one should simulate the PSM moving

Run the oracle for a task
1. python .\surrol\tasks\\\<task name>.py
2. this should simulate the oracle

Collect demonstration data
1. python surrol\data\data_generation.py --env \<env\> --steps \<steps\> [--video]

Run training on baselines methods
1. bash run\\\<experiment setup bash file>.sh