# Goal-Conditioned Reinforcement Learning for Surgical Robotic Manipulation

Surgical robotic systems must be capable of handling ambiguous manipulation tasks where multiple similar objects require explicit goal specification. We investigate how different goal-conditioning methods affect learning in ambiguous robotic manipulation using a goal-conditioned peg transfer task where a robot must move a specified block among multiple colored blocks on a board. We compare conditioning on spatial goal representations, i.e. 3D coordinates, with semantic representations, such as one-hot encodings and color specifications, across two peg transfer tasks of differing complexity using DDPG with HER. Behavior cloning experiments establish baseline performance limitations, with pure imitation learning achieving only 10\% success despite extensive demonstration data, motivating our focus on reinforcement learning approaches. Spatial conditioning consistently outperforms semantic approaches, with performance gaps increasing as task complexity grows. In a two-block peg transfer environment, spatial methods achieve ~60\% success while semantic methods reach 20-40\%. In a four-block environment, spatial methods maintain 40-50\% success while semantic methods fail completely. Qualitative video analysis reveals semantic methods fail primarily due to manipulation control problems, and in the more complex four-block environment, may also fail due to goal identification issues. These results demonstrate that spatial coordinates offer direct, actionable guidance for robot control and task execution, whereas semantic representations may impose abstraction challenges that worsen with greater task complexity. Our findings offer important insights for the design of goal-conditioned surgical robotic systems.

## Installation

The project is built on Python 3.7,
[PyBullet 3.2.7](https://github.com/bulletphysics/bullet3),
[Gym 0.15.7](https://github.com/openai/gym/releases/tag/0.15.7) with custom environments and slight modifications,
[OpenAI Baselines](https://github.com/openai/baselines) with added evaluation environment support, and
[TensorFlow 1.14](https://www.tensorflow.org/install/pip). See [requirements.txt](./requirements.txt) for package details. This project has been tested on Ubuntu and Windows.

> Depending on your operating system and setup, you might require additional installations as alerted to you by your system.

### Create and activate a conda virtual environment

```shell
conda create -n surrol python=3.7 -y
conda activate surrol
```

### Install our version of SurRoL with optional dependencies for RL training and evaluation

```shell
git clone https://github.com/daphne-barretto/SurRoL.git 
cd SurRoL
pip install -e .[all]
```

### Install our version of Baselines

```shell
cd baselines
pip install -e .[all]
```

## Get started

The [test_psm.ipynb notebook](./tests/test_psm.ipynb) allows you to start the environment, load the robot, and test the kinematics.

The [./run directory](./run) contains sample bash files used to train policies with a specified set of parameters.

The [./surrol/data directory](./surrol/data) contains data_generation.py to generate the heuristic-based demonstration data and data_postprocessing.py to post-process the goal-conditioning.

The [./surrol/tasks directory](./surrol/tasks) contains peg_transfer*.py files that implement the peg transfer tasks with two blocks and four blocks.

The [./analysis/plot_results.ipynb notebook](./analysis/plot_results.ipynb) is used for quantitative analysis and generating plots.

## How to run experiments
First `cd run`, then run the following bash scripts for each experiment:

### Baseline
```
./herdemo_pegtransfer_two_blocks_onlywithtargetblock.sh  # two blocks
./herdemo_pegtransfer_tbo.sh  # four blocks
```
where “tbo” stands for “target block only.”

### DDPG with HER Goal Conditioning Experiments: Two Blocks
```
# Observations are robot state and positions of all blocks
./herdemo_pegtransfer_two_blocks_nocolor_fourtuple.sh   # conditioned on RGBA color encoding
./herdemo_pegtransfer_two_blocks_nocolor_onehot.sh       # conditioned on one hot block encoding
./herdemo_pegtransfer_two_blocks_nocolor_onehottargetpeg.sh   # conditioned on one hot block encoding and target peg position
./herdemo_pegtransfer_two_blocks_nocolor_targetblock.sh  # conditioned on goal block position
./herdemo_pegtransfer_two_blocks_nocolor_targetblocktargetpeg.sh   # conditioned on goal block and target peg positions

# Observations are robot state, positions of all blocks, and color of all blocks
./herdemo_pegtransfer_two_blocks_fourtuple.sh   # conditioned on RGBA color encoding
./herdemo_pegtransfer_two_blocks_onehot.sh       # conditioned on one hot block encoding
./herdemo_pegtransfer_two_blocks_onehottargetpeg.sh   # conditioned on one hot block encoding and target peg position
./herdemo_pegtransfer_two_blocks_targetblock.sh  # conditioned on goal block position
./herdemo_pegtransfer_two_blocks_targetblocktargetpeg.sh   # conditioned on goal block and target peg positions
```

### DDPG with HER Goal Conditioning Experiments: Four Blocks
```
# Observations are robot state, positions of all blocks
./herdemo_pegtransfer_fourtuple.sh       # conditioned on RGBA color encoding
./herdemo_pegtransfer_onehot.sh       # conditioned on one hot block encoding
./herdemo_pegtransfer_onehottargetpeg.sh   # conditioned on one hot block encoding and target peg position
./herdemo_pegtransfer_targetblock.sh    # conditioned on goal block position
./herdemo_pegtransfer_targetblocktargetpeg.sh    # conditioned on goal block and target peg positions

# Observations are robot state, positions of all blocks, and color of all blocks
./herdemo_pegtransfer_color_fourtuple.sh       # conditioned on RGBA color encoding
./herdemo_pegtransfer_color_onehot.sh       # conditioned on one hot block encoding
./herdemo_pegtransfer_color_onehottargetpeg.sh   # conditioned on one hot block encoding and target peg position
./herdemo_pegtransfer_color_targetblock.sh    # conditioned on goal block position
./herdemo_pegtransfer_color_targetblocktargetpeg.sh    # conditioned on goal block and target peg positions
```

## License

This project is released under the [MIT license](LICENSE).

## Acknowledgement

The code is built with custom versions of the [**Sur**gical **Ro**bot **L**earning platform (**SurRoL**)](https://med-air.github.io/SurRoL/), [OpenAI Baselines](https://github.com/openai/baselines), and [Gym 0.15.7](https://github.com/openai/gym/releases/tag/0.15.7).

## Contact
For any questions, please feel free to email <a href="mailto:daphne.barretto@stanford.edu">daphne.barretto@stanford.edu</a>, <a href="mailto:alylee15@stanford.edu">alylee15@stanford.edu</a>, and <a href="mailto:elsabis@stanford.edu">elsabis@stanford.edu</a>.

