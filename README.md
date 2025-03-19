# Curriculum Learning for Long-Horizon, Goal-Oriented Surgical Robotics Tasks

Surgical robotics often involves long-horizon, goal-oriented tasks; sparse rewards in these scenarios create challenges in exploration that can result in low data and compute efficiency. Prior work has shown that additional data, such as demonstrations and sequences of learned skills for a task, can improve training efficiency and policy effectiveness at the cost of requiring potentially hard-to-obtain data. We apply curriculum learning, a data ordering strategy that gradually introduces more complex training concepts, to long-horizon, goal-oriented surgical robotics tasks to address these challenges without additional data. We find that in some tasks, curriculum learning can perform comparable to having demonstration data, supporting its potential in the surgical robotics domain. In more complex tasks, many curriculum learning strategies still struggle to learn precise manipulation, such as grasping objects. However, by exploring a variety of curricula, we make progress towards using curriculum learning to enable initial grasping behaviors to be learned in training for surgical robotics tasks.

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

[test_psm.ipynym](./tests/test_psm.ipynb) allows you to start the environment, load the robot, and test the kinematics.

The [./run directory](./run) contains sample bash files used to train policies with a specified set of parameters.

## Real-World Robots

The robot control API follows [the da Vinci Research Kit (dVRK)](https://github.com/jhu-dvrk/dvrk-ros/tree/master/dvrk_python/src/dvrk), which is compatible with the real-world dVRK robots.

## License

This project is released under the [MIT license](LICENSE).

## Acknowledgement

The code is built with custom versions of the [**Sur**gical **Ro**bot **L**earning platform (**SurRoL**)](https://med-air.github.io/SurRoL/), [OpenAI Baselines](https://github.com/openai/baselines), and [Gym 0.15.7](https://github.com/openai/gym/releases/tag/0.15.7).

## Contact
For any questions, please feel free to email <a href="mailto:daphne.barretto@stanford.edu">daphne.barretto@stanford.edu</a> and <a href="mailto:meganliu@stanford.edu">meganliu@stanford.edu</a>.
