# Curriculum Learning for Long-Horizon, Goal-Oriented Surgical Robotics Tasks

<Abstract and Contributions>

## Installation

The project is built on Ubuntu with Python 3.7,
[PyBullet](https://github.com/bulletphysics/bullet3),
[Gym 0.15.6](https://github.com/openai/gym/releases/tag/0.15.6),
and evaluated with [Baselines](https://github.com/openai/baselines),
[TensorFlow 1.14](https://www.tensorflow.org/install/pip).

### Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n surrol python=3.7 -y
    conda activate surrol
    ```

2. Install gym (slightly modified), tensorflow-gpu==1.14, baselines (modified).

### Install SurRoL

```shell
git clone https://github.com/med-air/SurRoL.git
cd SurRoL
pip install -e .
```

## Get started

The robot control API follows [dVRK](https://github.com/jhu-dvrk/dvrk-ros/tree/master/dvrk_python/src/dvrk)
(before "crtk"), which is compatible with the real-world dVRK robots.

You may have a look at the jupyter notebooks in [tests](./tests).
There are some test files for [PSM](./tests/test_psm.ipynb) and [ECM](./tests/test_ecm.ipynb),
that contains the basic procedures to start the environment, load the robot, and test the kinematics.

We also provide some [run files](./run) to evaluate the environments using baselines.

## License

This project is released under the [MIT license](LICENSE).

## Acknowledgement

The code is built with custom versions of the [**Sur**gical **Ro**bot **L**earning platform (**SurRoL**)](https://med-air.github.io/SurRoL/) and [OpenAI Baselines](https://github.com/openai/baselines).

## Contact
For any questions, please feel free to email <a href="mailto:daphne.barretto@stanford.edu">daphne.barretto@stanford.edu</a> and <a href="mailto:meganliu@stanford.edu">meganliu@stanford.edu</a>.
