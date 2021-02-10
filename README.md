# Learning from Sparse Demonstrations

This project is the implementation of the paper _**Learning from Sparse Demonstrations**_, co-authored by
Wanxin Jin, Todd D. Murphey, Dana Kulić, Neta Ezer, Shaoshuai Mou. Please find more details in

* Paper: https://arxiv.org/abs/2008.02159 for technical details.
* Demos: https://wanxinjin.github.io/posts/lfsd for video demos.

This repo has been tested with:
* Python 3.8.5, Ubuntu 20.04.2 LTS

## Project Structure
The current version of the project consists of three folders:


* **_CPDP_** : a package including an optimal control solver, functionalities for differentiating maximum principle, and functionalities to solve the differential maximum principle.  
* **_JinEnv_** : an independent package providing various robot environments to simulate on.
* **_Examples_** : various examples to reproduce the experiments in the paper.


## Dependency Packages
Please make sure that the following packages have already been installed before 
use of the PDP package or JinEnv Package.

   * [CasADi](https://web.casadi.org/): version > 3.5.1.
   * [IPOPT](https://coin-or.github.io/Ipopt/): need this for solving NLP in CasADi, the binary installation is fine.
   * [Numpy](https://numpy.org/): version > 1.18.1.
   * [Transforms3d](https://pypi.org/project/transforms3d/)
   * [Scipy](https://www.scipy.org/)
   * [matplotlib](https://matplotlib.org/)
   * [FFmpeg](https://ffmpeg.org/): need this for saving the data visualization.
   * [PyQt5](https://pypi.org/project/PyQt5/)

```
sudo apt update
sudo apt install coinor-libipopt-dev ffmpeg
pip3 install casadi numpy transfroms3d scipy matplotlib pyqt5
```


## How to Train Your Robots.
Below is the procedure of how to apply the codes to train your robot to learn from sparse demonstrations.

* **Step 1.** Load a robot environment from JinEnv library (specify parameters of the robot dynamics).
* **Step 2.** Specify a parametric time-warping function and a parametric  cost function (loaded from JinEnv).
* **Step 3.** Provide some sparse demonstrations and define the trajectory loss function.
* **Step 4.** Set the learning rate and start training your robot (apply CPDP) given initial guesses.
* **Step 5.** Done, check and simulate your robot visually (use animation utilities from JinEnv).

The quickest way to hand on the codes is to check and run the examples under the folder `./Examples/` .


Run the algorithm with pre-defined waypoints:
```
$ cd <ROOT_DIRECTORY>
$ python3 Examples/quad_example.py
```

Run the algorithm with human inputs:
```
$ cd <ROOT_DIRECTORY>
$ python3 Examples/quad_example_human_input.py
```

To obtain human input via matplotlib ginput():
```
$ cd <ROOT_DIRECTORY>
$ python3 test/test_input.py
```

To obtain human input via a GUI with PyQt5:
```
$ cd <ROOT_DIRECTORY>
$ python3 test/test_gui.py
```


## Contact Information and Citation
If you have encountered a bug in your implementation of the code, please feel free to let me known via email:

   * name: wanxin jin (he/his)
   * email: wanxinjin@gmail.com

The codes are under regularly update.

If you find this project helpful in your publications, please consider citing our paper.
    
    @article{jin2020learning,
      title={Learning from Sparse Demonstrations},
      author={Jin, Wanxin and Murphey, Todd D and Kuli{\'c}, Dana and Ezer, Neta and Mou, Shaoshuai},
      journal={arXiv preprint arXiv:2008.02159},
      year={2020}
    }