# AQRL: Autonomous Quadruped Robot with Reinforcement Learning

**Senior Design Project**<br>
**Oğuz Kağan Çayır**<br>
**Yeditepe University, Electrical and Electronics Engineering**<br>

This repository contains the final source package of a senior design project titled:

**Development of an Autonomous Quadruped Robot Using Reinforcement Learning and Multi-Sensor Integration: From Simulation to Real-World Implementation**

AQRL is a 12-DoF servo-driven quadruped robot platform developed to study simulation-based locomotion, analytical gait control, reinforcement learning and physical robot deployment. The project uses an existing SpotMicro-style mechanical body as the hardware base. The electronic system, MuJoCo simulation model, locomotion controller, reinforcement learning environment, IMU feedback pipeline and Raspberry Pi deployment structure were developed and integrated as part of this project.

The locomotion system uses a hybrid control approach. A Bezier-based trot gait planner and analytical inverse kinematics generate the base walking motion. A PPO policy is then trained as a residual controller that modifies gait parameters instead of directly commanding all joint angles. This keeps the learned behavior more structured and makes physical deployment safer for a low-cost servo-based robot.

The final policy uses an IMU-compatible observation structure based on roll, pitch, yaw rate, integrated yaw error, gait phase, previous action and command values. After deterministic deployment-style evaluation, the v9 1.85M checkpoint was selected as the final policy. This policy was exported as a TorchScript actor and deployed on Raspberry Pi for physical walking tests. A LiDAR sensor was also evaluated as a separate perception module, but it was not included in the final onboard walking loop.

The project builds on ideas from a previous SpotMicro-style quadruped work and open-source SpotMicroAI gait examples, especially for the Bezier trot concept. These ideas were adapted to the AQRL robot geometry, MuJoCo model, reinforcement learning environment and real hardware deployment pipeline.

## Repository Contents

- `assets/mujoco/`: AQRL MuJoCo XML model and mesh files.
- `src/controllers/`: Bezier trot planner, transform utilities and AQRL analytical inverse kinematics.
- `src/envs/`: IMU-compatible Gymnasium/MuJoCo reinforcement learning environment.
- `src/rl/`: PPO training and evaluation scripts.
- `src/sensors/`: MPU6050 processing and policy observation formatting.
- `src/robot/`: default standing pose used by simulation and physical deployment.
- `scripts/`: simulation viewing, policy export, Raspberry Pi deployment and hardware validation scripts.
- `runs/`: selected v9 1.85M policy, comparison policies and exported TorchScript actor.

This repository is intended as the final project code package for documentation, review and reproducibility of the implemented senior design project.
