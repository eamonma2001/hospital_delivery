# Autonomous Hospital Navigation Using Imitation Learning

![PR2_In_Hospital](videos/demonstration.mp4)

## Overview 
This project explores the application of imitation learning for autonomous robot navigation in a simulated hospital environment. By training a deep learning model on data collected from human-controlled demonstrations, the project aims to enable a robot to navigate complex and dynamic hospital settings autonomously.

## Project Structure
- robot_ws & simulation_ws: Modified versions of [the AWS RoboMaker PR2 Hospital Sample Application](https://github.com/aws-samples/aws-robomaker-sample-application-pr2-hospital), adapted for compatibility with ROS Neotic and Ubuntu 20.04.
- [data.zip](https://drive.google.com/file/d/1mQodM_7ZDt8_f2K3R26TLmGL7UIpg0rZ/view?usp=sharing): Contains pre-processed sensor data from demonstrations. Original .bag files are excluded due to their large size.
- process_odom.py & process_scan.py: Scripts for extracting sensor data from .bag files and converting them into formatted .csv files.
- model_nurse_to_exam.py, model_nurse_to_patient.py, model_nurse_to_surgery.py: Scripts for training models based on collected data.
- nurse_to_exam_model.tf, nurse_to_patient_model.tf, model_nurse_to_model.tf: Trained models.
- command.py: Script for using the trained model to generate navigation commands in real-time.

## Instruction for Running the Project
Part of the instructions are adapted from [the AWS RoboMaker PR2 Hospital Sample Application](https://github.com/aws-samples/aws-robomaker-sample-application-pr2-hospital)

### System Requirements
This project is optimized for an Ubuntu 20.04 environment with a graphical user interface (GUI). Please note the following considerations:

- Ubuntu 22.04: Attempts to configure the project on Ubuntu 22.04 were unsuccessful, indicating potential compatibility issues.
- Ubuntu 18.04 on Mac M1: Configuring an Ubuntu 18.04 virtual machine with a GUI on the Mac M1 presented challenges. While Ubuntu 18.04 runs smoothly on EC2 instances on the Mac M1, the lack of a GUI in this setup limits its usability for this project.

For the best experience and compatibility, it is recommended to use Ubuntu 20.04 with a GUI.

### Clone the repository

```bash
git clone https://github.com/eamonma2001/hospital_delivery
```

### Install requirements
Install ROS Neotic and Colcon as well as the other requirements in the requirements.txt. 
```bash
cd hospital_delivery
pip install -r requirements.txt
```

### Pre-build commands

```bash
sudo apt-get update
rosdep update
```

### Build Robot Application

```bash
cd robot_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
```

### Build Simulation Application

```bash
cd simulation_ws
rosws update
rosdep install --from-paths src --ignore-src -r -y
colcon build
```

## Run
Launch the application with the following commands:

- *Running Simulation Application in Tab 1:*
    ```bash
    source simulation_ws/install/local_setup.sh
    roslaunch pr2_hospital_simulation view_hospital.launch
    ```

- *Running Robot Application in Tab 2:*
    ```bash
    source robot_ws/install/local_setup.sh
    export ROBOT="sim"
    roslaunch pr2_hospital_robot pr2_2dnav.launch
    ```
- *Running command.py with Desirable Tasks in Tab 3:*
    ```bash
    python3 command.py patient
    ```

    ```bash
    python3 command.py exam
    ```

    ```bash
    python3 command.py surgery
    ```
