# AutoClean Edge Grasp Detection

This repository consists of all the scripts and steps required for running the image capture, point cloud data (PCD) capture,
and the detection process with runs the combination of YOLOv7 and edge depth detection.
The end goal is the detection and identification of the graspable edge pose for the successful grasping of an irregular part in a post-processing line.

It is part of the AutoClean project, in which parts printed with Multi Jet Fusion (MJF) are cleaned and inspected automaticall in a processing line.

The image dataset used for YOLOv4 and YOLOv7 CNN training is also included.

### AutoClean automation process:

The complete AutoClean process can be summarized in the following steps:

1) Printed parts (referred to as parts) are extracted from the powder bed to a box.
2) They are brought to the vibrating plate and a definite number ('num_parts') is added.
3) The start of the process is confirmed by the operator.
4) Vibrating process starts and lasts for 't_vibrate' seconds.
5) Parts drop at the landing platform (where manipulator picks one by one)
6) The manipulator is at 'home_pose', where it awaits for the detection of parts.
7) When vibration ends after 't_vibrate' seconds, the manipulator is notified (for example, by 'success_A').
8) Edge detection pipeline is used to determine graspable edge i for each part N.
9) Manipulator grasping protocol is started. ('manipulator_A_B')
10) The manipulator places part on the receiving end of the 'conveyor_strahlkabine'.
11) Strahlkabine is notified by a signal, which start the fine cleaning process.
12) The manipulator transfers the part to the QC-Station.
13) The manipulator transfers the part to the corresponding batch/box, depending on the outcome of the QC-Station.