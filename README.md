# AutoClean Edge Grasp Detection

This repository consists of all the scripts and steps required for running the image capture, point cloud data (PCD) capture,
and the detection process which is a combination of YOLOv4/YOLOv7 region proposal and edge depth detection.
The end goal is the detection and identification of the graspable edge pose for the successful grasping of an irregular part in a post-processing line.

It is part of the "[AutoClean](https://zaf.th-deg.de/public/project/286)" project, in which parts printed with Multi Jet Fusion (MJF) are cleaned and inspected autonomously in a processing line.

### Prerequisites:
```detection_sequence.sh``` executes the python scripts ```image_capture.py```, ```depth_capture.py``` and ```edge_depth.py``` which uses the "pyrealsense2" python package.
This package requires access to the camera to successfully execute the data capture processes.
 
The [darket](https://github.com/AlexeyAB/darknet) repository is required to run YOLOv4/YOLOv7 in ```yolo.sh```.

### Running the detection
```detection_sequence.sh``` runs the edge detection process, starting with the image and depth data capture.
The output of the YOLO CNN saves the edge ROI in detection.json.
The edge ROI is used to further process the depth data to obtain the graspable edge.
```edge_depth.py``` returns the TCP grasp pose in the form `[x, y, z, a, b, c]` where `[x, y, z]` is the positional
coordinates and `[a, b, c]` is the orientaion in axis-angle notation.

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
