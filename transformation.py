#!/usr/bin/env python3

from scipy.spatial.transform import Rotation as R
from tools.computations import *
import numpy as np
import time
import open3d as o3d
from tools.ur_realtime import ur_ready, ur_receive, ur_move

np.set_printoptions(suppress=True)
debug_mode = False

"""
--> homogeneous matrix m_n (4x4), denoted as M_h to obtain point in reference frame m, given the point in reference frame n.
P_m = M_h . P_n , where P_m and P_n are vectors [x, y, z]^T
M_h = [rot_mat disp_mat]
"""

class RobotControl:
   def __init__(self, camera = "d405"):
      self.camera = camera # choose between d405 or d435 setup
      self.tcp_pose = ur_receive()
      self.P_tcp = self.tcp_pose[:3]
      self.R_tcp_aa = self.tcp_pose[-3:]
      # Convert angle-axis to rpy and rotation matrix:
      self.tcp_roll, self.tcp_pitch, self.tcp_yaw = rv2rpy(self.R_tcp_aa)
      self.B_T_R = self.obtain_base_to_tcp_transformation()
      self.R_T_C = self.transform_camera_in_robot_frame()
      
   def obtain_base_to_tcp_transformation(self):
      # B_T_R: Base to TCP  (Base to Robot-TCP)
      if self.tcp_pose:
         rx = Rx(self.tcp_roll)
         ry = Ry(self.tcp_pitch)
         rz = Rz(self.tcp_yaw)
         self.R_tcp = np.dot(np.dot(rz,ry), rx)
         return compute_transform_matrix(disp=self.P_tcp, rot=self.R_tcp)
      else:
         print(f"{time.time()*1000}: No TCP pose received. Robot not connected.")

   def get_tcp_pose(self):
      return self.tcp_pose
   
   def get_tcp_transform_matrix(self):
      return self.B_T_R

   def transform_camera_in_robot_frame(self, p_m=None, inverse=False):
      # dx, dy, dz are camera offsets
      # theta_x, theta_y, theta_z are the camera angular offsets (How camera is oriented about x,y,z-axis)
      if self.camera == "d405":
         dx,dy,dz = -0.1215,-0.0193,-0.1245
         theta_z = 90
         rot_mat = Rz(theta_z)
      elif self.camera == "d435i":
         dx,dy,dz = -0.075,0.0,-0.155
         theta_z = -90
         rot_mat = Rz(theta_z)
      else:
         print("Camera not recognized.")

      t_mat = np.eye(4)
      disp_mat = np.array([[dx],
                           [dy],
                           [dz]])
      t_mat[:3, :3] = rot_mat
      t_mat[:3,-1:] = disp_mat
      if debug_mode:
         print(f"transform: \n {t_mat}")
      
      # transform point in Camera Coordinate System to TCP (Robot) Coordinate System.
      # Example:
      if p_m is not None:
         transformed_point = np.dot(t_mat, p_m) if not inverse else np.dot(np.linalg.inv(t_mat), p_m)
         if debug_mode:
            print(f"transformed point: \n {transformed_point}")
         return transformed_point
      else:
         return t_mat if not inverse else np.linalg.inv(t_mat)

class ObjectGrasp:
   def __init__(self, tcp_pose, P_g, R_g):
      # Pose values in the form: [P, R]
      self.current_pose = tcp_pose
      self.tcp_pose = tcp_pose.tcp_pose # in robot base frame, P = x,y,z and R = a,b,c : axis angle values
      self.P_g = P_g # P = x,y,z in camera frame
      self.R_g = R_g # R = rotation matrix in camera frame

      # Pre-gras pose:
      self.pre_grasp_dist = -0.025 # Clearance away from the part for pre-grasp pose
      self.pre_P_g = None
      self.pre_C_T_P = None #transformation_matrix_object_pre_grasp_in_camera_frame (initialized)
      self.pre_grasp_pose_in_robot_frame = None

      # Grasp pose
      self.C_T_P = None #transformation_matrix_object_in_camera_frame (initialized) 
      self.R_T_P = None #transformation_matrix_object_in_tcp_frame (initialized)
      self.B_T_P = None #transformation_matrix_object_in_robot_frame

      # Rotational values
      self.P_roll, self.P_pitch, self.P_yaw = None, None, None
      self.P_roll_deg, self.P_pitch_deg, self.P_yaw_deg = None, None, None 

      # Final pose values
      self.target_pose_in_camera_frame = None
      self.target_pose_in_robot_frame = None # TCP pose in form [x,y,z, a,b,c] to send to UR-Robot
      self.initialize_grasp_point()
   
   def initialize_grasp_point(self):
      self.P_roll, self.P_pitch, self.P_yaw = compute_euler(self.R_g)[0]
      self.P_roll_deg, self.P_pitch_deg, self.P_yaw_deg = compute_euler(self.R_g)[1]    
      self.target_pose_in_camera_frame = [self.P_g[0], self.P_g[1], self.P_g[2], self.P_roll_deg, self.P_pitch_deg, self.P_yaw_deg]

      # General robot transformation (Not affected by pre-grasp or grasp)
      self.B_T_R = self.current_pose.B_T_R #transformation_matrix_robot_base_to_TCP (initialized from "RobotPose" class)
      self.R_T_C = self.current_pose.R_T_C #transformation_matrix_camera_in_robot_frame (initialized from "RobotPose" class - (initialized based on camera calibration))

      # Grasp:
      self.C_T_P = compute_transform_matrix(disp=self.P_g, rot=self.R_g)
      self.R_T_P = np.dot(self.R_T_C, self.C_T_P)
      self.B_T_P = np.dot(self.B_T_R, self.R_T_P)
      self.target_pose_in_robot_frame = convert_to_ur_format(self.B_T_P)

      # Pregrasp: 
      self.calculate_pre_grasp()

      if debug_mode:
         print(f"Rotation angles from R_p: {self.roll_deg, self.pitch_deg, self.yaw_deg}") # check angles in degrees
         print(f"Validate Robot Target Pose in camera frame: {self.target_pose_in_camera_frame}")
         print(f"Base to TCP grasp B_T_P: \n {self.B_T_P}")
         print(f"Current TCP Pose wrt. Base: \n {self.tcp_pose.tcp_pose}")
         print(f"Target TCP Pose wrt. Base: \n {self.target_pose_in_robot_frame}")

   def calculate_pre_grasp(self):
      #pre_grasp_dist: How far away the TCP pre-grasp pose is
      # Rotation vector R_p stays the same, so not returned 
      # Translation along local z-axis (pre-grasp for robot):
      disp_pre_grasp_axis = [0,0,1]
      # Define local-z axis direction in rotated frame:
      local_z_axis = np.dot(self.R_g, disp_pre_grasp_axis)
      translation_vector = self.pre_grasp_dist * local_z_axis
      P_ppg = self.P_g[:3] + translation_vector # pre_grasp_point
      self.pre_C_T_P = compute_transform_matrix(disp=P_ppg, rot=self.R_g)
      self.pre_R_T_P = np.dot(self.R_T_C, self.C_T_P)
      self.pre_B_T_P = np.dot(self.B_T_R, self.R_T_P)
      self.pre_grasp_pose_in_robot_frame = convert_to_ur_format(self.B_T_P)
      self.pre_P_g = P_ppg.tolist()[0]

   def grasp(self):           
      if ur_ready:
         ur_move(0,self.target_pose_in_robot_frame)
         time.sleep(2)
         # Close gripper
           
      else:
         print("No access to UR. Cannot Proceed.")
   
   def pre_grasp(self):
      if ur_ready:
      # Open gripper
      
         ur_move(0,self.pre_grasp_pose_in_robot_frame)
         time.sleep(0.1)  
      else:
         print("No access to UR. Cannot Proceed.")

if __name__ == "__main__":  
   tcp_pose1 = RobotControl()
   object1 = ObjectGrasp(tcp_pose1, P_grasp, R_grasp)

   #object1.pre_grasp()
   object1.grasp()
   #object1.move()
   #object1.pre_drop()
   #object1.drop()
