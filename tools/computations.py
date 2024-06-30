import math as m
from math import degrees as r2d
from math import radians as d2r
import numpy as np

debug_mode = True

"""
rpy2rv : rotation matrix values to axis-angle
rv2rpy: axis-angle to rotation matrix

about x-axis: roll --> gamma
about y-axis: pitch --> beta
about z-axis: yaw --> alpha
"""

def Rx(theta):
  theta=d2r(theta)
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  theta=d2r(theta)
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  theta=d2r(theta)
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def rpy2rv(roll,pitch,yaw):
    # arguments are in degrees, converted to radians
    alpha = d2r(yaw)
    beta = d2r(pitch)
    gamma = d2r(roll)
    
    ca = m.cos(alpha)
    cb = m.cos(beta)
    cg = m.cos(gamma)
    sa = m.sin(alpha)
    sb = m.sin(beta)
    sg = m.sin(gamma)
    
    r11 = ca*cb
    r12 = ca*sb*sg-sa*cg
    r13 = ca*sb*cg+sa*sg
    r21 = sa*cb
    r22 = sa*sb*sg+ca*cg
    r23 = sa*sb*cg-ca*sg
    r31 = -sb
    r32 = cb*sg
    r33 = cb*cg
    
    theta = m.acos((r11+r22+r33-1)/2)
    if d2r(theta)==90:
       theta=r2d(89.999999)
       sth = m.sin(theta)
       kx = (r32-r23)/(2*sth)
       ky = (r13-r31)/(2*sth)
       kz = (r21-r12)/(2*sth)
    elif d2r(theta) == 0:
       kx, ky, kz = 0,0,0
    else:
       sth = m.sin(theta)
       kx = (r32-r23)/(2*sth)
       ky = (r13-r31)/(2*sth)
       kz = (r21-r12)/(2*sth)

    # output is in radians
    return [(theta*kx),(theta*ky),(theta*kz)]

def rv2rpy(r_aa):
    rx,ry,rz = r_aa # in radians
    theta = m.sqrt((rx*rx) + (ry*ry) + (rz*rz)) # in radians
    theta = 0.00001 if theta==0 else theta
    kx = rx/theta
    ky = ry/theta
    kz = rz/theta
    cth = m.cos(theta)
    sth = m.sin(theta)
    vth = 1-m.cos(theta)
    
    r11 = kx*kx*vth + cth
    r12 = kx*ky*vth - kz*sth
    r13 = kx*kz*vth + ky*sth
    r21 = kx*ky*vth + kz*sth
    r22 = ky*ky*vth + cth
    r23 = ky*kz*vth - kx*sth
    r31 = kx*kz*vth - ky*sth
    r32 = ky*kz*vth + kx*sth
    r33 = kz*kz*vth + cth
    
    beta = m.atan2(-r31,m.sqrt(r11*r11 + r21*r21))
    
    if beta > d2r(89.99):
        beta = d2r(89.99)
        alpha = 0
        gamma = m.atan2(r12,r22)
    elif beta < -d2r(89.99):
        beta = -d2r(89.99)
        alpha = 0
        gamma = -m.atan2(r12,r22)
    else:
        cb = m.cos(beta)
        alpha = m.atan2(r21/cb,r11/cb)
        gamma = m.atan2(r32/cb,r33/cb)

    return [r2d(gamma),r2d(beta),r2d(alpha)]

def compute_euler(rotation_matrix):
   """ test values:
   rx = Rx(197.123)
   ry = Ry(12.12)
   rz = Rz(18.12145)
   rm = np.dot(np.dot(rz,ry), rx) """

   if debug_mode:
      print(f"rotation matrix: \n {rotation_matrix}")

   # Extract roll, pitch, and yaw angles
   roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
   pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
   yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

   # Convert angles to degrees if needed
   roll_deg = np.degrees(roll)
   pitch_deg = np.degrees(pitch)
   yaw_deg = np.degrees(yaw)

   if debug_mode:
      print("Roll (radians):", roll)
      print("Pitch (radians):", pitch)
      print("Yaw (radians):", yaw)

      print("Roll (degrees):", roll_deg)
      print("Pitch (degrees):", pitch_deg)
      print("Yaw (degrees):", yaw_deg)
   
   return [[roll, pitch, yaw],[roll_deg, pitch_deg, yaw_deg]]

   # values returned in degrees
   #return roll_deg, pitch_deg, yaw_deg

def compute_transform_matrix(disp=[0,0,0], rot=np.eye(3)):
      # disp is true_pre_grasp_pose[0]
      # rot is true_pre_grasp_pose[1]
      dx,dy,dz = disp
      t_mat = np.eye(4)
      disp_mat = np.array([[dx],
                           [dy],
                           [dz]])
      t_mat[:3, :3] = rot
      t_mat[:3,-1:] = disp_mat
      return t_mat

def convert_to_ur_format(transform_matrix):
   # 6 value TCP Coordinate in angle-axis form Robot:
   ur_pose = transform_matrix[:, -1][:3].tolist()
   # Convert rotation matrix R_T_P[:3, :3] to rpy for UR:
   pose_rot = compute_euler(transform_matrix[:3, :3])[1]
   roll, pitch, yaw = pose_rot
   pose_axis_angle_rot = rpy2rv(roll, pitch, yaw)
   ur_pose.extend(pose_axis_angle_rot) # Output is in radians
   if debug_mode:
      print(f"In UR axis-angle format: \n Pose: {ur_pose} \n Angle Clarification: {pose_rot}")

   return ur_pose

def calculate_midpoints(bbox):
    bbox_vertices = np.asarray(bbox.get_box_points())
    midpoints = []
    mid_points = []
    # Calculate midpoints along the lengths
    midpoints.append((bbox_vertices[0] + bbox_vertices[1]) / 2)
    midpoints.append((bbox_vertices[1] + bbox_vertices[3]) / 2)
    midpoints.append((bbox_vertices[0] + bbox_vertices[2]) / 2)
    midpoints.append((bbox_vertices[2] + bbox_vertices[3]) / 2)

    midpoints.append((bbox_vertices[4] + bbox_vertices[5]) / 2)
    midpoints.append((bbox_vertices[5] + bbox_vertices[7]) / 2)
    midpoints.append((bbox_vertices[4] + bbox_vertices[6]) / 2)
    midpoints.append((bbox_vertices[6] + bbox_vertices[7]) / 2)

    midpoints.append((bbox_vertices[1] + bbox_vertices[5]) / 2)
    midpoints.append((bbox_vertices[5] + bbox_vertices[7]) / 2)
    midpoints.append((bbox_vertices[1] + bbox_vertices[3]) / 2)
    midpoints.append((bbox_vertices[3] + bbox_vertices[7]) / 2)

    # Calculate midpoints along the depths
    #mid_points.append((midpoints[0] + midpoints[1]) / 2)
    #mid_points.append((midpoints[4] + midpoints[5]) / 2)
    mid_points.append(midpoints[1])
    mid_points.append(midpoints[2])
    mid_points.append(midpoints[5])
    mid_points.append(midpoints[7]) # 7
    return mid_points

def transform_dict_values(key, value):
    W,H=1280,720 # image size
    if key in ['x1','x2','center_x', 'width']:
        return int(value*W)
    elif key in ['y1','y2','center_y', 'height']:
        return int(value*H)

def convert_to_pixel(edges):
    for i,edge in enumerate(edges):
        box_position = edge['relative_coordinates']
        key_mapping = {'center_x':'x1',
                       'center_y':'y1',
                       'width':'x2',
                       'height':'y2'}
        
        # remember dictionaries and other mutable objects are passed by reference, not passed by value! 
        box_position_calc = box_position.copy()
        
        box_position_calc["center_x"] -= (box_position["width"]/2)
        box_position_calc["center_y"] -= (box_position["height"]/2)
        box_position_calc["width"] = box_position_calc["center_x"] + (box_position["width"]/2)
        box_position_calc["height"] = box_position_calc["center_y"] + (box_position["height"]/2)

        box_position_new = {key_mapping.get(k,k): transform_dict_values(key_mapping.get(k,k), v) for k, v in box_position_calc.items()}       
        #box_position = {k: transform_dict_values(k,v) if k in ['center_x','center_y', 'width', 'height'] else v for k, v in box_position.items()}
        box_position = {k: transform_dict_values(k,v) for k, v in box_position.items()}

        edge['actual_coordinates'] = box_position_new
        edge['relative_coordinates'] = box_position

    return edges