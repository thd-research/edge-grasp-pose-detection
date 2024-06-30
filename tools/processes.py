import open3d as o3d
import json
import glob
import os
import itertools
import numpy as np
import pickle
import cv2

debug_mode = True

def save_instance_to_json(instance, file_name = 'camera_instance.json'):
    folderpath = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(folderpath, '..')
    file_path = os.path.join(file_path, file_name)
    with open(file_path, 'w') as file:
        json.dump(instance.to_dict(), file, indent=4)
        print(f"instance saved to {file_name} to {file_path}.")

def save_instance(instance, file_name=None):
    folderpath = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(folderpath, '..')
    file_name = 'camera_parameters.pkl' if file_name is None else file_name   
    file_path = os.path.join(file_path, file_name)
    print(f"Instance saved to {file_path}.")
    # Serialize and save an instance of a class for offline camera mode
    with open(file_path, 'wb') as file:
        pickle.dump(instance, file)

def load_camera_instance_from_json():
    # Import here to prevent circular import Error!
    from check_calibration import CameraApp
    file_name = 'camera_instance.json'
    folderpath = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(folderpath, '..')
    file_path = os.path.join(file_path, file_name)
    try:
        with open(file_path, 'r') as file:
            if debug_mode:
                print(f"File found at {file_path}")
            data = json.load(file)
            if debug_mode:
                print(f"data loaded from camera_instance: {data}")
            return CameraApp.from_dict(data)
    except FileNotFoundError:
        print("No camera instance saved yet.")
        return None

def load_file(file_type):
    folderpath = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(folderpath, '..')
    file_type = str(file_type)
    files = glob.glob(file_path+file_type)
    latest_file = None
    if files:
        latest_file = max(files, key=os.path.getctime)
    if debug_mode:
        print(f"Files found: {files}")
        print(f"Latest file found at: {latest_file}")
    if file_type == '/roi/*.json':
        with open(latest_file) as json_file:
            detected_objects = json.load(json_file)
        return detected_objects[0]
    elif file_type == '/images/*.jpg':
        img = cv2.imread(latest_file)
        return img
    elif file_type == '/pcd/*.pcd':
        pcd = o3d.io.read_point_cloud(latest_file)
        return pcd
    elif file_type == 'camera_parameters.pkl':
        file_path = os.path.join(file_path,file_type)
        if debug_mode:
            print(f"Current file path is {file_path}")
        try:
            with open(file_path, 'rb') as file:
                loaded_camera_parameters = pickle.load(file)
                return loaded_camera_parameters
        except FileNotFoundError:
            print(f"No existing camera_parameters.pkl found.")
            return None
    else:
        print("file type unrecognized. Trying custom retrieval...")
        file_path = os.path.join(file_path,file_type)
        if debug_mode:
            print(f"Current file path is {file_path}")
        try:
            with open(file_path) as json_file:
                retrieved_data = json.load(json_file)
                print(f"Chosen file found at: {file_path}")
            return retrieved_data
        except FileNotFoundError:
            print(f"No file: {file_type} found.")
            return None

def planedetect(pcd_data, dist_thresh=0.0075):
    plane_model, inliers = pcd_data.segment_plane(distance_threshold=dist_thresh,  #initially was 0.1
                                         ransac_n=5,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    z = -(a*0 + b*0 + d)/c
    inlier_cloud = pcd_data.select_by_index(inliers)
    #inlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    outlier_cloud = pcd_data.select_by_index(inliers, invert=True)
    #outlier_cloud.paint_uniform_color([0, 1, 0])
    return inlier_cloud,outlier_cloud, z, inliers, plane_model

def pcd_crop(pcd, lims =   [[-0.25,0.15], # left right -x +x
                            [-0.16,0.12], # down up -y +y
                            [0.2,0.65]] ):
    crop_limits_points = list(itertools.product(*lims))
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(crop_limits_points))
    cropped_pcd = pcd.crop(bounding_box)
    return cropped_pcd

def pixels_to_m(img_coords):
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        cfg = rs.config()  
        cfg.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
        config = pipeline.start(cfg)
        dev = config.get_device()
        depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
        depth_sensor = dev.first_depth_sensor()
        preset = 4
        depth_sensor.set_option(rs.option.visual_preset,preset)
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("No image from camera.")
            pipeline.stop()
            return 0.,0.
        else:
            x,y = img_coords
            x,y = int(x)-20, int(y)-45
            if debug_mode:
                print(f"Values to check: {x,y}")
            Z = depth_frame.get_distance(x,y)
            intrinsic = color_frame.profile.as_video_stream_profile().get_intrinsics()
            fx, fy, cx, cy = intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy
            # check if fx = 212.595139 and fy = 212.595139; ppx = 214.363174 and ppy = 119.456047
            # resolution of depth: 424 x 240
            depth_scale = depth_sensor.get_depth_scale()
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy

            print(f"X,Y,Z: {X,Y,Z}")
            pipeline.stop()
            return X,Y,Z

def create_coordinates_frame(center, rotation=np.eye(3)):
    # Create a coordinate frame at the center with orientation from the rotation matrix
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.020)
    coordinate_frame.translate(center)
    coordinate_frame.rotate(rotation, center=center)
    return coordinate_frame

def create_locator(center,rgb=[0.25,0.25,0.25]):
    locator = o3d.geometry.TriangleMesh.create_box(width=0.005, height=0.005, depth=0.005)
    locator.paint_uniform_color([rgb[0], rgb[1], rgb[2]]) 
    locator.translate(center)
    return locator
