import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import datetime
import os
import time
import glob
import sys, json, cv2, itertools

for arg in sys.argv[1:]:
    print(f"capture pcd to: {arg}")

#pipeline = rs.pipeline()

#folderpath = os.getcwd()
folderpath = os.path.dirname(os.path.abspath(__file__))
new_directory = sys.argv[1] if len(sys.argv) > 1 else folderpath # only the first argument is taken if multiple are given #to do: use multiple directories!

# replace teh old existing directory with the new one
new_path = os.path.join(folderpath.rsplit(os.path.sep, 1)[0], new_directory)
folderpath = new_path
os.makedirs(new_path, exist_ok=True) #make the new path

folder_name = "pcd" # subdirectory for storing pointclouds changed from data to pcd
#subpath = "scripts/datacapture/"+folder_name # when runnning from vscode
subpath = folderpath+"/"+folder_name # when running from terminal / bash script

# create a folder if it doesn't exist to store images
if not os.path.exists(subpath):
    os.makedirs(subpath)
else:
    pass

def load_file(file_type):
    folderpath = os.path.dirname(os.path.abspath(__file__))
    file_type = str(file_type)
    files = glob.glob(folderpath+file_type)
    latest_file = max(files, key=os.path.getctime)
    print(f"File found at: {latest_file}")
    if file_type == '/*.json':
        with open(latest_file) as json_file:
            detected_objects = json.load(json_file)
        return detected_objects
    elif file_type == '/images/*.jpg':
        img = cv2.imread(latest_file)
        return img
    elif file_type == '/pcd/*.pcd':
        pcd = o3d.io.read_point_cloud(latest_file)
        return pcd
    else:
        print("file type unrecognized.")
        return None

def run_camera():
    print("checking device connection ... ")
    
    pipeline = rs.pipeline()

    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)

    #config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)  # Set the resolution and format of the color stream
    #check device connection first:
    check = None
    try:
        check = pipeline.start(cfg)
    except RuntimeError:
        print("no device detected.")
        
    if check:
        pipeline.stop() # stop pipeline and start again outside of exception
        t = 0.15
        print(f"sleeping for {t} seconds")
        time.sleep(t)
        config = pipeline.start(cfg)
        # to set higher density capture
        dev = config.get_device()
        depth_sensor = dev.first_depth_sensor()
        #print(f"Current preset: {depth_sensor.get_option(rs.option.visual_preset)}")
        preset = 4
        depth_sensor.set_option(rs.option.visual_preset,preset)
        range = depth_sensor.get_option_range(rs.option.visual_preset)
        #print(f"range: {range} - current preset name: {depth_sensor.get_option_value_description(rs.option.visual_preset, preset)}")

        #return True
        return pipeline
    else:
        return None
    
def direct_capture():
    if run_camera():
        
        pipeline = rs.pipeline()
        def reset():
            #pipeline.stop()
            ctx = rs.context()
            dev_list = ctx.query_devices()
            for dev in dev_list:
                dev.hardware_reset()
            time.sleep(0.5)
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
        pipeline.start(cfg)
        st = time.time()
        while time.time() - st < 20:
            try:
                frames = pipeline.wait_for_frames()
                break
            except:
                reset()
        print(f"direct pcd capture at {time.time()} ")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        depth_frame = frames.get_depth_frame()
        pc = rs.pointcloud()
        pointcloud = pc.calculate(depth_frame)
        points = np.asanyarray(pointcloud.get_vertices())

        pc2 = rs.pointcloud()
        pcd = pc2.calculate(depth_frame)
        #filename = f'direct_save_{timestamp}.ply'
        #pcd.export_to_ply(filename, color_frame)
        
        pipeline.stop()

        return points,pcd
    else:
        return None

def capture_pc(pipeline):
    def reset():
        pipeline.stop()
        ctx = rs.context()
        dev_list = ctx.query_devices()
        for dev in dev_list:
            dev.hardware_reset()
        time.sleep(0.5)
    print(f"capturing point cloud at {time.time()} ")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        frames = pipeline.wait_for_frames()
    except RuntimeError:
        reset()
        frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    #color_frame = frames.get_color_frame()
    pc = rs.pointcloud()
    pointcloud = pc.calculate(depth_frame)
    points = np.asanyarray(pointcloud.get_vertices())
    
    pc2 = rs.pointcloud()
    pcd = pc2.calculate(depth_frame)
    filename = f'direct_save_{timestamp}.ply'
    #pcd.export_to_ply(filename, color_frame)
    
    pipeline.stop()

    return points,pcd

def save_points_to_ply(pc):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    points = pc
    filename = f'captured_{timestamp}.ply'
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(points)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property float scalar_Intensity\n')
        f.write('end_header\n')
        for p in points:
            if (abs(p[0]) or abs(p[1]) or abs(p[2])) != 0:
                f.write('{} {} {}\n'.format(p[0], p[1], p[2]))
            else:
                pass
            
        print(f"PCD captured as .ply at {timestamp}")

def save_points_to_pcd(pc):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    points = pc
    filename = f'pcd_{int(time.time()*1000)}.pcd'
    filename = os.path.join(folderpath,subpath,filename)
    
    with open(filename, 'w') as f:
        f.write('# .PCD v0.7 - Point Cloud Data\n')
        f.write('VERSION 0.7\n')
        f.write('FIELDS x y z\n')
        f.write('SIZE 4 4 4\n')
        f.write('TYPE F F F\n')
        f.write(f'WIDTH {len(points)}\n')
        f.write('HEIGHT 1\n')
        f.write('POINTS {}\n'.format(len(points)))
        f.write('DATA ascii\n')
        for p in points:
            if True:#(abs(p[0]) or abs(p[1]) or abs(p[2])) != 0:
                f.write('{} {} {}\n'.format(p[0], p[1], p[2]))
            else:
                pass
            
        print(f"PCD captured as .pcd at {int(time.time()*1000)}")

def display_pcd(*filenames):
    geo_list = []
    for filename in filenames:
        if isinstance(filename, o3d.cpu.pybind.geometry.PointCloud):
            geo_list.append(filename)
        else:
            pcd = o3d.io.read_point_cloud(filename)
            geo_list.append(pcd)
    o3d.visualization.draw_geometries(geo_list)
    
def capture_window_pcd(max_count):
    #get number of files in the subpath:
    files = glob.glob(os.path.join(folderpath,subpath, f"*.pcd"))
    #sort files by generated time:
    files.sort(key=os.path.getmtime)
    if len(files)>max_count:
        num_files_to_delete = len(files) - max_count
        for i in range(num_files_to_delete):
            try:
                os.remove(files[i])
                print(f"deleted {files[i]}")
            except Exception as e:
                print(f"error deleting files outside capture window: {e}")

def crop_pcd(pcd):
    universal_z_min = 0.45
    universal_z_max = 1
    crop_limits = [
        [-0.50,0.28], # left right -x: -0.28 +x: 0.28
        [-0.14,0.14], # down up -y +y
        [universal_z_min,universal_z_max] # z near far -(-z +z)
    ]
    
    crop_limits_points = list(itertools.product(*crop_limits))
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(crop_limits_points))
    cropped_pcd = pcd.crop(bounding_box)
    return cropped_pcd

def planedetect(pcd_data, dist_thresh):
    #initially was 0.1
    plane_model, inliers = pcd_data.segment_plane(distance_threshold=dist_thresh,ransac_n=5,num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    z = -(a*0 + b*0 + d)/c
    inlier_cloud = pcd_data.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    outlier_cloud = pcd_data.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 1, 0])
    #distances = inlier_cloud.compute_point_cloud_distance(outlier_cloud)
    cloud_array = np.asarray(outlier_cloud.points)
    z_ = np.asarray(outlier_cloud.points)[:,2]
    mask = z_<= z
    filtered_cloud = cloud_array[mask]
    z_coordinates=filtered_cloud[:,2]
    print(f"z: {z_coordinates}")
    if any(z_coordinates):
        greatest_offset = np.max(np.abs(z_coordinates - z))
        vibrateplate_empty = True
    else:
        greatest_offset = 0.0
        vibrateplate_empty=False

    return inlier_cloud,outlier_cloud, z, greatest_offset, vibrateplate_empty

def main():
    pcdstate = True

    if pcdstate:
        rspipeline=run_camera()
        if rspipeline:
            points,pcd = capture_pc(rspipeline)
            corrected_points = points
            save_points_to_pcd(corrected_points)
            capture_window_pcd(100)
            #display_pcd(load_file('/pcd/*.pcd'))
        else:
            pass
    else:
        pass

if __name__ == "__main__":
    
    main()

    #display_pcd(crop_pcd(load_file('/pcd/*.pcd')))
        
