import cv2
import json, time, datetime, sys, os, glob
from tools.processes import save_instance
import pyrealsense2 as rs
import numpy as np
import threading
import open3d as o3d
import itertools

# Toggle Methods 1 (own) and 2 (online)
method = 1
# Toggle use "nominal_depth" or "variable_height"
variable_height = True

# Paths
folderpath = os.path.dirname(os.path.abspath(__file__))
new_directory = sys.argv[1] if len(sys.argv) > 1 else folderpath # only the first argument is taken if multiple are given #to do: use multiple directories!
new_path = os.path.join(folderpath.rsplit(os.path.sep, 1)[0], new_directory)
folderpath = new_path
os.makedirs(new_path, exist_ok=True) #make the new path
folder_name = "pcd" # subdirectory for storing pointclouds changed from data to pcd
#subpath = "scripts/datacapture/"+folder_name # when runnning from vscode
subpath = folderpath+"/"+folder_name # when running from terminal / bash script
subpath_json = folderpath
# create a folder if it doesn't exist to store images
if not os.path.exists(subpath):
    os.makedirs(subpath)
else:
    pass

# Add comparison to PCD and place locator in cropped PCD.

# Configure depth and color streams
"""
pipeline = rs.pipeline()
config = rs.config()
imgw = 1280
imgh = 720
config.enable_stream(rs.stream.depth, imgw, imgh, rs.format.z16, 30)
config.enable_stream(rs.stream.color, imgw, imgh, rs.format.bgr8, 30)
"""

class Locator3d:
    def __init__(self, point):
        self.point = point
    
    def create_locator(self):
        self.locator = o3d.geometry.TriangleMesh.create_box(width=0.005, height=0.005, depth=0.005)
        self.locator.paint_uniform_color([0.6, 0.0, 1.0]) 
        self.locator.translate(self.point)
        return self.locator

class PointCloudData:
    def __init__(self, points):
        self.pcd = None
        self.geometries = []
        self.locators = []
        self.points = points
        self.pipeline = rs.pipeline()
        if not self.pipeline:
            raise Exception("Error: Could not open camera.")
        
    def pcd_capture(self, pcdstate = True):
        def pcd_crop(pcd, lims =   [[-0.30,0.30], # left right -x +x
                            [-0.16,0.12], # down up -y +y
                            [0.2,0.65]] ):
            
            crop_limits_points = list(itertools.product(*lims))
            bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(crop_limits_points))
            cropped_pcd = pcd.crop(bounding_box)

            return cropped_pcd
        
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
                print("File type unrecognized.")
                return None
    
        def run_camera_high_density(start=True):
            if not start:
                self.pipeline.stop()
                return False
            else:
                cfg = rs.config()  
                cfg.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
                cfg.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
                # apply high density PCD
                config = self.pipeline.start(cfg)
                dev = config.get_device()
                depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
                depth_sensor = dev.first_depth_sensor()
                preset = 4
                depth_sensor.set_option(rs.option.visual_preset,preset)
                return True         
            
        def capture_pc():
            def reset():
                self.pipeline.stop()
                ctx = rs.context()
                dev_list = ctx.query_devices()
                for dev in dev_list:
                    dev.hardware_reset()
                time.sleep(0.5)
            print(f"Capturing point cloud at {time.time()} ")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            try:
                frames = self.pipeline.wait_for_frames()
            except RuntimeError:
                reset()
                frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            #color_frame = frames.get_color_frame()
            pc = rs.pointcloud()
            pointcloud = pc.calculate(depth_frame)
            points = np.asanyarray(pointcloud.get_vertices())

            pc2 = rs.pointcloud()
            pcd = pc2.calculate(depth_frame)
            filename = f'direct_save_{timestamp}.ply'
            #pcd.export_to_ply(filename, color_frame)

            #self.pipeline.stop()

            return points,pcd
        
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
        
        if pcdstate:
            rspipeline=run_camera_high_density()
            if rspipeline:
                points,pcd = capture_pc()
                corrected_points = points
                save_points_to_pcd(corrected_points)
                capture_window_pcd(1)
                time.sleep(0.5)
                run_camera_high_density(start=False)
                self.pcd = pcd_crop(load_file('/pcd/*.pcd'))
                for point in self.points:
                    locator = Locator3d(point)
                    self.locators.append(locator.create_locator())
                self.geometries.append(self.pcd)
                for locator_in_pcd in self.locators:
                    self.geometries.append(locator_in_pcd)
                print(f"geometries: {self.geometries} - pcd: {self.pcd} - locator: {self.locators}")
            else:
                pass
        else:
            pass

    def show(self):
        o3d.visualization.draw_geometries(self.geometries)

class Dot:
    def __init__(self, x, y, pixel_value, coord_3d):
        self.x = x
        self.y = y
        self.pixel_value = pixel_value
        self.coord_3d = coord_3d

    def get_coordinates(self):
        return (self.x, self.y)

    def get_pixel_value(self):
        return self.pixel_value

    def get_coord_3d(self):
        return self.coord_3d

    def to_dict(self):
        return {"coordinates": [self.x, self.y], "coord_3d": self.coord_3d}

class CameraApp:
    def __init__(self, serial_number='218622275676', fx=None, fy=None, cx=None, cy=None, nominal_depth=None, max_dots=5):
        self.serial_number = serial_number
        self.context = rs.context()
        devices = self.context.query_devices()
        for device in devices:
            if device.get_info(rs.camera_info.serial_number) == self.serial_number:
                print(f"{self.serial_number} camera found.")
            else:
                print(f"No {self.serial_number} camera found.")
        self.max_dots = max_dots
        self.dots = []
        self.pipeline = rs.pipeline()
        if not self.pipeline:
            raise Exception("Error: Could not open camera.")
        #cv2.namedWindow('Camera')
        self.center_dot = None
        self.cfg = rs.config() 
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
        self.cfg.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
        self.nominal_depth = nominal_depth
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def initialize_camera(self):
        initialization_time = 5 # seconds
        config = self.pipeline.start(self.cfg)
        dev = config.get_device()
        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        depth_sensor = dev.first_depth_sensor()
        preset = 4
        depth_sensor.set_option(rs.option.visual_preset,preset)
        start_time = time.time()
        try:
            print("Initializing camera...")
            while time.time() - start_time < initialization_time:
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
                self.frame__height, self.frame_width, _ = frame.shape
                x,y = self.frame_width // 2, self.frame__height // 2
                self.nominal_depth = depth_frame.get_distance(x,y)
                intrinsic = color_frame.profile.as_video_stream_profile().get_intrinsics()
                self.fx, self.fy, self.cx, self.cy = intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy
            if self.nominal_depth!=0:
                print(f"Nominal depth registered at {x,y}: {self.nominal_depth} m.")
            else:
                print("Nominal depth registered as 0.0 . No possibility to estimate X,Y.")
        finally:
            self.pipeline.stop()
        
    def to_dict(self):
        return {
                "serial_number": self.serial_number,
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
                "nominal_depth": self.nominal_depth
                }
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["serial_number"], data["fx"], data["fy"], data["cx"], data["cy"], data["nominal_depth"])

    def set_nominal_depth(self, frame, depth_frame):
        height, width, _ = frame.shape
        x,y = width // 2, height // 2
        self.nominal_depth = depth_frame.get_distance(x,y)
        if self.nominal_depth!=0:
            print(f"Nominal depth registered at {x,y}: {self.nominal_depth} m.")
        else:
            print("Nominal depth registered as 0.0 . No possibility to estimate X,Y.")

    def get_nominal_depth(self):
        return self.nominal_depth
    
    def get_depth(self, coord, depth_frame):
        # Use a grid-average system to get better value maybe ? 
        x,y = coord
        return depth_frame.get_distance(x,y)
    
    def obtain_coordinates_3d(self, color_frame,depth_frame, x,y):
        if method==1:
            intrinsic = color_frame.profile.as_video_stream_profile().get_intrinsics()
            self.fx, self.fy, self.cx, self.cy = intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy
            if variable_height:
                Z = depth_frame.get_distance(x,y)
                X = (x - self.cx) * Z / self.fx
                Y = (y - self.cy) * Z / self.fy
            else:
                X = (x - self.cx) * self.nominal_depth / self.fx
                Y = (y - self.cy) * self.nominal_depth / self.fy
            return (X,Y)
        elif method==2:
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            depth = depth_frame.get_distance(x, y)
            depth_point = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [x, y], depth)
            X,Y,_ = depth_point
            return (X,Y)

    def initialize_center_dot(self, frame, color_frame, depth_frame):
        height, width, _ = frame.shape
        self.set_nominal_depth(frame, depth_frame)
        Z = self.get_nominal_depth()
        X,Y = self.obtain_coordinates_3d(color_frame, depth_frame, width // 2, height // 2)
        self.center_dot = Dot(width // 2, height // 2, (255, 0, 0), (X,Y,Z))  # Blue dot for center

    def add_dot(self, x, y, pixel_value, coordinates_3d):
        if len(self.dots) >= self.max_dots:
            self.dots.pop(0)
        self.dots.append(Dot(x, y, pixel_value, coordinates_3d))
        self.store_dot(self.dots[-1])

    def store_dot(self, dot):
        data = dot.to_dict()
        json_file_path = os.path.join(subpath_json, 'pixel_data.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file)
        print(f"Stored {data} in {json_file_path}")

    def draw_dots(self, frame):
        height, width, _ = frame.shape
        self.center_dot.x, self.center_dot.y = width // 2, height // 2
        cv2.circle(frame, self.center_dot.get_coordinates(), 5, (255, 0, 0), -1)  # Blue center dot

        for dot in self.dots:
            x, y = dot.get_coordinates()
            #pixel_value = dot.get_pixel_value()
            coord_3d = dot.get_coord_3d()
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red dot
            text_position = f"({x}, {y})"
            X,Y,Z = coord_3d
            text_value = f"({X:.3f}, {Y:.3f}, {Z:.3f})"

            dx1 = 10 if x < 1170 else -100
            dy1 = 10 if y > 22 else -25
            cv2.putText(frame, text_position, (x + dx1, y - dy1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            dx2 = 10 if x < 1080 else -200
            dy2 = 10 if y < 700 else -25
            cv2.putText(frame, text_value, (x + dx2, y + dy2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            color_frame = param['color_frame']
            depth_frame = param['depth_frame']
            frame = param['frame']
            #pixel_value = frame[y, x].tolist()  # Convert to list for JSON serialization
            pixel_value = frame[y, x].tolist()  # Convert to list for JSON serialization
            print(f"Clicked at ({x}, {y}) with pixel value {pixel_value}")
            X,Y = self.obtain_coordinates_3d(color_frame, depth_frame, x,y)
            Z = self.get_depth((x,y), depth_frame)
            self.add_dot(x, y, pixel_value, (X,Y,Z))
    
    def capture_image():
        pass

    def capture_pcd():
        pass
   
    def run(self):
        # apply high density PCD
        config = self.pipeline.start(self.cfg)
        dev = config.get_device()
        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        depth_sensor = dev.first_depth_sensor()
        preset = 4
        depth_sensor.set_option(rs.option.visual_preset,preset)
        try:
            while True:

                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                #depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                if self.center_dot is None:
                    self.initialize_center_dot(color_image, color_frame, depth_frame)

                self.draw_dots(color_image)

                cv2.imshow('Camera', color_image)
                cv2.setMouseCallback('Camera', self.mouse_callback, {'frame': color_image, 
                                                                     'color_frame':color_frame,
                                                                     'depth_frame':depth_frame})

                
                key = cv2.waitKey(1)
                if key == ord('q'): # q for quit
                    break
                elif key == ord('r'): # r for reset
                    self.dots.clear()
                elif key == ord('m'): # m for move
                    # Check if there are currently dots, and use the latest dot ato move to:
                    if not self.dots:
                        print("No dots to move to. Left-click on the stream somewhere first, then press 'm'.")
                    else:
                        cv2.destroyAllWindows()
                        self.pipeline.stop()
                        break
                        
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            move_points = []
            for dot in self.dots:
                move_points.append(dot.get_coord_3d())
            pcd = PointCloudData(move_points)
            pcd.pcd_capture()
            pcd.show()

            #self.cap.release()
            cv2.destroyAllWindows()
        finally:
            self.pipeline.stop()

if __name__ == "__main__":
    app = CameraApp()
    app.initialize_camera()
    app.run()
 

