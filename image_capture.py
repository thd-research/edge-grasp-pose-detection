import cv2
import numpy as np
import pyrealsense2 as rs
import datetime
import os
import time
import glob
import sys

for arg in sys.argv[1:]:
    print(f"capture image to: {arg}.")

pipeline = rs.pipeline()

# with os.getcwd(), if the python script is invoked by a bash script at a different directory, then that dir is taken!
#folderpath = os.getcwd()
# the command below gives the absolute path where the python script resides, irrespective of how it is invoked
# in image_capture_v3.py, the directory is changed to the passed argument, unless not stated, which then saves in the /datacapture dir
folderpath = os.path.dirname(os.path.abspath(__file__))
new_directory = sys.argv[1] if len(sys.argv) > 1 else folderpath # only the first argument is taken if multiple are given #to do: use multiple directories!

# replace teh old existing directory with the new one
new_path = os.path.join(folderpath.rsplit(os.path.sep, 1)[0], new_directory)
folderpath = new_path
os.makedirs(new_path, exist_ok=True) #make the new path

folder_name = "images"
#subpath = "scripts/datacapture/"+folder_name # when runnning from vscode
subpath = folderpath+"/"+folder_name # when running from terminal / bash script

# create a folder if it doesn't exist to store images
if not os.path.exists(subpath):
    os.makedirs(subpath)
else:
    pass

def run_camera():
    print("checking device connection ... ")
    #pipeline = rs.pipeline()
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
        pipeline.start(cfg)
        return True
    else:
        return None
    
def capture_image(timestamp):
    print(f"capturing image at {time.time()} ")
    frame = pipeline.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,
                                     alpha = 0.5), cv2.COLORMAP_JET)
    h,w,channels = color_image.shape

    filename = f"image_{int(time.time()*1000)}.jpg"
    image_path = os.path.join(folderpath,subpath,filename)
    #cv2.imwrite("~/git_autoclean/component_pose_estimation/scripts/Experiment_image_capture/data_images/"+filename,color_image)
    cv2.imwrite(image_path,color_image)
    print(f"image captured at {int(time.time()*1000)}")
    time.sleep(0.5)

def display_camera():
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense Color Image', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        #pipeline.stop()
        cv2.destroyAllWindows()

def capture_window_image(max_count):
    #get number of files in the subpath:
    files = glob.glob(os.path.join(folderpath,subpath, f"*.jpg"))
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


if __name__ == "__main__":
    if run_camera():
        #display_camera() 
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        capture_image(timestamp)
        capture_window_image(100)
        
    else:
        pass
