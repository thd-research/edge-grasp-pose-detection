import json
import glob
import os
import cv2
import numpy as np
import copy
import time
import itertools
import open3d as o3d
from shapely.geometry import Polygon, Point
import pyrealsense2 as rs


tower_pc = False
debug_mode = True
display_mode = True
display = debug_mode and display_mode
feedback = debug_mode
z_state = True
d_frame = True
gripper_clearance = 0.025 

def load_file(file_type):
    folderpath = os.path.dirname(os.path.abspath(__file__))
    file_type = str(file_type)
    files = glob.glob(folderpath+file_type)
    latest_file = max(files, key=os.path.getctime)
    if feedback:
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

def obtain_edge_regions():
    detected_objects = load_file('/*.json')
    # objects are saved in a list, each being a dictionary of its own
    # number of objects detected: 
    all_detected = detected_objects[0]['objects']
    num_detected = len(all_detected)

    # check for zero detections
    if num_detected != 0:
        to_identify = 'edge'
        all_edges = [i for i in all_detected if i['name'] == to_identify] # 'edge' or 'redcube'
        if not all_edges:
            print(f"No {to_identify} objects detected.")

        # set a threshold for the width and height (instead of confidence)
        # both width and height ratios greater than 0.2 should be removed
        thresh_width = 0.2
        thresh_height = 0.2
        filtered_objects = [obj for obj in all_edges if not (obj["relative_coordinates"]["width"] > thresh_width and obj["relative_coordinates"]["height"] > thresh_height)]
        if feedback:
            print(filtered_objects)

    return filtered_objects

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

def get_width(box_coords):
    # Create a Shapely Polygon object from the coordinates
    polygon = Polygon(box_coords)
    # Get the exterior ring of the polygon
    exterior_ring = polygon.exterior
    # Calculate the lengths of the sides of the exterior ring
    #lengths = [Point(exterior_ring.coords[i]).distance(Point(exterior_ring.coords[(i + 1) % len(exterior_ring.coords)])) for i in range(len(exterior_ring.coords))]
    lengths = [Point(exterior_ring.coords[i]).distance(Point(exterior_ring.coords[i + 1])) for i in range(len(exterior_ring.coords) - 1)]
    # Determine the maximum and minimum lengths as the length and width of the rectangle
    length = max(lengths)
    width = min(lengths)

    return width

def canny_edge_method(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 125, 235) # 195, 235
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, edges

def pcd_info(point_cloud):
    # Get the point coordinates
    points = np.asarray(point_cloud.points)

    # Compute the minimum and maximum coordinates along each axis
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])

    print("Minimum (x, y, z) point:", (min_x, min_y, min_z))
    print("Maximum (x, y, z) point:", (max_x, max_y, max_z))
    return min_x, max_x, min_y, max_y, min_z, max_z #0,1, 2,3, 4,5

def pcd_crop(pcd, lims =   [[-0.25,0.15], # left right -x +x
                            [-0.16,0.12], # down up -y +y
                            [0.2,0.65]] ):
    
    crop_limits_points = list(itertools.product(*lims))
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(crop_limits_points))
    cropped_pcd = pcd.crop(bounding_box)

    return cropped_pcd

def create_locator(center):
    locator = o3d.geometry.TriangleMesh.create_box(width=0.005, height=0.005, depth=0.005)
    locator.paint_uniform_color([0.6, 0.0, 1.0]) 
    locator.translate(center)
    return locator

def create_coordinates_frame(center, rotation=np.eye(3)):
    # Create a coordinate frame at the center with orientation from the rotation matrix
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.025)
    coordinate_frame.translate(center)
    coordinate_frame.rotate(rotation, center=center)

    return coordinate_frame

def run_camera_high_density(pipeline=None,start=True):
    if not start:
        pipeline.stop()
    else:
        pipeline = rs.pipeline()
        cfg = rs.config()  
        cfg.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
        # apply high density PCD
        config = pipeline.start(cfg)
        dev = config.get_device()
        # Toggle between these two maybe:   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
        depth_sensor = dev.first_depth_sensor()
        preset = 4
        depth_sensor.set_option(rs.option.visual_preset,preset)
        depth_scale = depth_sensor.get_depth_scale()
        
        return pipeline

def pixels_to_m(img_coords):
        pipeline = rs.pipeline()
        cfg = rs.config()  
        cfg.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
        # apply high density PCD
        config = pipeline.start(cfg)
        dev = config.get_device()
        # Toggle between these two maybe:   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
            depth_scale = depth_sensor.get_depth_scale()
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy

            print(f"X,Y,Z: {X,Y,Z}")
            pipeline.stop()
            return X,Y,Z

def get_position_in_m(centre_point, z_state=False, d_frame=False, display_mode=False):
    
    x,y = centre_point
    x,y = int(x), int(y)
    X,Y,Z = pixels_to_m([x,y])

    y-=45
    x-=20
    
    if debug_mode:
        print(f"Registered Grasp Coordinates: {X,Y}")
    if not z_state and not d_frame:
        return [X,Y]
    elif z_state and not d_frame:
        return [X,Y,Z]
    elif z_state and d_frame:
        # Instead of returning, show directly on feed:
        #return [X,Y,z], depth_frame
        if display_mode:
            pipeline = run_camera_high_density()
            while True:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                #crosshair_center = (x,y)
                cv2.line(depth_colormap, (x - 10, y), (x + 10, y), (0, 0, 0), 1)
                cv2.line(depth_colormap, (x, y - 10), (x, y + 10), (0, 0, 0), 1)

                # Add text next to the crosshair
                text_position = (x + 20, y + 20)
                cv2.putText(depth_colormap, 'X: {:.3f}, Y: {:.3f}, Z: {:.2f} m'.format(X,Y,Z),
                            text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.imshow('Depth Frame', depth_colormap)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    run_camera_high_density(pipeline,False)
                    cv2.destroyAllWindows()
                    break

    return [X,Y,Z]

def get_position_in_m_update(edge):

    rc = edge['relative_coordinates']
    ac = edge['actual_coordinates']
    x,y,w,h = ac['x1'], ac['y1'], rc['width'], rc['height']
    
    x1,y1 = x,y
    x2,y2 = x+w, y+h

    X1,Y1,Z1 = pixels_to_m([x1,y1])
    X2,Y2,Z2 = pixels_to_m([x2,y2])
    
    if debug_mode:
        print(f"Registered Grasp Coordinates in region: : {X1,Y1}")

    if z_state and d_frame:
        # Instead of returning, show directly on feed:
        #return [X,Y,z], depth_frame
        if display_mode:
            pipeline = run_camera_high_density()
            while True:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                #crosshair_center = (x,y)
                cv2.line(depth_colormap, (x - 10, y), (x + 10, y), (0, 0, 0), 1)
                cv2.line(depth_colormap, (x, y - 10), (x, y + 10), (0, 0, 0), 1)

                # Add text next to the crosshair
                text_position = (x + 20, y + 20)
                cv2.putText(depth_colormap, 'X: {:.3f}, Y: {:.3f}, Z: {:.2f} m'.format(X1,Y1,Z1),
                            text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.imshow('Depth Frame', depth_colormap)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    run_camera_high_density(pipeline,False)
                    cv2.destroyAllWindows()
                    break
        
        min_pos = [X1, Y1, 0.35]
        max_pos = [X2, Y2, 0.55]
        pos = [min_pos, max_pos]
        
        return pos

    """
    arr_in_m_update = np.empty((0,5))
    arr_in_pixels = np.empty((0,5))
    ppm_x = 1618 # obtained from pre-calibrated data
    ppm_y = 1600
    ppm = [ppm_x,ppm_y,ppm_x,ppm_y,1]
    for i,cube in enumerate(arr):
        #cube[0]+=img_centre[0]
        cube[0]-=img_centre[0]
        cube[1]-=img_centre[1]
        #cube[1] = -cube[1]
        arr_in_pixels = np.vstack((arr_in_pixels,cube))
        cube[1] = -cube[1]
        cube = [round(cube_val/ppm_val,5) for cube_val,ppm_val in zip(cube,ppm)]
        print("cube value: ",cube)
        arr_in_m_update = np.vstack((arr_in_m_update,cube))

    return arr_in_pixels,arr_in_m_update
    """

def calculate_line_lengths(line_set):
    lines = np.asarray(line_set.lines)
    points = np.asarray(line_set.points)

    line_lengths = []
    for line in lines:
        start_point = points[line[0]]
        end_point = points[line[1]]
        length = np.linalg.norm(start_point - end_point)
        line_lengths.append(length)

    # Compute the median length of lines
    median_length = np.median(line_lengths)

    # Filter out lines longer than the median length
    filtered_lines = []
    filtered_lengths = []
    for line, length in zip(lines, line_lengths):
        if length <= median_length:
            filtered_lines.append(line)
            filtered_lengths.append(length)

    return filtered_lines

def generate_grasp_candidate_2(plane_model, pcd, inliers):
    # Toggle Rodrigues:
    default = False
    rodrigues = False
    approximation = True
    
    # Get Normal vector from plane model:
    [a,b,c,d] = plane_model
    normal = np.array([a,b,c])
    # Bounding Box parameters:
    # Centroid:
    inliers_points = np.asarray(pcd.points)[inliers]
    center = np.mean(inliers_points, axis=0)
    #centroid_plane = create_locator(center)
    width, height, depth = 0.05,0.015,0.005
    
    if default:
        rotation = np.eye(3)
        rotation[0,:]= normal
    elif rodrigues:
        axis = np.cross(normal, [0, 0, 1])  # Find the axis of rotation
        axis /= np.linalg.norm(axis)  # Normalize the axis
        angle = np.arccos(np.dot(normal, [0, 0, 1]))  # Find the angle of rotation
        rotation, _ = cv2.Rodrigues(axis * angle)  # Get the rotation matrix using Rodrigues' formula
    elif approximation:
        polygon = o3d.geometry.PointCloud()
        polygon.points = o3d.utility.Vector3dVector(inliers_points)
        bbox = polygon.get_oriented_bounding_box()


        # Get dimensions of the bounding box for area:
        # Get dimensions of the bounding box
        extent = bbox.extent
        length = extent[0]
        width = extent[1]
        area = length*width

        center = bbox.get_center()
        
        rotation = bbox.R
        #print(f"Center of bbox is : {center} - rotation is {rotation}")
        coord_frame = create_coordinates_frame(center, rotation)
        plane_center_params = [center, rotation, area]
    #bbox = o3d.geometry.OrientedBoundingBox(center, rotation, [width, height, depth])

    return bbox, coord_frame, plane_center_params

def generate_line_set(cropped_pcd, num_points=100):
    threshold = 0.005 # 0.005
    # Detect a plane in each of the edge regions:
    pcd_broken = planedetect(cropped_pcd, threshold) 
    inlier_pcd = pcd_broken[0]
    inlier_pcd.paint_uniform_color([0.6,0.6,0.6])

    # Toggle between complete crop and plane:
    plane = True

    if plane:
        # Obtain highest points on the plane:
        points = np.asarray(inlier_pcd.points)
    else:
        # Obtain highest points in the region:
        points = np.asarray(cropped_pcd.points)
    if len(points)!=0:
        highest_indices = np.argsort(points[:, 2])[:num_points]
        highest_points = points[highest_indices]
        #return highest_points
        
        # Create a lineset connecting the points:
        lines = []
        for i in range(len(highest_indices) - 1):
            lines.append([i, i+1])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(highest_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        calculate_line_lengths(line_set)

        # For each line_set, calculate the vector normal:
        grasp_candidates = []
        for i,line in enumerate(lines):
            start_index, end_index = line
            start_point = highest_points[start_index]
            end_point = highest_points[end_index]

            # middle point:
            middle_point = (start_point + end_point)/2
            # direction vector:
            direction_vector = end_point - start_point
            # generate grasp pose for each line in the line_set:
            #grasp_candidate = generate_grasp_candidate(middle_point, direction_vector)
            #grasp_candidates.append(grasp_candidate)
            # Obtain orientation of the grasp candidates

            # pcd_broken: [0] inlier_cloud, [1] outlier_cloud, [2] z, [3] inliers, [4] plane_model

            plane_model = pcd_broken[-1]
            inliers = pcd_broken[3]
            grasp_candidate_orientation = generate_grasp_candidate_2(plane_model, cropped_pcd, inliers)
            plane_center_params = grasp_candidate_orientation[2]
            coord_frame= grasp_candidate_orientation[3]

            # grasp_candidate_orientation: [0] bbox, [1] coord_frame, [2] plane_center_params
            # plane_center_params: [0] center, [1] rotation (wrt) 0,0,0, [2] area

            plane_params_single = [plane_center_params, plane_model]

            grasp_candidates.append(grasp_candidate_orientation[0])
            grasp_candidates.append(grasp_candidate_orientation[1])

        return line_set, grasp_candidates, inlier_pcd, plane_params_single, coord_frame

    else:
        return None
      
# TRANSFERRED TO NEW SCRIPT
def oversegment(img, edge, num_edge, attempt, prev_x, prev_y, canny=False):
    rc = edge['relative_coordinates']
    ac = edge['actual_coordinates']
    x,y,w,h = ac['x1'], ac['y1'], rc['width'], rc['height']
    
    x_o, y_o = x,y
    #x_o += prev_x
    #y_o += prev_y

    if debug_mode:
        print(f"Current values: {edge}; xo,yo: {x_o,y_o}, prev_x,prev_y; {prev_x,prev_y}")

    #scaling for the width
    w *= 1
    h *= 1
    w,h = int(w),int(h)

    x1,y1 = x, y
    x2,y2 = x+w, y+h
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

    cropped_img = img[y1:y2, x1:x2]
    
    if not canny:
        # for thresholding
        hue_min,hue_max = 49,51 #0,67
        saturation_min,saturation_max = 62,93 # 37 # 7,73
        value_min,value_max = 137,147 # 209,219 # 135,255 
        hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        lower_color = np.array([hue_min, saturation_min, value_min])
        upper_color = np.array([hue_max, saturation_max, value_max])
        mask = cv2.inRange(hsv_img, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, mask = canny_edge_method(cropped_img)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    window_name = 'cropped roi attempt '+str(attempt)

    # obtain the largest contour:
    rect = cv2.minAreaRect(contours_sorted[0])
    axis_aligned_rect = cv2.boundingRect(contours_sorted[0])
    if debug_mode:
        print(f"rotated rect: {rect}")
        print(f"axis aligned rect: {axis_aligned_rect}")

    rect_list =  [list(item) if isinstance(item, tuple) else item for item in rect]
    rotation_about_z = rect_list[2]
    boxf = cv2.boxPoints(rect)
    box = np.intp(boxf) # useful to crop pcd
    if debug_mode:
        print(f"boxf: {boxf}")
    for row in boxf:
        row[0] += x_o
        row[1] += y_o
    
    if debug_mode:
        print(f"boxf updated after xo,yo: {boxf}")
   
        if display:
            cv2.circle(img, (int(row[0]),int(row[1])), 3, (0,255,255), -1)

    centroidf = np.mean(boxf, axis=0)
    
    

    centroidf_in_m_test = get_position_in_m(centroidf, z_state=True, d_frame=True, display_mode=True)
    centroidf_in_m = get_position_in_m(centroidf, False, False)
    
    if all(x == 0.0 and x is not True for x in centroidf_in_m):
        if debug_mode:
            print("Getting Z did not work the first time, trying a second and last time...")
        time.sleep(0.5)
        centroidf_in_m = get_position_in_m(centroidf, False) # try again one more time after a small pause
    
    grasp_pose = [centroidf_in_m, rotation_about_z]
    if feedback:
        print(f"centroid_in_m: {centroidf_in_m} - rotation_about_z: {rotation_about_z}")

    boxf_in_m = np.array([get_position_in_m(row) for row in boxf])  
    if debug_mode:
        print(f"boxf_in_m : {boxf_in_m}")

    """
    >>>>>> RETURN THE CENTROID AND THE ORIENTATION MEASURED FROM BOXF_IN_M INSTEAD OF THE BOXF_IN_M ITSELF
    >>>>>> ADD TO THE GEOMTETRIES THAT POSITION WITH THE ORIENTATION TO CHECK CORRECTNESS
    """
    if display:
        
        cv2.circle(img, (int(centroidf[0]),int(centroidf[1])), 3, (0,255,255), -1)
        cv2.drawContours(cropped_img,[box],0,(0,255,255),2)  # yellow oriented boxes
        cv2.imshow('oversegment', img)
        cv2.imshow(window_name, cropped_img)
        cv2.imshow('thresholding',mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        pass

    return boxf_in_m, rotation_about_z, [x_o, y_o], axis_aligned_rect, centroidf

def adjust_coords(edge,prev_coords, axis_aligned_rect):
    # returns a dictionary similar to "edge" to be parsed to "oversegment"
    if debug_mode:
        print(f"edge received: {edge}")
        print(f"aarect: {axis_aligned_rect}")
    
    x,y,w,h = axis_aligned_rect
    edge['relative_coordinates']['width'] = w
    edge['relative_coordinates']['height'] = h
    edge['actual_coordinates']['x1'] = prev_coords[0]+x
    edge['actual_coordinates']['y1'] = prev_coords[1]+y

    if debug_mode:
        print(f"Processed edge: {edge}")

    return edge

# TRANSFERRED TO NEW SCRIPT
def obtain_edge_depth():
    detected_objects = load_file('/*.json')
    # objects are saved in a list, each being a dictionary of its own
    # number of objects detected: 
    all_detected = detected_objects[0]['objects']
    num_detected = len(all_detected)

    # check for zero detections
    if num_detected != 0:
        to_identify = 'edge'
        all_edges = [i for i in all_detected if i['name'] == to_identify] # 'edge' or 'redcube'
        if not all_edges:
            print(f"No {to_identify} objects detected.")
        # set a threshold for the width and height (instead of confidence)
        # both width and height ratios greater than 0.2 should be removed
        thresh_width = 0.2
        thresh_height = 0.2
        filtered_edges = [obj for obj in all_edges if not (obj["relative_coordinates"]["width"] > thresh_width and obj["relative_coordinates"]["height"] > thresh_height)]
        if feedback:
            print(filtered_edges)

        # Convert from ratios to pixels based on image size:
        #edges = convert_to_pixel(all_edges)
        edges = convert_to_pixel(filtered_edges)
        img = load_file('/images/*.jpg')
        grasp_regions = []
        centroidf_values = []
        attempts = 1
        for i,edge in enumerate(edges):
            attempt = 0
            prev_x, prev_y = 0,0   
            grasp_region, rot, prev_coords, aarect,centroidf = oversegment(img, edge, i, attempt, prev_x, prev_y, True)
            prev_x,prev_y = prev_coords
            edge = adjust_coords(edge,prev_coords,aarect)
            grasp_width = get_width(grasp_region)
            if debug_mode:
                print(f"Average grasp width: {grasp_width}")
            attempt = 1
            if grasp_width > 0.01:
                while attempt < attempts:
                    prev_x,prev_y = prev_coords
                    if debug_mode:
                        if attempt == 1:
                            print(f"Entering attempts loop.")
                        print(f"Attempt: {attempt}")
                    prev_x,prev_y = prev_coords
                    grasp_region, rot, prev_coords, _ , centroidf = oversegment(img, edge,i, attempt, prev_x, prev_y, True)
                    grasp_width = get_width(grasp_region)
                    if debug_mode:
                        print(f"Results of attempt {attempt}: grasp_region: {grasp_region}, prev_coords: {prev_coords}")
                    if grasp_width < 0.01:
                        break
                    else:
                        pass
                    attempt+=1
                if debug_mode:
                    print(f"Completed attempts at attempt: {attempt}")
            else:
                pass
            
            grasp_regions.append(grasp_region)
            centroidf_values.append(centroidf)
            print(f"centroidf_values: {centroidf_values}")
            grasp_poses = []
            for i,centroidf in enumerate(centroidf_values):
                grasp_pose = {}
                print(f"centroidf: {centroidf}")
                centroidf_adjusted = copy.deepcopy(centroidf)
                centroidf_adjusted[0] += 20
                centroidf_adjusted[1] += 45
                print(f"Comparison: centroidf_adjusted: {centroidf_adjusted} -- centroidf: {centroidf}")
                centroid_depth_accurate = get_position_in_m(centroidf_adjusted,z_state=False,d_frame=False,display_mode=False)
                centroidf_depth = get_position_in_m(centroidf, z_state=True, d_frame=True)
                print(f"Comparison: centroidf_depth: {centroidf_depth} -- centroidf_depth_accurate: {centroid_depth_accurate}")
                #centroidf_list = centroidf.tolist()
                centroid_depth_accurate.append(centroidf_depth[2])

                #grasp_poses.append(centroid_depth_accurate)
                grasp_pose['edge_id'] = i
                grasp_pose['centroid_accurate'] = centroid_depth_accurate
                grasp_poses.append(grasp_pose)

                if debug_mode:
                    print(f"grasp: {grasp_poses}")
                    print(f"centroid accurate: {centroid_depth_accurate}")

        # return the accurate edge grasp   
        return grasp_poses

    else:
        print("No edges detected.")
        return None

def process_pcd_to_extract_parts():

    # detect in whole PCD or cropped region:
    whole = False

    geometries = []
    plane_params = []
    pcd = load_file('/pcd/*.pcd')
    pcd_broken = planedetect(pcd) # pcd_broken: inlier (plane), outlier, z
    inlier_pcd, outlier_pcd, z, inliers, plane_model = pcd_broken

    geometries.append(pcd_crop(outlier_pcd))

    # Obtain the edge regions from the CNN:
    edge_regions = obtain_edge_regions()
    edge_regions_pixels = convert_to_pixel(edge_regions)
    edge_regions_in_m = []
    pos = edge_regions_pixels

    for i,p in enumerate(pos):  
        plane_param = {}
        #min_pos,max_pos = apply_ppm_update(edge)
        
        min_pos, max_pos = get_position_in_m_update(p)
        
        #geometries.append(create_locator(min_pos))
        #geometries.append(create_locator(max_pos))

        crop_limits = [
            [min_pos[0],max_pos[0]], # left right -x +x # -0.25,0.25
            [min_pos[1],max_pos[1]], # down up -y +y # -0.16,0.16
            [min_pos[2],max_pos[2]] # z near far -(-z +z)
        ]
        
        crop_limits_points = list(itertools.product(*crop_limits))
        bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(crop_limits_points))
        if whole:
            edge_region_cropped = pcd.crop(bounding_box)
        else:
            edge_region_cropped = outlier_pcd.crop(bounding_box)
        #edge_region_cropped.paint_uniform_color([0.6,0.6,0.6])
        #geometries.append(edge_region_cropped)
        
        line_set_results = generate_line_set(edge_region_cropped)

        geometries.append(line_set_results[0])
        geometries.append(line_set_results[2])
        #plane_params.append(line_set_results[3]) # Use a dictionary instead to keep track of the parts
        plane_param['edge_id'] = i
        plane_param['plane_center'] = line_set_results[3][0]
        plane_param['plane_model'] = line_set_results[3][1]
        plane_param['coord_frame'] = line_set_results[4]

        plane_params.append(plane_param)
        for _ in line_set_results[1]:
            geometries.append(_)
        #for edge_grasp_point in generate_line_set(edge_region_cropped):
            #print(f"edge grasp point: {edge_grasp_point}")
            #geometries.append(create_locator(edge_grasp_point))
            
    if display:
        #o3d.visualization.draw_geometries(geometries)
        pass


    return geometries, plane_params

def nearest_highest_point(pcd, chosen_point):
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(chosen_point, 100)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
    nearest_points_sorted = [pcd.points[i] for i in idx]
    nearest_points_sorted.sort(key=lambda point: point[2]) 
    nearest_point = nearest_points_sorted[0]

    return nearest_point

def verify_same_plane(query, center, pcd):
    # Get the index of the second point in the point cloud
    distances = np.linalg.norm(pcd.points - center, axis=1)
    index_center = np.argmin(distances)
    if debug_mode:
        print(f"index_center : {index_center} - {pcd.points[index_center]}")
        print(f"actual center: {center}")
    #index_center = np.where((pcd.points == center).all(axis=1))[0][0]
    index_point3 = (index_center + 5) % len(pcd.points) 
    point3 = np.asarray(pcd.points[index_point3]) # next 10th point arbitrarily chosen on the plane

    # Compute the plane defined by the points
    plane_normal = np.cross(center - query, point3 - query)
    plane_normal /= np.linalg.norm(plane_normal)

    query_dist = np.abs(np.dot(plane_normal, center - query))
    threshold_distance = 0.0075

    return query_dist <= threshold_distance

def align_grasp_axis(nearest_point, plane_center, plane_coeffs=None):
    def calculate_orientation(center, target):
        # Calculate the vector from center to target
        direction = np.array(target) - np.array(center)
        direction /= np.linalg.norm(direction)

        # Calculate the rotation matrix
        z_axis = np.array([0, 0, 1])
        rotation_matrix = np.eye(3)
        rotation_matrix[:, 2] = direction
        rotation_matrix[:, 0] = np.cross(z_axis, direction)
        rotation_matrix[:, 0] /= np.linalg.norm(rotation_matrix[:, 0])
        rotation_matrix[:, 1] = np.cross(rotation_matrix[:, 2], rotation_matrix[:, 0])
        return rotation_matrix
    
    # Additional rotation to align z-axis with plane center
    if plane_coeffs is not None:
        rotation_towards_plane_center = calculate_orientation(nearest_point, plane_center)
    else:
        rotation_towards_plane_center = calculate_orientation(nearest_point, plane_center)

    pre_grasp_pose = [nearest_point, rotation_towards_plane_center]

    # return the 6D pre grasp pose
    return pre_grasp_pose

def get_closest_midpoint(query_point, midpoints):
    distances = [np.linalg.norm(np.array(query_point) - np.array(midpoint)) for midpoint in midpoints]
    return midpoints[np.argmin(distances)]

def main():
    
    geometries = []
    geometries_plane,plane_params = process_pcd_to_extract_parts()
    geometries+=geometries_plane
    grasp_poses = obtain_edge_depth() # check if format is similar ; compare with simulate()
    pcd_region = pcd_crop(load_file('/pcd/*.pcd'))
    complete_edge_poses = []

    # Combine the two dictionaries:
    for plane_param in plane_params:
        for edge_pose in grasp_poses:
            if plane_param['edge_id'] == edge_pose['edge_id']:
                complete_edge_pose = plane_param.copy()
                complete_edge_pose.update(edge_pose)
                complete_edge_poses.append(complete_edge_pose)
    print(f"Complete edge poses: {complete_edge_poses}")

    for complete_edge in complete_edge_poses:
        grasp = complete_edge['centroid_accurate']
        geometries.append(create_coordinates_frame(center=grasp))
        closest_point = nearest_highest_point(pcd_region, grasp)
        geometries.append(create_locator(closest_point))
        pre_grasp_pose = align_grasp_axis(closest_point, complete_edge['plane_center'][0].tolist(), plane_coeffs=complete_edge['plane_model'])

    # Area, width, same plane and points underneath estimation:
        query = closest_point
        center = complete_edge['plane_center'][0]
        pcd = pcd_region
        same_plane = verify_same_plane(query, center, pcd)
        edge_width = complete_edge['plane_center'][2]
        edge_depth = complete_edge['plane_center'][3]
        if debug_mode:
            print(f"Check on same plane: {verify_same_plane(query, center, pcd)}")
            print(f"width of edge is: {edge_width}")
            print(f"depth of edge is: {edge_depth}")
        if same_plane:
            if (edge_depth < gripper_clearance) and (edge_width > gripper_clearance):
                geometries.append(create_locator(closest_point))
                #geometries.append(create_coordinates_frame(center=grasp))
                # Create coordinate frame for potential pre_grasp_pose 1
                geometries.append(create_coordinates_frame(center=pre_grasp_pose[0],rotation=pre_grasp_pose[1]))
                midpoints = complete_edge['coord_frame']
                for midpoint in midpoints:
                    geometries.append(create_locator(midpoint))
                closest_midpoint = get_closest_midpoint(closest_point, midpoints)
                geometries.append(create_locator(closest_midpoint))

                selected_grasp_pose = align_grasp_axis(closest_midpoint, complete_edge['plane_center'][0].tolist(), plane_coeffs=complete_edge['plane_model'])
                # Create coordinate frame for true pre_grasp_pose
                geometries.append(create_coordinates_frame(center=selected_grasp_pose[0],rotation=selected_grasp_pose[1]))

                # Centre point for camera coordinates visualization:
                camera_centre_point = [0,0,0]
                camera_centre_rotation = np.eye(3) # for reference

                centre_mesh = create_coordinates_frame(center=camera_centre_point) #, rotation=selected_grasp_pose[1])
                geometries.append(centre_mesh)
                
                if debug_mode:
                    print(f"displacement of true grasp pose: {selected_grasp_pose[0]}")
                    print(f"rotation matrix of true grasp pose: {selected_grasp_pose[1]}")
            else:
                # current coordinate frame represent grasp
                #geometries.append(create_locator(closest_point))
                #geometries.append(create_coordinates_frame(center=grasp))
                #geometries.append(create_coordinates_frame(center=pre_grasp_pose[0],rotation=pre_grasp_pose[1]))
                pass
            

        #geometries.append(create_coordinates_frame(center=pre_grasp_pose[0],rotation=pre_grasp_pose[1]))


    #for i,grasp in enumerate(grasp_poses):
        #geometries.append(create_locator(grasp))

        #geometries.append(create_coordinates_frame(center=grasp))
        #closest_point = nearest_highest_point(pcd_region, grasp)
        #geometries.append(create_locator(closest_point))
        #pre_grasp_pose = align_grasp_axis(closest_point, plane_center, plane_coeffs)
            
    geometries.append(pcd_region)

    if display:
        o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    main()
