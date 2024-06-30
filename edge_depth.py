import json
import glob
import os
import numpy as np
import copy
import time
import itertools
import open3d as o3d
from shapely.geometry import Polygon, Point
from tools.processes import load_file, planedetect, pcd_crop, save_instance_to_json, load_camera_instance_from_json, create_coordinates_frame
from tools.computations import calculate_midpoints, convert_to_pixel
from check_calibration import CameraApp

# Global variables
debug_mode = True
display_mode = True
display = debug_mode and display_mode
z_state = True
d_frame = True
gripper_clearance = 0.025 

class GraspableEdge:
    def __init__(self, i, region_extents, pcd_data):
        self.edge_region_extents = region_extents
        self.lower_ext, self.upper_ext = self.edge_region_extents
        self.bounding_limits =  [
                                [self.lower_ext[0],self.upper_ext[0]], # left right -x +x 
                                [self.lower_ext[1],self.upper_ext[1]], # down up -y +y 
                                [self.lower_ext[2],self.upper_ext[2]] # z near far -(-z +z)
                            ]
        self.pcd_data = pcd_data
        self.edge_threshold = 0.005/1.5 # Threshold for edge plane detection (larger = wider edges)
        self.edge_id = i

        # initialized in self.generate_bounding_volume()
        self.edge_validity = False # Check non-validity: edge contains 0 points
        self.edge_plane_points = None
        self.edge_plane_normal = None    
        self.edge_center = None
        self.edge_plane_model = None
        self.edge_width = None
        self.edge_depth = None
        self.edge_length = None 
        self.bbox_midpoints = None
        self.bbox_rotation = None

        self.edge_centroid = np.array([(self.lower_ext[0]+self.upper_ext[0])/2., 
                                       (self.lower_ext[1]+self.upper_ext[1])/2., 
                                       0.0])
        self.edge_grasp_point = None
        self.edge_grasp_frame = None
        self.edge_grasp_coord_frame = None
        self.edge_grasp_accuracy = 0.
        
        self.generate_bounding_volume()
        self.obtain_grasp_point() # copied from original sim edge_depth.py
      
    def generate_bounding_volume(self):
        bounding_limits_points = list(itertools.product(*self.bounding_limits))
        bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounding_limits_points))
        edge_region_cropped = self.pcd_data.pcd_objects.crop(bounding_box) 

        # Filter through another plane within oriented bounding box (can also be replaced with box params):
        edge_plane = planedetect(edge_region_cropped, self.edge_threshold)        
        self.edge_plane_points = edge_plane[0]
        if len(np.asarray(self.edge_plane_points.points))!=0:
            self.edge_validity = True
            self.edge_plane_model = edge_plane[-1]
            [a,b,c,_] = self.edge_plane_model
            inliers = edge_plane[3]
            inliers_points = np.asarray(self.pcd_data.pcd_objects.points)[inliers]

            polygon = o3d.geometry.PointCloud()
            polygon.points = o3d.utility.Vector3dVector(inliers_points)
            self.bbox = polygon.get_oriented_bounding_box()
                          
            self.bbox_midpoints = calculate_midpoints(self.bbox)
            self.edge_plane_normal = np.array([a,b,c])
            plane_center = np.mean(inliers_points, axis=0) # center of point clouds
            center = self.bbox.get_center() # center of oriented bbox
            length = self.bbox.extent[0]
            width = self.bbox.extent[1]
            depth = self.bbox.extent[2]
            self.bbox_rotation = self.bbox.R

            self.edge_center = center
            self.edge_width = width
            self.edge_depth = depth
            self.edge_length = length

        else:
            self.edge_validity = False

    def obtain_grasp_point(self):
        # Region extents as a Shapely Polygon/Rectangle
        min_coords = np.minimum(np.array(self.lower_ext)[:2],np.array(self.upper_ext)[:2])
        max_coords = np.maximum(np.array(self.lower_ext)[:2],np.array(self.upper_ext)[:2])
        corners = [min_coords,
                            [min_coords[0], max_coords[1]],
                            [max_coords[0], min_coords[1]],
                            max_coords]
        rectangle = Polygon(corners)
        def nearest_neighbour_point(query_point):
            pcd_tree = o3d.geometry.KDTreeFlann(self.pcd_data.pcd_objects)
            [k, idx, _] = pcd_tree.search_knn_vector_3d(query_point, 1)
            return np.asarray(self.pcd_data.pcd_objects.points)[idx[0]]  
        def compute_grasp_accuracy(grasp_points):
            dmax=max([np.linalg.norm(grasp_points[0][:2] - np.array(corner)) for corner in corners])
            dist=np.linalg.norm(abs(grasp_points[0] - grasp_points[1]))
            self.edge_grasp_accuracy = abs(((dmax-dist)/dmax))
        def calculate_orientation():
                # Calculate the vector from center to target
                direction = self.edge_grasp_point - np.array(self.edge_center)
                direction /= np.linalg.norm(direction)

                # Calculate the rotation matrix
                z_axis = np.array([0, 0, 1])
                rotation_matrix = np.eye(3)
                rotation_matrix[:, 2] = direction
                rotation_matrix[:, 0] = np.cross(z_axis, direction)
                rotation_matrix[:, 0] /= np.linalg.norm(rotation_matrix[:, 0])
                rotation_matrix[:, 1] = np.cross(rotation_matrix[:, 2], rotation_matrix[:, 0])
                self.edge_grasp_frame = [self.edge_grasp_point, rotation_matrix]

        # Use centroid value to perform nearest neighbour search, Check if within rectangle:
        centroid_grasp_point = nearest_neighbour_point(self.edge_centroid)
        centroid_grasp_point_valid = rectangle.contains(Point(centroid_grasp_point))
        # To compute accuracy using height from centroid_grasp_point
        edge_centroid_with_z = self.edge_centroid.copy()  
        edge_centroid_with_z[2] = centroid_grasp_point[2] 

        if (self.edge_depth < gripper_clearance) and (self.edge_width > gripper_clearance):
            # Solution I:              
            
            # Find midpoint closest to the centroid_grasp_point, Check if within rectangle:
            ind_nearest_midpoint = np.argmin([np.linalg.norm(centroid_grasp_point - point) for point in self.bbox_midpoints])
            nearest_midpoint = self.bbox_midpoints[ind_nearest_midpoint]
            nearest_midpoint[-1] = 0.
            midpoint_as_grasp_point = nearest_neighbour_point(np.array(nearest_midpoint))
            midpoint_grasp_point_valid = rectangle.contains(Point(midpoint_as_grasp_point))
            """
            Check distance of (xm,ym) to (xc,yc)
            If within ROI extents, accuracy>0, else 0; obtain overall accuracy of point:
            if both within, select one closest to centroid, else choose one within rectangle
            Accuracy: ((dmax-d)/dmax)*100
            """
            if centroid_grasp_point_valid and midpoint_grasp_point_valid:
                grasp_points = [centroid_grasp_point, midpoint_as_grasp_point]
                # Get distance to centroid
                ind_nearest_grasp_point = np.argmin([np.linalg.norm(self.edge_centroid - point) for point in grasp_points])
                self.edge_grasp_point = self.bbox_midpoints[ind_nearest_grasp_point]
            elif centroid_grasp_point_valid:
                self.edge_grasp_point = centroid_grasp_point
                grasp_points = [edge_centroid_with_z, self.edge_grasp_point]
            elif midpoint_grasp_point_valid:
                self.edge_grasp_point = midpoint_as_grasp_point
                grasp_points = [edge_centroid_with_z, self.edge_grasp_point]
            else:
                grasp_points = [edge_centroid_with_z, centroid_grasp_point]
                self.edge_grasp_point = edge_centroid_with_z

            # Align grasp to center of plane/edge bbox
            calculate_orientation()
            
        else:
            # Solution II:
            self.edge_grasp_point = centroid_grasp_point
            self.edge_grasp_frame = [self.edge_grasp_point, self.bbox_rotation]
            grasp_points = [edge_centroid_with_z, self.edge_grasp_point]
            
        compute_grasp_accuracy(grasp_points)

    def show(self):
        # To visualize pcd data with open3d
        geometries = []
        geometries.append(self.pcd_data.pcd_objects)
        geometries.append(create_coordinates_frame(center=self.edge_grasp_frame[0],rotation=self.edge_grasp_frame[1]))
        obb_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(self.bbox)
        obb_ls.paint_uniform_color([1, 0, 0])
        geometries.append(obb_ls)
        o3d.visualization.draw_geometries(geometries)
          
class Detection:
    def __init__(self, camera, pcd_data):
        self.timestamp = int(time.time() * 1000)
        self.pcd_data = pcd_data
        self.nominal_depth = None
        self.num_of_edges = None
        self.edges = None
        self.edges_pixels = None
        # TO DO: Load camera parameters from json with threshold time
        self.nominal_depth = camera.nominal_depth
        self.fx, self.fy, self.cx, self.cy = camera.fx, camera.fy, camera.cx, camera.cy
        self.graspable_edges = []
        self.initialize()
        
    def initialize(self):
        def pixel_to_coord(x,y):
            X = (x - self.cx) * self.nominal_depth / self.fx
            Y = (y - self.cy) * self.nominal_depth / self.fy
            return X,Y
        self.set_edge_regions_auto()
        self.filter_edge_regions()
        self.edges_pixels = convert_to_pixel(self.edges)
        for i,edge_pixel in enumerate(self.edges_pixels):
            w,h = edge_pixel['relative_coordinates']['width'], edge_pixel['relative_coordinates']['height']
            x1,y1 = edge_pixel['actual_coordinates']['x1'],edge_pixel['actual_coordinates']['y1']
            X1,Y1 = pixel_to_coord(x1,y1)
            X2,Y2 = pixel_to_coord(x1+w,y1+h)
            min_pos = [X1,Y1,self.nominal_depth]
            max_pos = [X2,Y2,self.nominal_depth/2]
            edge_region_extents = [min_pos,max_pos]
            # an instance is added to the self.graspable_edges list
            graspable_edge = GraspableEdge(i, edge_region_extents, self.pcd_data)
            if debug_mode:
                print(f"For edge: {i} \n --------------- \n ")
                attributes = vars(graspable_edge)
                for attr, val in attributes.items():
                    print(f">>> {attr} -- {val} \n")
                print(f"         \n --------------- \n ")
            self.graspable_edges.append(graspable_edge)

    def set_edge_regions_auto(self):
        self.edge_regions = load_file('/roi/*.json')
        for key,value in self.edge_regions.items():
            setattr(self, key, value)

    def filter_edge_regions(self):
        # number of objects detected:
        all_detected = self.objects
        self.num_of_edges = len(all_detected)
        if self.num_of_edges != 0:
            to_identify = 'edge'
            all_edges = [i for i in all_detected if i['name'] == to_identify] # 'edge' or 'redcube'
            if not all_edges:
                print(f"No {to_identify} objects detected.")
            # set a threshold for the width and height (instead of confidence)
            # both width and height ratios greater than 0.2 should be removed
            thresh_width = 0.2
            thresh_height = 0.2
            filtered_objects = [obj for obj in all_edges if not (obj["relative_coordinates"]["width"] > thresh_width and obj["relative_coordinates"]["height"] > thresh_height)]
            if debug_mode:
                print(f"{time.time() * 1000} : \n {filtered_objects}")
            self.edges = filtered_objects

class PcdData:
    def __init__(self):
        self.geometries = []
        self.pcd = None
        self.pcd_objects = None
        self.initialize()

    def initialize(self):
        self.pcd = load_file('/pcd/*.pcd')
        _, self.pcd_objects, _, _, _ = planedetect(self.pcd)
        self.pcd_objects = pcd_crop(self.pcd_objects)
        self.add_geometry(self.pcd_objects)

    def add_geometry(self, geometry):
        self.geometries.append(geometry)

if __name__ == "__main__":
    """
    1) Create a detection module
    2) The detection module reads the "detection*.json" file
    3) It then creates an instance of each edge detected with the parameters:
            a) position (center)
            b) width, height
            c) confidence
    4) For each instance, move the robot over the edge (Use "Z" value from the background (known and determinable)):
        a) capture an image and PCD.
        b) apply edge depth again with the same "Z" value.
        c) since camera is right above (Or almost) the edge, the error is reduced when calculating the 6D edge.
    5) For each instance then, 6D estimation and grasping is achieved.
    """
    pcd = PcdData()
    
    # Load camera parameters needed for parts-per-metric calculations:
    camera =  load_camera_instance_from_json()
    if not camera:
        camera = CameraApp()
        camera.initialize_camera() # On empty drop-off region to get background height and camera intrinsics
        save_instance_to_json(camera)
    detection = Detection(camera, pcd)
    for graspable_edge in detection.graspable_edges:
        if debug_mode:
            print(graspable_edge.edge_grasp_frame)
            graspable_edge.show()
    