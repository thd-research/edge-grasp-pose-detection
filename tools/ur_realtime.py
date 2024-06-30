from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
from rtde_control import RTDEControlInterface
import time, json
from tqdm import tqdm
import os, subprocess

"""
For AutoClean Project, a UR5e Robot is being used, with IP 192.168.100.40.
A camera is mounted on the wrist.
A Schunk Co-Act Gripper is used.

The Gripper outputs are connected to Digital Output 0,1. (HIGH - Closed, LOW - Opened)
Referenced from:
https://sdurobotics.gitlab.io/ur_rtde/api/api.html
https://sdurobotics.gitlab.io/ur_rtde/examples/examples.html

"""
debug_mode = True

class URRealTime:
    def __init__(self):
        self.ip = "192.168.100.40"
        self.ur_poses='poses.json'
        self.ur_file="robotlog.json"
        self.condition=False
        self.connected=False
        self.safety_status=False
        self.ur_motion=False
        self.num_outputs = 20
        self.poses=None
        self.actions=None
        self.last_pose_saved=False
        self.ur_poses_initiliaze()

        self.rtde_r = None
        self.rtde_io = None
        self.rtde_c = None

        self.actual_tcp = [0.,0.,0., 0.,0.,0.]
        self.grasp_target = None

        if self.ur_ping():
            print(f"Robot is operational: {self.ur_ready()}")
        else:
            print("Robot parameters are not initialized.")

    def ur_checklog(self):
        # check if there already exists a file for robotlog (Last pose for robot):
        folderpath = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(folderpath, self.ur_file)
        if not os.path.exists(file_path):
            robotlog = {"last_pose": 0}
            print("Caution. Creating a new robotlog.json file with new value '0' for the last pose.")
            with open(self.ur_file, 'w') as json_file:
                json.dump(robotlog, json_file)
            return 0
        else:
            with open(self.ur_file,'r') as json_file:
                robotlog=json.load(json_file)
                print(f"robotlog.json found. Last commanded pose is "+str(robotlog["last_pose"])+"'.")
            return robotlog["last_pose"]

    def ur_ready(self):
        # Checks if everything is setup to perform a movement.
        control = []
        print(f"Initiliazing communication with UR at {self.ip}...")
        response = self.ur_ping()
        print(f"Ping response: {response}")
        print(f"Condition: {self.condition}")
        control.append(self.condition)
        self.ur_connection()
        print(f"Connection status: {self.connected}")
        control.append(self.connected)
        self.ur_status()
        print(f"Robot status: {self.safety_status}")
        control.append(self.safety_status)
        return response and self.condition and self.connected and self.safety_status
    
    def ur_status(self):
        safety_bits = self.rtde_r.getSafetyStatusBits()
        if safety_bits == 0 or safety_bits == 1:
            self.safety_status = True
        else:
            self.safety_status = True
            print("Robot is in error state. Check teach pendant to continue.")
            
    def ur_ping(self):
        response = subprocess.run(['ping', '-c', '1', '-W', '1', self.ip])
        if response.returncode == 0:
            try:
                self.rtde_r = RTDEReceiveInterface(self.ip)
                self.rtde_io = RTDEIOInterface(self.ip)
                self.rtde_c = RTDEControlInterface(self.ip)
                self.condition=True
                return True
            except RuntimeError:
                self.condition=False
                print("No control possible on Robot. Enable remote control on Robot teach pendant.")
                return False
        else:
            print("UR is not powered or not connected physically.")
            self.condition=False
            return False

    def ur_connection(self,state=None):
        if state is None:
            self.connected = self.rtde_r.isConnected()
            print(f"Connection status: {self.connected}")
        elif not state and self.condition:
            self.rtde_r.disconnect()
            self.connected = self.rtde_r.isConnected()
            print(f"UR is disconnected. Connection status: {self.connected}")
        else:
            # if state is True:
            if self.condition and not self.rtde_r.disconnect():
                self.rtde_r.reconnect()
                self.connected = self.rtde_r.isConnected()
                print(f"UR is re-connected. Connection status: {self.connected}")
            else:
                self.connected = self.rtde_r.isConnected()
                print(f"Connection status: {self.connected}. Please check Teach Pendant.")

    def ur_receive(self):
        if self.condition:
            self.actual_tcp = self.rtde_r.getActualTCPPose()
        else:
            self.actual_tcp = None
            print("No TCP pose obtained from UR. Check UR connection.")
        return self.actual_tcp
    
    def ur_output(self,num=None, state=None):
        if num is None:
            for output in range(0,self.num_outputs):
                try:
                    if self.rtde_r.getDigitalOutState(output):
                        if debug_mode:
                            print("UR output States: \n")
                            print(f"{output} : {self.rtde_r.getDigitalOutState(output)} \n")
                            print(" --------------- \n")
                except IndexError:
                    print(f"end of outputs at: {output}")
        elif num is not None and state is None:
            return self.rtde_r.getDigitalOutState(num)
        else:
            self.rtde_io.setStandardDigitalOut(num, state) 
            self.rtde_io.setToolDigitalOut(num, state) 
            time.sleep(0.5)

    def ur_gripper(self, action):
        if action=='open':
            state=False
        elif action=='close':
            state=True
        else:
            print("Unknown gripper command.")
        self.ur_output(0, not state)
        self.ur_output(1, state)

    def ur_move_simple(self,pose):
        if self.condition:
            self.ur_motion = True
            self.rtde_c.moveL(pose)
        else:
            self.ur_motion = False
            print("No Control possible. Cannot move robot.")
            print("Check 'Remote Control' option on Teach Pendant.")

    def ur_move_from_A_to_B(self, A, B):
        self.last_pose_saved = False
        # check last commanded pose:
        try:
            folderpath = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(folderpath, self.ur_file)
            with open(file_path, 'r') as json_file:
                robotlog=json.load(json_file)
        except FileNotFoundError:
            print("robotlog.json not found.")

        last_destination_pose = robotlog["last_pose"]
        # make selection more robust:
        if A<0 or A>(len(self.poses)-1):
            self.ur_motion = False
            if debug_mode:
                print("Selection for robot start pose Out of Bounds!.")     
        elif B<0 or B>(len(self.poses)-1):
            self.ur_motion = False
            if debug_mode:
                print("Selection for robot end pose Out of Bounds!.")
        else:
            #if A!=current_pose:
            if A!=last_destination_pose or B==last_destination_pose:
                if B==last_destination_pose:
                    self.ur_motion = False
                    if debug_mode:
                        print(f"Robot is already at pose {B}.")  
                else:
                    self.ur_motion = False
                    if debug_mode:
                        print("Please check the commanded poses again. Robot cannot execute motion!")   
            else:
                self.ur_motion = True
                if debug_mode:
                    print(f"Robot moving from pose {A} to pose {B}.")
                step=1
                if A>B:
                    step=-1
                for pose in range(A, B+step, step):
                    if debug_mode:
                        print(f"Pose: {pose}")
                    self.ur_move(self.poses[pose])
                    time.sleep(0.1)

                robotlog["last_pose"] = B
                with open(file_path, 'w') as json_file:
                    json.dump(robotlog, json_file)
                    self.last_pose_saved = True

                self.ur_motion = False

    def ur_perform_action(self, action, sort=None, grasp_target=None):
        # target should be of form: [pre_grasp, grasp]
        # Check what "last_pose" is to perform action under that specific pose
        try:
            folderpath = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(folderpath, self.ur_file)
            with open(file_path,'r') as json_file:
                robotlog=json.load(json_file)
                last_destination_pose = robotlog["last_pose"]
        except FileNotFoundError:
            print("robotlog.json not found.")

        if self.last_pose_saved is True:
            #remember to change moveL to True to stop motion (asynchronous)
            if last_destination_pose == 3 and action == 0:
                self.ur_gripper('close')
                self.ur_move_simple(self.actions[action])
            elif last_destination_pose == 2 and action == 1:
                self.ur_move_simple(self.actions[action])
                self.ur_gripper('open')
                time.sleep(2)
            elif last_destination_pose == 5 and action == 2:
                self.sorting_box = sort
                if self.sorting_box is None:
                    print("No sorting box provided.")
                elif isinstance(self.sorting_box, int):
                    self.ur_gripper('close')
                    self.ur_move_simple(0, self.actions[action][self.sorting_box])
                    time.sleep(1)
                    self.ur_gripper('open')
                    time.sleep(2)
                else:
                    print("No integer value for sorting box provided.")

            elif last_destination_pose == 1 or last_destination_pose == 4:
                self.grasp_target=grasp_target
                if self.grasp_target is None:
                    print("No grasp target provided.")
                elif isinstance(self.grasp_target) and len(self.grasp_target)==2:
                    self.ur_gripper('open')
                    self.ur_move_simple(self.grasp_target)
                    time.sleep(2)
                    self.ur_gripper('close')
                    time.sleep(2)
                else:
                    print("No proper grasp target [pre_grasp, grasp] provided. \n")
                    print(f"grasp_target is {self.grasp_target} with type {type(self.grasp_target)} and length {len(self.grasp_target)} \n")
            else:
                print("Not under legal pose to peform that action!")
        else:
            print("Switch pose for action. \n")
            print(f"Current pose at {last_destination_pose} \n")

    def ur_save_pose(self):
        poses = {
            1: "qs_inspection_pose",
            2: "qs_inspection_rot_z_pose",
            3: "qs_inspection_rot_x_pose",
            4: "conveyor_drop",
            5: "conveyor_wait",
            6: "vibrating_pregrasp_pose",
            7: "vibrating_grasp_pose",
            8: "strahlkabine_pregrasp_pose",
            9: "strahlkabine_grasp_pose",
            10: "strahlkabine_driveout_pose1",
            11: "strahlkabine_driveout_pose2",
            12: "sorting_predrop_pose1",
            13: "sorting_drop_pose1",
            14: "sorting_predrop_pose2",
            15: "sorting_drop_pose2",
            16: "sleep_pose",
            17: "start_pose",
            18: "vibratingplate_pose",
            19: "conveyor_pose",
            20: "qs_pose",
            21: "strahlkabine_pose",
            22: "sorting_pose"
        }
        def pose_selection():
            while True:
                try:
                    print("Poses: ")
                    for num, name in poses.items():
                        print(f"{num}: {name}")
                    pose_select=int(input("Select pose to register: "))
                    if 1<=pose_select<=22:
                        return pose_select
                    else:
                        print("Number out of range. Range is between 1 and 22.")
                except ValueError:
                    print("Invalid input. Enter an integer number.")

        def save_pose_to_json(key, pose):
            folderpath = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(folderpath, self.ur_poses)
            try:
                with open(file_path, 'r') as file:
                    data=json.load(file)
            except FileNotFoundError:
                print("Cannot find an existing poses.json file! \n Saving new file.")
                data = {}
            data[key]=pose
            with open(file_path,'w') as file:
                json.dump(data, file, indent=4)
            print(f"New pose saved as {key}: {pose} to {file_path}.")
        
        selected_pose=pose_selection()
        pose_key=poses[selected_pose]
        if self.ur_ready():
            ur_tcp_pose=self.ur_receive()
        save_pose_to_json(pose_key, ur_tcp_pose)



    def ur_poses_initiliaze(self):
        folderpath = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(folderpath, self.ur_poses)
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Variables obtained from JSON data:
        velocity = data['velocity']
        acceleration = data['acceleration']
        blend = data['blend']

        qs_inspection_pose = data['qs_inspection_pose'] + [velocity, acceleration, blend]
        qs_inspection_rot_z_pose = data['qs_inspection_rot_z_pose'] + [velocity, acceleration, blend]
        qs_inspection_rot_x_pose = data['qs_inspection_rot_x_pose'] + [velocity, acceleration, blend]
        part_inspection = [qs_inspection_pose, 
                           qs_inspection_rot_z_pose,
                           qs_inspection_pose,
                           qs_inspection_rot_x_pose,
                           qs_inspection_pose]

        conveyor_drop = data['conveyor_drop'] + [velocity, acceleration, blend]
        conveyor_wait = data['conveyor_wait'] + [velocity, acceleration, blend]
        part_conveyor = [conveyor_wait, 
                         conveyor_drop]
        
        vibrating_pregrasp_pose = data['vibrating_pregrasp_pose'] + [velocity, acceleration, blend]
        vibrating_grasp_pose = data['vibrating_grasp_pose'] + [velocity, acceleration, blend]
        part_vibratingplate = [vibrating_pregrasp_pose, 
                               vibrating_grasp_pose]

        strahlkabine_pregrasp_pose = data['strahlkabine_pregrasp_pose'] + [velocity, acceleration, blend]
        strahlkabine_grasp_pose = data['strahlkabine_grasp_pose'] + [velocity, acceleration, blend]
        part_strahlkabine_pick = [strahlkabine_pregrasp_pose, 
                                  strahlkabine_grasp_pose]

        strahlkabine_driveout_pose1 = data['strahlkabine_driveout_pose1'] + [velocity, acceleration, blend]
        strahlkabine_driveout_pose2 = data['strahlkabine_driveout_pose2'] + [velocity, acceleration, blend]
        part_strahlkabine_driveout = [strahlkabine_driveout_pose1, 
                                      strahlkabine_driveout_pose2]

        sorting_predrop_pose1 = data['sorting_predrop_pose1'] + [velocity, acceleration, blend]
        sorting_drop_pose1 = data['sorting_drop_pose1'] + [velocity, acceleration, blend]
        sorting_predrop_pose2 = data['sorting_predrop_pose2'] + [velocity, acceleration, blend]
        sorting_drop_pose2 = data['sorting_drop_pose2'] + [velocity, acceleration, blend]
        part_sorting = [[sorting_predrop_pose1, sorting_drop_pose1, sorting_predrop_pose1],
                        [sorting_predrop_pose2, sorting_drop_pose2, sorting_predrop_pose2]]

        sleep_pose = data['sleep_pose']
        start_pose = data['start_pose']
        vibratingplate_pose = data['vibratingplate_pose']
        conveyor_pose = data['conveyor_pose']
        qs_pose = data['qs_pose']
        strahlkabine_pose = data['strahlkabine_pose']
        sorting_pose = data['sorting_pose']
        # Pose and Index
        self.poses = [start_pose, # 0
                    vibratingplate_pose, # 1
                    conveyor_pose, # 2
                    qs_pose, # 3
                    strahlkabine_pose, # 4
                    sorting_pose] # 5
        # Action and Index
        self.actions = [part_inspection, # 0
                        part_conveyor, # 1
                        part_sorting, # 2
                        part_vibratingplate, # 3
                        part_strahlkabine_pick, # 4
                        part_strahlkabine_driveout] # 5
                        

if __name__ == "__main__":
    ur_autoclean = URRealTime()
    ur_autoclean.ur_poses_initiliaze()


    
