import snap7
from snap7 import util
import shutil
import time
import subprocess

debug_mode=True
timeout=10.0
ip1="192.168.100.50"

class SPSBase:
    def __init__(self,ip):
        self.ip=ip
        self.client = None
        self.initialize()

    def initialize(self):
        if self.sps_ping():
            self.client = snap7.client.Client()
            self.client.connect(self.ip, 0, 1)
        else:
            print("SPS is not powered or not connected.")
        
    def sps_ping(self):
        response = subprocess.run(['ping', '-c', '1', '-W', '1', self.ip])
        if response.returncode == 0:
            return True
        else:
            return False

class SPSSensor(SPSBase):
    def __init__(self,ip,sensor_number):
        super().__init__(ip)
        if sensor_number>=2:
            self.sensor_num=sensor_number
        else:
            print("Can't assign lower DB value than 2.")

    def get_sensor_bool(self):
        db_read = self.client.db_read(16, 0, 1) ### Read DB, size 1 Byte
        SensorRead = util.get_bool(db_read, 0, self.sensor_num)
        return SensorRead

class SPSConveyor(SPSBase):
    def __init__(self,ip, conveyor_number, sensors):
        super().__init__(ip)
        self.antrieb_num=conveyor_number
        self.move_rel_done_num=conveyor_number+1
        self.sensor_in, self.sensor_out=sensors
        # Conveyor with stepper motor: Takes 0,1 in DB 16
        self.antrieb_ready=None
        self.move_rel_done=None
        
    def initialize_conveyor(self):
        self.power_conveyor(False)
        time.sleep(0.5)
        self.power_conveyor(True)
        stime=time.time()
        while True:
            db_read = self.client.db_read(16, 0, 1) ### Read DB 16, size 1 Byte
            self.antrieb_ready = util.get_bool(db_read, 0, self.antrieb_num)
            time.sleep(0.5)
            if self.antrieb_ready:
                if debug_mode:
                    print(f"Antrieb ready: {self.antrieb_ready}")
                return True
            else:
                if (time.time()-stime)>timeout:
                    print(f"Timed out after {timeout} seconds.")
                    self.power_conveyor(False)
                    return False
                else:
                    pass

    def power_conveyor(self, state):
        data = bytearray(1)
        util.set_bool(data,0,0,state) #
        self.client.db_write(17, 0, data)
        if debug_mode:
            print(f"Power internal motor drive: {state} ")

    def move_conveyor(self, state, dir=None, vel=50):
        data = bytearray(4)
        util.set_dint(data, 0, vel) # velocity
        self.client.db_write(17, 10, data)
        time.sleep(0.25)
        if state:
            self.power_conveyor(True)
            time.sleep(0.25)
            if dir=="FWD": 
                print(f"Moving conveyor1 forwards at velocity {vel} mm/s.")
                data = bytearray(1)
                util.set_bool(data,0,0,False) # set conveyor to move
                self.client.db_write(17,3, data) # 3 is backward byte
                util.set_bool(data,0,0,True) # set conveyor to move
                self.client.db_write(17,2, data) # 2 is forward byte
                
            elif dir=="BWD":
                print(f"Moving conveyor1 backwards at velocity {vel} mm/s.")
                data = bytearray(1)
                util.set_bool(data,0,0,False) # set conveyor to move
                self.client.db_write(17,2, data) # 2 is forward byte
                util.set_bool(data,0,0,True) # set conveyor to move
                self.client.db_write(17,3, data) # 3 is backward byte 
        else:
            print("Stopping conveyor1...")
            data = bytearray(1)
            util.set_bool(data,0,0,False) # set conveyor to move
            self.client.db_write(17,2, data) # 2 is forward byte
            util.set_bool(data,0,0,False) # set conveyor to move
            self.client.db_write(17,3, data) # 3 is backward byte 
            time.sleep(0.25)
            self.power_conveyor(False)

        data = bytearray(1)
        util.set_bool(data,0,0,state) # set conveyor to move
        self.client.db_write(17,2, data)

    def move_conveyor_1000(self):
        data = bytearray(4)
        util.set_dint(data, 0, 1000) # distance
        self.client.db_write(17,6,data)
        util.set_dint(data, 0, 100) # velocity
        self.client.db_write(17, 10, data)

        time.sleep(0.25)
        self.power_conveyor(True)
        data = bytearray(1)
        util.set_bool(data,0,0,False) # set conveyor to move
        self.client.db_write(17,2, data) # 2 is forward byte
        util.set_bool(data,0,0,True) # set conveyor to move
        self.client.db_write(17,3, data) # 3 is backward byte

    def simple_move(self):
        if self.initialize_conveyor():
            stime=time.time()
            if not self.sensor_out.get_sensor_bool():
                self.move_conveyor_1000()
            else:
                time.sleep(0.1)
                self.move_conveyor(False)
            while True:
                if (time.time()-stime)>timeout or self.sensor_out.get_sensor_bool():
                    time.sleep(0.1)
                    self.move_conveyor(False)
                else:
                    continue

    def simple_dump(self):
        # Used to clear the conveyor and count how many items were on
        self.parts_dumped=0
        if self.initialize_conveyor():
            stime=time.time()
            if self.sensor_out.get_sensor_bool():
                # perform dump
                self.move_conveyor_1000()
                time.sleep(0.1)
                c2_previous=self.sensor_out.get_sensor_bool()
                self.parts_dumped+=1
                # stay in while loop
                while True:
                    if self.sensor_out.get_sensor_bool() and not c2_previous:
                        self.parts_dumped+=1
                        if debug_mode:
                            print(f"Parts added to Strahlkabine:  {self.parts_dumped}")
                    c2_previous=self.sensor_out.get_sensor_bool()
                    if (time.time()-stime)>10:
                        time.sleep(0.1)
                        self.move_conveyor(False)
                        print(f"Total parts added: {self.parts_dumped}")
                # count parts
                # stop moving after set time (break out of loop)
                # save number of parts dumped
            else:
                # Nothing is present on conveyor
                print(f"Nothing to dump. Conveyor empty.")
                print(f"Sensor_IN: {self.sensor_in.get_sensor_bool()} \n Sensor_OUT: {self.sensor_out.get_sensor_bool()}")

class SPSVibratingPlate(SPSBase):
    def __init__(self,ip):
        super().__init__(ip)
        if self.sps_ping():
            db_read = self.client.db_read(18, 0, 6) # Read DB 18, 0 to read the whole DB, size 6 Byte 
            self.motor_ok = util.get_bool(db_read, 0, 0) # get_bool(datablock, byte, bit)
            self.motor_freq = util.get_dint(db_read, 2) # get_double integer (datablock, byte)
        else:
            self.motor_ok=False
        self.motor_status=False
        self.motor_frequency=0
        self.motor_direction=None
        # direction: False --> Forward
        # direction: True --> Reverse
        if self.client is not None:
            self.motor_initialize()

    def motor_initialize(self):
        self.motor_ena_rst_execute(False)
        self.motor_fwd_execute(False)
        self.motor_rev_execute(False)
        stime=time.time()
        while True:
            if self.motor_check():
                self.motor_ena_rst_execute(True)
                return True
            else:
                self.motor_ena_rst_execute(False)
                print(f"Waiting for motor for {time.time()-stime} s.")
                time.sleep(0.5)
                if (time.time()-stime)>timeout:
                    print(f"Timed out after {timeout} seconds. \n")
                    print("motor_check Time-out: Press the reset button on the motor driver.")
                    self.motor_ena_rst_execute(False)
                    return False
                else:
                    pass
            time.sleep(0.5)

    def change_motor_freq(self,freq):
        if isinstance(freq, (float, int)):
            freq = str(freq)
            if self.motor_read_freq() == freq:
                print(f"Frequency already at {freq}.")
            else:
                self.motor_read_freq(freq)
        else:
            print("Invalid frequency value. Please provide a float or an integer.")

    def plate_vibrate(self,state,direction):
        self.motor_direction = direction
        # direction: False --> Forward
        # direction: True --> Reverse
        if self.motor_initialize():
            if direction:
                self.motor_fwd_execute(False)
                self.motor_rev_execute(state)
            elif not direction:
                self.motor_fwd_execute(state)
                self.motor_rev_execute(False)
            track=True
        else:
            track=False # This prevents the motor from running continuously in case motor not OK anymore.
        if track and state:
            if direction:
                self.motor_fwd_execute(False)
                self.motor_rev_execute(state)
            elif not direction:
                self.motor_fwd_execute(state)
                self.motor_rev_execute(False)
        else:
            self.change_motor_freq(0)
            if direction:
                self.motor_fwd_execute(False)
                self.motor_rev_execute(state)
            elif not direction:
                self.motor_fwd_execute(state)
                self.motor_rev_execute(False)

    def motor_fwd_execute(self, state):
        # write MotorFwd byte 0 bit 0 in DB19
        data = bytearray(1)
        util.set_bool(data,0,0,state) # set_bool(datablock, byte, bit, STATE)
        self.client.db_write(19, 0, data) # write (datablock, byte, DATA)
        print(f"Motor Forward Execution state: {state}.")

    def motor_ena_rst_execute(self, state):
        # write MotorEnaRst byte 1 bit 0 in DB19
        data = bytearray(1)
        util.set_bool(data,0,0,state)  
        self.client.db_write(19, 1, data) 
        print(f"Motor EnaRst (Enable Reset) state: {state}.")

    def motor_rev_execute(self, state):
        # write MotorRev byte 2 bit 0 in DB19
        data = bytearray(1)
        util.set_bool(data,0,0,state) 
        self.client.db_write(19, 2, data) 
        print(f"Motor Reverse Execution state: {state}.")

    def motor_speed_ref(self, freq):
        # write MotorSpeedRef byte 4 bit 0 in DB19
        data = bytearray(4)
        util.set_dint(data, 0, freq) # frequency
        self.client.db_write(19, 4, data) # write (datablock, byte, DATA)
        print(f"Motor SpeedRef (Speed Reference) state: {freq}.")

    def motor_read_freq(self):
        db_read = self.client.db_read(18, 0, 6) ### Read DB 18, size 6 Byte
        self.motor_frequency = util.get_dint(db_read, 2) # get_double integer (datablock, byte)
        return self.motor_frequency

    def motor_check(self):
        db_read = self.client.db_read(18, 0, 6) ### Read DB 18, size 6 Byte
        self.motor_ok = util.get_bool(db_read, 0, 0)
        return self.motor_ok

    