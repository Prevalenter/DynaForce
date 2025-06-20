import sys
sys.path.append("../..")

import math
# from DM_CAN import *
# from contact_retarget
from contact_retarget.utils.motor.DM_CAN import *
from contact_retarget.utils.motor.rviz import ValveRviz
import serial
import time


class DMMotor:
    def __init__(self, port='/dev/ttyACM0', control_mode=3):
        self.Motor1=Motor(DM_Motor_Type.DM4310, 0x01, 0x11)

        self.serial_device = serial.Serial(port, 921600, timeout=0.5)
        self.MotorControl1=MotorControl(self.serial_device)
        self.switch_control_mode(control_mode)
        self.MotorControl1.addMotor(self.Motor1)

        self.MotorControl1.save_motor_param(self.Motor1)

        self.enable()

    def switch_control_mode(self, control_mode):
        # MIT = 1
        # POS_VEL = 2
        # VEL = 3
        # Torque_Pos = 4
        if control_mode == 1:
            if self.MotorControl1.switchControlMode(self.Motor1,Control_Type.MIT):
                print("switch MIT success")
        elif control_mode == 2:
            if self.MotorControl1.switchControlMode(self.Motor1,Control_Type.POS_VEL):
                print("switch POS_VEL success")
        elif control_mode == 3:
            if self.MotorControl1.switchControlMode(self.Motor1,Control_Type.VEL):
                print("switch VEL success")
        elif control_mode == 4:
            if self.MotorControl1.switchControlMode(self.Motor1,Control_Type.Torque_Pos):
                print("switch Torque_Pos success")
        # if self.MotorControl1.switchControlMode(self.Motor1,Control_Type.VEL):
        #     print("switch VEL success")

    def close_port(self):
        self.serial_device.close()

    def enable(self):
        self.MotorControl1.enable(self.Motor1)
        time.sleep(0.5)
        
    def disable(self):
        self.MotorControl1.disable(self.Motor1)
        time.sleep(0.5)

    def set_vel(self, vel):
        self.MotorControl1.control_Vel(self.Motor1, vel)
        
    def set_pos_force(self, pos, vel, force):
        self.MotorControl1.control_pos_force(self.Motor1, pos, vel, force)

    def fresh_motor_status(self):
        self.MotorControl1.refresh_motor_status(self.Motor1)
        
    def get_state(self):
        # self.MotorControl1.refresh_motor_status(self.Motor1)
        self.fresh_motor_status()
        return [self.Motor1.getPosition(), self.Motor1.getVelocity(), self.Motor1.getTorque()]



if __name__ == '__main__':
    motor = DMMotor(port='/dev/ttyACM1')
    rviz_show = ValveRviz()

    print('start')

    print('set vel 5')
    motor.set_vel(5)
    # time.sleep(5)
    for i in range(50):
        print(motor.get_state())
        rviz_show.pub_joint_pos([motor.get_state()[0]])
        time.sleep(0.1)
    
    print('set vel 0')
    motor.set_vel(0)
    time.sleep(1)
    print('set vel -5')
    motor.set_vel(-5)
    for i in range(50):
        print(motor.get_state())
        rviz_show.pub_joint_pos([motor.get_state()[0]])
        time.sleep(0.1)
    print('set vel 0')
    motor.set_vel(0)
    time.sleep(1)
       
    motor.disable()
    motor.close_port()
    print('done') 