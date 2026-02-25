from utils.PID import *
from utils.mma import *

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators
from sensor_msgs.msg import Imu

import math
import threading
from collections import deque

import matplotlib
matplotlib.use('TkAgg')   # change to 'Qt5Agg' if TkAgg not available
import matplotlib.pyplot as plt

class RPYControllerNode(Node):
    def __init__(self):
        super().__init__('rpy_controller_node')
        # Set Kp , Ki , Kd
        self.k_roll     = [0,0,0]
        self.k_pitch    = [0,0,0]
        self.k_yaw      = [0,0,0]
        self.k_thrust   = [500,0,0]
        # Init controller
        self.roll_controller     = PID(kp=self.k_roll[0], ki=self.k_roll[1], kd=self.k_roll[2])
        self.pitch_controller    = PID(kp=self.k_pitch[0], ki=self.k_pitch[1], kd=self.k_pitch[2])
        self.yaw_controller      = PID(kp=self.k_yaw[0], ki=self.k_yaw[1], kd=self.k_yaw[2])
        self.thrust_controller   = PID(kp=self.k_thrust[0], ki=self.k_thrust[1], kd=self.k_thrust[2])

        # Create Sub & Pub
        self.cmd_pub  = self.create_publisher(Actuators, '/motor_commands', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self._odom_received = False
        # Timer & Control
        self.dt = 0.002
        self.control_timer  = self.create_timer(self.dt, self._control_loop)

        # Set refference
        self.ref_roll       = 0
        self.ref_pitch      = 0
        self.ref_yaw        = 0
        self.ref_altitute   = 5

        # Control param
        self.altitute       = 0
        self.roll           = 0
        self.pitch          = 0
        self.yaw            = 0

    def _odom_cb(self, msg):
        
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        q = msg.pose.pose.orientation

        _, _, self.altitute = p.x, p.y, p.z
        # self.vel_x, self.vel_y, self.vel_z = v.x, v.y, v.z

        # Quaternion → Euler (ZYX)
        # Roll
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self.roll = math.atan2(sinr, cosr)
        # Pitch
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        sinp = max(-1.0, min(1.0, sinp))
        self.pitch = math.asin(sinp)
        # Yaw
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

        self._odom_received = True

    def _control_loop(self):

        if self._odom_received == False:
            return
        
        alt_err = self.ref_altitute - self.altitute
        thrust_cmd = self.thrust_controller.compute(error=alt_err, dt=self.dt)

        FR_vel, HL_vel, FL_vel, HR_vel = mma(thrust_cmd, self.ref_roll, self.ref_pitch, self.ref_yaw)        

        self.pub_motor_cmd(FR_vel , HL_vel , FL_vel , HR_vel)
        print(f"error : {alt_err}")
    def pub_motor_cmd(self, FR_vel , HL_vel , FL_vel , HR_vel):
        cmd = Actuators()
        cmd.velocity = [FR_vel , HL_vel , FL_vel , HR_vel]
        self.cmd_pub.publish(cmd)


# --- Entry point -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = RPYControllerNode()
    rclpy.spin(node)

    # try:
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()
        # plt.ioff()
        # plt.show()   # keep final plot open after Ctrl+C


if __name__ == '__main__':
    main()
