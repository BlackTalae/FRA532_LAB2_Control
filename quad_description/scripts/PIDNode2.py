from utils.PID import *
from utils.mma import *

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Float32MultiArray
from tf_transformations import euler_from_quaternion

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
        self.k_roll     = [15, 0.2, 6.0]
        self.k_pitch    = [15, 0.2, 6.0]
        self.k_yaw      = [0,0,0]
        self.k_thrust   = [1500.0 , 0.2 , 0]
        # self.k_thrust   = [0.2 , 0.0002 , 0.0002]
        # self.k_thrust   = [0.0 , 0.0 , 0.0]
        # Init controller
        self.OMEGA_MAX_RPY    = 400.0
        self.OMEGA_MAX_THRUST = 400.0
        self.roll_controller     = PID(kp=self.k_roll[0]  , ki=self.k_roll[1]  , kd=self.k_roll[2]   , out_max=self.OMEGA_MAX_RPY, out_min=-self.OMEGA_MAX_RPY)
        self.pitch_controller    = PID(kp=self.k_pitch[0] , ki=self.k_pitch[1] , kd=self.k_pitch[2]  , out_max=self.OMEGA_MAX_RPY, out_min=-self.OMEGA_MAX_RPY)
        self.yaw_controller      = PID(kp=self.k_yaw[0]   , ki=self.k_yaw[1]   , kd=self.k_yaw[2]    , out_max=self.OMEGA_MAX_RPY, out_min=-self.OMEGA_MAX_RPY)
        self.thrust_controller   = PID(kp=self.k_thrust[0], ki=self.k_thrust[1], kd=self.k_thrust[2] , out_max=self.OMEGA_MAX_THRUST, out_min=-self.OMEGA_MAX_THRUST)

        # Create Sub & Pub
        self.cmd_pub  = self.create_publisher(Actuators, '/motor_commands', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self._odom_received = False
        # Timer & Control
        self.dt = 0.005
        self.control_timer  = self.create_timer(self.dt, self._control_loop)

        # Set refference
        self.ref_roll       = 0
        self.ref_pitch      = 0
        self.ref_yaw        = 0
        self.ref_altitute   = 10

        self.v_z            = 0

        # Control param
        self.altitute       = 0
        self.roll           = 0
        self.pitch          = 0
        self.yaw            = 0

        # ---- Debug publishers (multiple floats) ----
        self.err_pub = self.create_publisher(Float32MultiArray, '/pid/error', 10)
        self.cmd_pub_dbg = self.create_publisher(Float32MultiArray, '/pid/cmd', 10)
        self.hpy_pub = self.create_publisher(Float32MultiArray, '/pid/hpy', 10)


    def _odom_cb(self, msg):
        
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        q = msg.pose.pose.orientation

        _, _, self.altitute = p.x, p.y, p.z
        _, _, self.v_z = v.x, v.y, v.z

        self.roll , self.pitch , self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._odom_received = True

    def _control_loop(self):

        if self._odom_received == False:
            return


        alt_err     = self.ref_altitute - self.altitute
        roll_err    = wrap_pi(self.ref_roll - self.roll)
        pitch_err   = wrap_pi(self.ref_pitch - self.pitch)

        thrust_cmd  = self.thrust_controller.compute(error=alt_err, dt=self.dt , derivative_term=self.v_z)
        roll_cmd    = self.roll_controller.compute(error=roll_err,dt=self.dt)
        pitch_cmd   = self.pitch_controller.compute(error=pitch_err,dt=self.dt)
        
        hover_base = 1.5 * 9.81 # mg
        FR_vel, HL_vel, FL_vel, HR_vel = mma(thrust_cmd, -roll_cmd, -pitch_cmd, 0.0, hover_base)
        kf = 8.54858e-06
        FR_vel, HL_vel, FL_vel, HR_vel = math.sqrt(FR_vel), kf*HL_vel, kf*FL_vel, kf*HR_vel 

        # print("ALTITUTE ERROR" , alt_err)
        # print(thrust_cmd , roll_cmd , pitch_cmd)
        print(FL_vel , HL_vel , FR_vel , HR_vel )

        self.pub_motor_cmd(FR_vel , HL_vel , FL_vel , HR_vel)
        # print(self.roll , self.pitch)
        # self.pub_motor_cmd(700.0 , 700.0 , 700.0 , 700.0)
        
        # print(f"command : {thrust_cmd, roll_cmd, pitch_cmd}")
        # print(f"error : {alt_err}")
        # print(f"command : {FR_vel, HL_vel, FL_vel, HR_vel}")
        # print(f"command : {self.ref_altitute,self.altitute}")

        self.pub_pid_debug(
            alt_err,
            roll_err,
            pitch_err,
            thrust_cmd,
            roll_cmd,
            pitch_cmd,
            self.altitute,
            self.roll,
            self.pitch
        )

    def pub_motor_cmd(self, FR_vel , HL_vel , FL_vel , HR_vel):
        cmd = Actuators()
        cmd.velocity = [FR_vel , HL_vel , FL_vel , HR_vel]
        self.cmd_pub.publish(cmd)

    def pub_pid_debug(self, alt_err, roll_err, pitch_err,
                    thrust_cmd, roll_cmd, pitch_cmd,
                    heigh, roll, pitch):

        # errors
        e = Float32MultiArray()
        e.data = [float(alt_err),
                float(roll_err),
                float(pitch_err)]
        self.err_pub.publish(e)

        # commands
        u = Float32MultiArray()
        u.data = [float(thrust_cmd),
                float(roll_cmd),
                float(pitch_cmd)]
        self.cmd_pub_dbg.publish(u)

        f = Float32MultiArray()
        f.data = [float(heigh),
                float(roll),
                float(pitch)]
        self.hpy_pub.publish(f)

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
