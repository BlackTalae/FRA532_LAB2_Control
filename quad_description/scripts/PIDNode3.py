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
        # self.k_roll     = [2.0, 0.0085, 0.06]
        # self.k_pitch    = [2.0, 0.0085, 0.06]
        # self.k_yaw      = [0.015 ,0.01 , 0.0]
        # self.k_thrust   = [2 , 0.08 , 0.002] 

        # self.k_roll     = [2.5, 0.0015, 0.06]
        # self.k_pitch    = [2.5, 0.0015, 0.06]
        # self.k_yaw      = [0.015 ,0.005 , 0.0]
        # self.k_thrust   = [2 , 0.08 , 0.002] 

        self.k_roll     = [0.00002, 0.0, 0.0]
        self.k_pitch    = [0.00002, 0.0, 0.0]
        self.k_yaw      = [0.0 ,0.0 , 0.0]
        self.k_thrust   = [2 , 0.08 , 0.002] 

        self.k_x        = [0.2, 0.0003, 0.005]
        self.k_y        = [2.0, 0.0085, 0.06]
        self.k_z        = [2.0, 0.0085, 0.06]

        self.F_MAX_RPY    = 3.0 # PID u
        self.F_MAX_THRUST = 3.0 # PID u
        self.F_MAX_ROTOR  = 15.0 # Command

        self.OMEGA_MAX    = 1500.0
        self.roll_controller     = PID(kp=self.k_roll[0]  , ki=self.k_roll[1]  , kd=self.k_roll[2]   , out_max=self.F_MAX_RPY, out_min=-self.F_MAX_RPY)
        self.pitch_controller    = PID(kp=self.k_pitch[0] , ki=self.k_pitch[1] , kd=self.k_pitch[2]  , out_max=self.F_MAX_RPY, out_min=-self.F_MAX_RPY)
        self.yaw_controller      = PID(kp=self.k_yaw[0]   , ki=self.k_yaw[1]   , kd=self.k_yaw[2]    , out_max=self.F_MAX_RPY, out_min=-self.F_MAX_RPY)
        self.thrust_controller   = PID(kp=self.k_thrust[0], ki=self.k_thrust[1], kd=self.k_thrust[2] , out_max=self.F_MAX_THRUST, out_min=-self.F_MAX_THRUST)

        self.pos_x_controller    = PID(kp=self.k_x[0]     , ki=self.k_x[1]     , kd=self.k_x[2])
        self.pos_y_controller    = PID(kp=self.k_y[0]     , ki=self.k_y[1]     , kd=self.k_y[2])
        self.pos_z_controller    = PID(kp=self.k_z[0]     , ki=self.k_z[1]     , kd=self.k_z[2])
        # Create Sub & Pub
        self.cmd_pub  = self.create_publisher(Actuators, '/motor_commands', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self._imu_cb, 10)
        self._odom_received = False
        # Timer & Control
        self.inner_dt = 0.005
        self.outer_dt = 0.005
        self.inner_control_timer  = self.create_timer(self.inner_dt, self._inner_loop)
        self.outer_control_timer  = self.create_timer(self.outer_dt, self._outer_loop)

        self.enable_outer_loop = False

        # Set refference
        self.ref_roll       = 0
        self.ref_pitch      = 1
        self.ref_yaw        = 0
        self.ref_vel_z      = 0
        self.ref_pos_x      = 0
        self.ref_pos_y      = 0 
        self.ref_pos_z      = 0

        # Control param
        self.vel_z          = 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        self.pos_x, self.pos_y, self.pos_z  = 0, 0, 0

        # ---- Debug publishers (multiple floats) ----
        self.err_pub = self.create_publisher(Float32MultiArray, '/pid/error', 10)
        self.cmd_pub_dbg = self.create_publisher(Float32MultiArray, '/pid/cmd', 10)
        self.hpy_pub = self.create_publisher(Float32MultiArray, '/pid/hpy', 10)

    def _imu_cb(self,msg):
        pass
    def _odom_cb(self, msg):
        
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        q = msg.pose.pose.orientation

        self.pos_x, self.pos_y, self.pos_z = p.x, p.y, p.z
        _, _, self.vel_z = v.x, v.y, v.z

        self.roll , self.pitch , self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._odom_received = True
        
    def _outer_loop(self):
        if self.enable_outer_loop == False:
            return
        if self._odom_received == False:
            return
        
        x_err = self.ref_pos_x - self.pos_x
        y_err = self.ref_pos_y - self.pos_y
        z_err = self.ref_pos_z - self.pos_z

        self.ref_pitch  = self.pos_x_controller.compute(error=x_err, dt=self.outer_dt)

    def _inner_loop(self):
        if self._odom_received == False:
            return


        vel_z_err     = self.ref_vel_z - self.vel_z

        # World-frame angle errors
        roll_err_world  = wrap_pi(self.ref_roll  - self.roll)
        pitch_err_world = wrap_pi(self.ref_pitch - self.pitch)

        # Rotate to body frame using current yaw
        # cos_yaw,sin_yaw = math.cos(self.yaw),math.sin(self.yaw)

        # roll_err  =  cos_yaw * roll_err_world + sin_yaw * pitch_err_world
        # pitch_err = -sin_yaw * roll_err_world + cos_yaw * pitch_err_world
        roll_err  = wrap_pi(self.ref_roll  - self.roll)
        pitch_err = wrap_pi(self.ref_pitch - self.pitch)


        yaw_err     = wrap_pi(self.ref_yaw - self.yaw)
        thrust_cmd  = self.thrust_controller.compute(error=vel_z_err, dt=self.inner_dt,anti_windup=True)
        roll_cmd    = self.roll_controller.compute(error=roll_err,dt=self.inner_dt)
        pitch_cmd   = self.pitch_controller.compute(error=pitch_err,dt=self.inner_dt)
        yaw_cmd     = self.yaw_controller.compute(error=yaw_err,dt=self.inner_dt)

        hover_base = ((1.5 * 9.81)/4) + 0.046 #0.039 # 0.036 # mg
        FR_thrust, HL_thrust, FL_thrust, HR_thrust = mma(thrust_cmd, roll_cmd, -pitch_cmd, yaw_cmd, hover_base)
        
        kf = 8.54858e-06
        FR_vel = thrust_to_omega(FR_thrust, kf, self.F_MAX_ROTOR)
        HL_vel = thrust_to_omega(HL_thrust, kf, self.F_MAX_ROTOR)
        FL_vel = thrust_to_omega(FL_thrust, kf, self.F_MAX_ROTOR)
        HR_vel = thrust_to_omega(HR_thrust, kf, self.F_MAX_ROTOR)


        # print("ALTITUTE ERROR" , alt_err)
        # print(thrust_cmd , roll_cmd , pitch_cmd)
        # print(thrust_cmd , hover_base)
        # print(FL_vel , HL_vel , FR_vel , HR_vel )

        self.pub_motor_cmd(FR_vel , HL_vel , FL_vel , HR_vel)
        # print(self.roll , self.pitch)
        # self.pub_motor_cmd(700.0 , 700.0 , 700.0 , 700.0)
        
        # print(f"command : {thrust_cmd, roll_cmd, pitch_cmd}")
        # print(f"error : {alt_err}")
        # print(f"command : {FR_vel, HL_vel, FL_vel, HR_vel}")
        # print(f"command : {self.ref_altitute,self.altitute}")

        self.pub_pid_debug(
            vel_z_err,
            roll_err,
            pitch_err,
            thrust_cmd,
            roll_cmd,
            pitch_cmd,
            self.vel_z,
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
