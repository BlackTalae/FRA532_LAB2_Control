# Quadrotor Control Project: Optimal & Predictive Control

This project explores the design and implementation of advanced control strategies for a quadrotor in a ROS 2 / Gazebo simulation environment. It covers **Linear Quadratic Regulator (LQR)**, **Integral-LQR (LQI)**, and **Model Predictive Control (MPC)**.

---

## 1. System Architecture

The project consists of three primary control nodes:
1.  **LQR Controller (`LQR.py`)**: A 12-state optimal controller targeting hover and position hold.
2.  **LQI Controller (`LQI.py`)**: An extension of LQR with 4 integral states ($x, y, z, \psi$) to eliminate steady-state error under constant disturbances.
3.  **MPC Controller (`MPC.py`)**: A predictive controller that optimizes control inputs over a future horizon ($N=20$) for high-performance trajectory tracking.
4.  **Trajectory Generator (`send_trajectory_2.py`)**: A utility to publish time-varying reference paths (Circle, Helix, etc.).

---

## 2. Dynamic Modeling & Linearization

The quadrotor is a nonlinear system. For linear control (LQR/LQI/MPC), we linearize it around the **hover equilibrium**.

### A. Linearized State-Space Model
The system is expressed as: $\mathbf{\dot{x}} = A\mathbf{x} + B\mathbf{u}$

**State Vector ($\mathbf{x}$):** 12 states
$\mathbf{x} = [x, y, z, \phi, \theta, \psi, \dot{x}, \dot{y}, \dot{z}, \dot{\phi}, \dot{\theta}, \dot{\psi}]^T$

**Input Vector ($\mathbf{u}$):** 4 inputs
$\mathbf{u} = [F_{total}, \tau_{roll}, \tau_{pitch}, \tau_{yaw}]^T$

**Linearized Matrices (A & B):**
Assuming small angles ($\phi, \theta \approx 0$) and $F_{total} \approx mg$:
- $\ddot{x} \approx g \cdot \theta$
- $\ddot{y} \approx -g \cdot \phi$
- $\ddot{z} \approx \frac{dF}{m}$
- $\ddot{\phi}, \ddot{\theta}, \ddot{\psi}$ are governed by $\frac{\tau}{I}$ ratios.

---

## 3. Moment of Inertia Calculation

Accurate physics modeling is crucial. The moments of inertia ($I_{xx}, I_{yy}, I_{zz}$) include the base link and the four rotors using the **Parallel Axis Theorem**:
$$I_{total} = I_{base} + \sum_{i=1}^{4} (I_{rotor, i} + m_{rotor} \cdot d_i^2)$$

**Final Calculated Values:**
- $I_{xx} = 0.0356547$ kg·m²
- $I_{yy} = 0.0705152$ kg·m²
- $I_{zz} = 0.0990924$ kg·m²

---

## 4. Control Strategies

### A. Linear Quadratic Regulator (LQR)
LQR is an optimal control method that minimizes a cost function $J$ to find the optimal gain matrix $K$. In this project, it is used for precise hover and position hold.

#### 1. Linearized State-Space Model
The system is modeled as $\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}$ linearized around the hover equilibrium ($\phi, \theta \approx 0, F_{total} \approx mg$):

$$
\begin{bmatrix} 
\dot{x} \\ \dot{y} \\ \dot{z} \\ \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \\ \ddot{x} \\ \ddot{y} \\ \ddot{z} \\ \ddot{\phi} \\ \ddot{\theta} \\ \ddot{\psi} 
\end{bmatrix} = 
\begin{bmatrix} 
0_{6 \times 6} & I_{6 \times 6} \\
\begin{matrix} 0 & 0 & 0 & 0 & g & 0 \\ 0 & 0 & 0 & -g & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 \\ \vdots & & & \dots & & 0 \end{matrix} & 0_{6 \times 6} 
\end{bmatrix} 
\mathbf{x} + 
\begin{bmatrix} 
0_{8 \times 4} \\
\text{diag}(\frac{1}{m}, \frac{1}{I_{xx}}, \frac{1}{I_{yy}}, \frac{1}{I_{zz}})
\end{bmatrix} 
\mathbf{u}
$$

*   **State Vector ($\mathbf{x}$)**: $[x, y, z, \phi, \theta, \psi, \dot{x}, \dot{y}, \dot{z}, \dot{\phi}, \dot{\theta}, \dot{\psi}]^T$
*   **Input Vector ($\mathbf{u}$)**: $[dF, \tau_r, \tau_p, \tau_y]^T$ where $dF = F_{total} - mg$.

#### 2. Weight Design (Q & R Matrices)
The controller minimizes $J = \int_{0}^{\infty} (\mathbf{x}^T Q \mathbf{x} + \mathbf{u}^T R \mathbf{u}) dt$ with the following weights:
- **Q (State Penalty)**: $\text{diag}(120, 120, 900, 10, 10, 1500, 10, 10, 50, 1, 1, 10)$
    - *High Z weight (900)*: Ensures rapid altitude correction.
    - *High Yaw weight (1500)*: Maintains strict directional heading.
- **R (Input Penalty)**: $\text{diag}(1, 1, 1, 0.001)$
    - *Low Yaw penalty (0.001)*: Allows high control effort for fast heading response.

#### 3. Key Features
*   **Body-Heading Transform**: World-frame errors are rotated into the drone's body-heading frame, ensuring stability at any yaw angle.
*   **Wind Robust Tuning**: High velocity damping and torque penalty ($R_{roll, pitch} = 5.0$ in some versions) are used to prevent simulation jitter under 4 m/s wind.
*   **Optimal Control**: Solves the Algebraic Riccati Equation (ARE) offline to obtain the gain matrix $K$.

#### 4. Implementation Procedure
The LQR controller in `LQR.py` is implemented through the following systematic steps:

1.  **Define Physical Parameters**: Extract mass ($m$), gravity ($g$), and moments of inertia ($I_{xx}, I_{yy}, I_{zz}$) from the URDF/Xacro files.
2.  **Build Linearized Model**: Construct state matrix $A$ and input matrix $B$ by linearizing dynamics around the hover point.
3.  **Define Cost Matrices ($Q$ & $R$)**: Assign weights to penalize state errors and control effort respectively.
4.  **Solve Algebraic Riccati Equation (ARE)**: Compute the unique positive-definite matrix $P$ using `scipy.linalg.solve_continuous_are`:
    $$A^T P + PA - PBR^{-1}B^T P + Q = 0$$
5.  **Compute Optimal Gain ($K$)**: Calculate the gain matrix as $K = R^{-1} B^T P$.
6.  **Apply Control Law**: Compute the control correction $\mathbf{u}_{corr} = K \cdot (\mathbf{x}_{ref} - \mathbf{x}_{current})$ and add it to the hover trim $\mathbf{u}_{hover}$.
7.  **Motor Mixing**: Map the control inputs $[F, \tau_r, \tau_p, \tau_y]$ to individual motor speeds using $\Gamma^{-1}$.

### B. Integral-LQR (LQI)
LQI augments the system to 16 states. By integrating the position error, it provides the "muscle" to push back against constant wind without an offset:
$$ \mathbf{x}_{aug} = [\mathbf{x}_{12}, \int e_x, \int e_y, \int e_z, \int e_\psi]^T $$

### C. Model Predictive Control (MPC)
The MPC node solves a constrained optimization problem at every time step ($DT=0.05s$):
$$\min_{U} \sum_{k=0}^{N-1} (\mathbf{x}_k^T Q \mathbf{x}_k + \mathbf{u}_k^T R \mathbf{u}_k) + \mathbf{x}_N^T Q_N \mathbf{x}_N$$
*   **Horizon ($N=20$)**: Looks 1.0s into the future.
*   **Constraints**: Motor speeds are clipped within $[0, 1500]$ rad/s.
*   **Solver**: Uses `scipy.optimize.minimize` (L-BFGS-B) with Hessian normalization for numerical stability.

---

## 5. Motor Mixing (Allocation Matrix)

Control efforts $[F, \tau_r, \tau_p, \tau_y]$ are converted to motor speeds $\omega_i^2$ via the allocation matrix $\Gamma$:
$$
\begin{bmatrix} F \\ \tau_r \\ \tau_p \\ \tau_y \end{bmatrix} = \begin{bmatrix} 
k_F & k_F & k_F & k_F \\
-k_F \cdot L_y & k_F \cdot L_y & k_F \cdot L_y & -k_F \cdot L_y \\
-k_F \cdot L_x & k_F \cdot L_x & -k_F \cdot L_x & k_F \cdot L_x \\
-k_F \cdot k_M & -k_F \cdot k_M & k_F \cdot k_M & k_F \cdot k_M 
\end{bmatrix} 
\begin{bmatrix} \omega_0^2 \\ \omega_1^2 \\ \omega_2^2 \\ \omega_3^2 \end{bmatrix}
$$
*Note: $L_x=0.13, L_y=0.22$ based on URDF geometry.*

---

## 6. Drone Configuration & Parameters

The quadrotor follows an **X-configuration** with a standard **ENU (East-North-Up)** axis convention for the world frame.

### A. Coordinate System & Axis Layout
- **X-axis (Front)**: Directed towards the front of the drone (between Motor 0 and 2).
- **Y-axis (Left)**: Directed towards the left of the drone (between Motor 1 and 2).
- **Z-axis (Up)**: Directed upwards, perpendicular to the motor plane.
- **Roll ($\phi$)**: Rotation about the X-axis.
- **Pitch ($\theta$)**: Rotation about the Y-axis.
- **Yaw ($\psi$)**: Rotation about the Z-axis.

### B. Motor Layout (Top View)
```text
      Front (+X)
         ^
         |
    (FL) 2   0 (FR)
      \ / \ /
       X   X  ---> (+Y) Left
      / \ / \
    (RL) 1   3 (RR)
         |
```
| Motor ID | Position | Rotation | Lever Arm ($L_x, L_y$) |
| :--- | :--- | :--- | :--- |
| **0** | Front-Right | CCW (+) | $0.13, -0.22$ |
| **1** | Rear-Left | CCW (+) | $-0.13, 0.20$ |
| **2** | Front-Left | CW (-) | $0.13, 0.22$ |
| **3** | Rear-Right | CW (-) | $-0.13, -0.20$ |

### C. Physical Parameters
The following values are used for dynamic modeling and control law computation:

| Parameter | Symbol | Value | Units |
| :--- | :--- | :--- | :--- |
| Total Mass | $m$ | 1.525 | kg |
| Gravity | $g$ | 9.81 | m/s² |
| Thrust Coefficient | $k_F$ | $8.54858 \times 10^{-6}$ | N/(rad/s)² |
| Torque Coefficient | $k_M$ | 0.06 | N·m / N |
| Max Motor Speed | $\omega_{max}$ | 1500 | rad/s |
| Roll Inertia | $I_{xx}$ | 0.0356547 | kg·m² |
| Pitch Inertia | $I_{yy}$ | 0.0705152 | kg·m² |
| Yaw Inertia | $I_{zz}$ | 0.0990924 | kg·m² |

---

## 7. Test Trajectories

The `send_trajectory_2.py` utility provides several predefined patterns for testing controller performance. These trajectories generate a time-sequenced `nav_msgs/Path` (lookahead path) for the MPC and a `geometry_msgs/PoseStamped` for LQR.

### A. Available Trajectory Types

| Mode | Description | Key Parameters |
| :--- | :--- | :--- |
| **HOVER** | Holds a fixed position at a specific altitude. | $z, dwell$ |
| **VERTICAL** | Linear motion along the Z-axis. | $z_{min}, z_{max}, speed$ |
| **CIRCLE** | A circular path in the X-Z plane. | $radius, speed$ |
| **XZ_SQUARE** | A square path in the X-Z plane. | $size, speed$ |
| **SINE** | A sinusoidal wave along the X-axis in the X-Z plane. | $amp, freq, speed$ |
| **HELIX** | A 3D helical upward path. | $radius, turns, height, speed$ |
| **SPIRAL** | An inward-conical spiral path. | $radius_{start}, turns, height, speed$ |
| **FIG_8** | A complex 3D figure-eight pattern. | $width, height, depth, speed$ |
| **FIG_8_2D** | A 2D figure-eight pattern in the X-Z plane. | $width, height, speed$ |
| **LIN_3D** | A linear point-to-point path in 3D space. | $start, target, speed$ |

### B. Features
- **Lookahead Path**: Generates $N=20$ future poses (spaced at $DT=0.05s$) for predictive control (MPC).
- **Round-Trip**: Can be configured to traverse waypoints in reverse back to the start.
- **Yaw Alignment**: Automatically orients the drone's heading ($yaw$) to face the direction of motion.
- **Speed-Based**: Segment durations are calculated dynamically based on a constant target speed (m/s).

---

## 8. Performance Metrics & Visualization

To evaluate and compare the performance of different controllers (LQR, LQI, MPC), we use the following quantitative metrics and visualization strategies.

### A. Performance Metrics
We measure the error between the reference trajectory $\mathbf{x}_{ref}(t)$ and the actual state $\mathbf{x}(t)$.

- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors.
  $$MAE = \frac{1}{T} \int_{0}^{T} |e(t)| dt$$
- **Root Mean Square Error (RMSE)**: Penalizes larger deviations more heavily, useful for detecting oscillations.
  $$RMSE = \sqrt{\frac{1}{T} \int_{0}^{T} e(t)^2 dt}$$
- **Peak Error ($e_{max}$)**: Maximum deviation from the path, critical for obstacle avoidance.
  $$e_{max} = \max |e(t)|$$
- **Robustness**: The maximum wind disturbance (m/s) the drone can withstand while maintaining a position error within a defined tolerance (e.g., < 0.2m).

### B. Visualization Strategies
- **3D Path Analysis**: Plotting the actual 3D path against the reference path to visualize "lag" or "cutting corners" (common in LQR without lookahead).
- **State-over-Time**: Individual plots for $X, Y, Z$, and $Yaw$ to identify which axes are most susceptible to disturbances.
- **Control Input Monitoring**: Visualization of $F_{total}$ and $\tau$ to ensure the controller is not saturating the motors or causing high-frequency jitter (over-tuning).
- **Tracking Error Heatmap**: (Optional) Projecting tracking error onto the path to identify high-stress segments (e.g., tight turns).

*Recommended Tool: [PlotJuggler](https://github.com/facontidavide/PlotJuggler) for real-time ROS 2 topic visualization.*

---

## 9. How to Run

1.  **Launch Simulation**:
    ```bash
    ros2 launch quad_description sim.launch.py controller:=MPC world:=wind.sdf
    ```
2.  **Start Trajectory**:
    ```bash
    # You can edit the MODE in send_trajectory_2.py main() before running
    ros2 run quad_description send_trajectory_2.py
    ```

---

## 10. Results & Summary
- **LQR**: Stable hover, but drifts slightly in wind.
- **LQI**: Zero steady-state error in 4 m/s wind.
- **MPC**: Superior tracking of complex trajectories (like Helix/Fig-8) with minimal lag due to its predictive nature.
