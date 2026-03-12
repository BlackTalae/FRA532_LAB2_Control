# Quadrotor Control Project: Optimal & Predictive Control

This project explores the design and implementation of advanced control strategies for a quadrotor in Gazebo simulation environment. It covers **Linear Quadratic Regulator (LQR)** and **Model Predictive Control (MPC)**.

---

## 1. System Architecture

Add later

---

## 2. Dynamic Modeling & Linearization

The quadrotor is a nonlinear system. For linear control, we linearize it around the **hover equilibrium**.

### Full Nonlinear Dynamics
Translational accelerations in the world frame ($\ddot{x}, \ddot{y}, \ddot{z}$) and angular accelerations in the body frame ($\ddot{\phi}, \ddot{\theta}, \ddot{\psi}$):
- $\ddot{x} = \frac{1}{m} (\cos\phi \sin\theta \cos\psi + \sin\phi \sin\psi) F_{total}$

- $\ddot{y} = \frac{1}{m} (\cos\phi \sin\theta \sin\psi - \sin\phi \cos\psi) F_{total}$

- $\ddot{z} = \frac{1}{m} (\cos\phi \cos\theta) F_{total} - g$

- $\ddot{\phi} = \frac{\tau_{roll}}{I_{xx}}$

- $\ddot{\theta} = \frac{\tau_{pitch}}{I_{yy}}$

- $\ddot{\psi} = \frac{\tau_{yaw}}{I_{zz}}$

### Linearized Dynamics Around Hover
Assuming small angles ($\phi, \theta, \psi \approx 0$) and $F_{total} \approx mg$:
- $\ddot{x} \approx g \cdot \theta$

- $\ddot{y} \approx -g \cdot \phi$

- $\ddot{z} \approx \frac{F_{total}}{m} - g$

- $\ddot{\phi} \approx \frac{\tau_{roll}}{I_{xx}}$

- $\ddot{\theta} \approx \frac{\tau_{pitch}}{I_{yy}}$

- $\ddot{\psi} \approx \frac{\tau_{yaw}}{I_{zz}}$
---

## 3. Drone Configuration & Parameters

The quadrotor follows an **X-configuration** with a standard **ENU (East-North-Up)** axis convention for the world frame.

### A. Coordinate System & Axis Layout
- **X-axis (Front)**: Directed towards the front of the drone (between Motor 0 and 2).
- **Y-axis (Left)**: Directed towards the left of the drone (between Motor 1 and 2).
- **Z-axis (Up)**: Directed upwards, perpendicular to the motor plane.
- **Roll ($\phi$)**: Rotation about the X-axis.
- **Pitch ($\theta$)**: Rotation about the Y-axis.
- **Yaw ($\psi$)**: Rotation about the Z-axis.

### B. Motor Layout

![alt text](images/rotor.png)

![alt text](images/length.png)

| Motor ID | Position | Rotation | Lever Arm ($L_x, L_y$) |
| :--- | :--- | :--- | :--- |
| **0** | Front-Right | CCW (+) | $0.13, -0.22$ |
| **1** | Rear-Left | CCW (+) | $-0.13, 0.20$ |
| **2** | Front-Left | CW (-) | $0.13, 0.22$ |
| **3** | Rear-Right | CW (-) | $-0.13, -0.20$ |

### C. Physical Parameters
The following values are used for dynamic modeling and control law computation from `robot_params.xacro` and `quadrotor_base.xacro`:

| Parameter | Symbol | Value | Units |
| :--- | :--- | :--- | :--- |
| Total Mass | $m$ | 1.525 | $kg$ |
| Gravity | $g$ | 9.81 | $m/s^2$ |
| Thrust Coefficient | $k_F$ | $8.54858 \times 10^{-6}$ | $N/(rad/s)^2$ |
| Torque Coefficient | $k_M$ | 0.06 | $N·m / N$ |
| Max Motor Speed | $\omega_{max}$ | 1500 | $rad/s$ |
| Roll Inertia | $I_{xx}$ | 0.0356547 | $kg·m^2$ |
| Pitch Inertia | $I_{yy}$ | 0.0705152 | $kg·m^2$ |
| Yaw Inertia | $I_{zz}$ | 0.0990924 | $kg·m^2$ |

---

### **Moment of Inertia Calculation**

Accurate physics modeling is crucial. The moments of inertia ($I_{xx}, I_{yy}, I_{zz}$) include the base link and the four rotors using the **Parallel Axis Theorem**:
$$I_{total} = I_{base} + \sum_{i=1}^{4} (I_{rotor, i} + m_{rotor} \cdot d_i^2)$$

**Final Calculated Values:**
- $I_{xx} = 0.0356547$ kg·m²
- $I_{yy} = 0.0705152$ kg·m²
- $I_{zz} = 0.0990924$ kg·m²

---

## 4. Motor Mixing

Control efforts $[F, \tau_r, \tau_p, \tau_y]$ are converted to motor speeds $\omega_i^2$ via the allocation matrix $\Gamma$:
$$
\begin{bmatrix} F \\ \tau_r \\ \tau_p \\ \tau_y \end{bmatrix} = \begin{bmatrix} 
k_F & k_F & k_F & k_F \\
-k_F \cdot L_{y_{front}} & k_F \cdot L_{y_{rear}} & k_F \cdot L_{y_{front}} & -k_F \cdot L_{y_{rear}} \\
-k_F \cdot L_x & k_F \cdot L_x & -k_F \cdot L_x & k_F \cdot L_x \\
-k_F \cdot k_M & -k_F \cdot k_M & k_F \cdot k_M & k_F \cdot k_M 
\end{bmatrix} 
\begin{bmatrix} \omega_0^2 \\ \omega_1^2 \\ \omega_2^2 \\ \omega_3^2 \end{bmatrix}
$$
*Note: $L_x=0.13, L_{y_{front}}=0.22, L_{y_{rear}}=0.20$ based on URDF geometry.*

---

## 5. Control Strategies

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

### B. Model Predictive Control (MPC)
The MPC node solves a constrained optimization problem at every time step ($DT=0.05s$):
$$\min_{U} \sum_{k=0}^{N-1} (\mathbf{x}_k^T Q \mathbf{x}_k + \mathbf{u}_k^T R \mathbf{u}_k) + \mathbf{x}_N^T Q_N \mathbf{x}_N$$
*   **Horizon ($N=20$)**: Looks 1.0s into the future.
*   **Constraints**: Motor speeds are clipped within $[0, 1500]$ rad/s.
*   **Solver**: Uses `scipy.optimize.minimize` (L-BFGS-B) with Hessian normalization for numerical stability.

---

## 7. Test Trajectories

The `send_trajectory_2.py` utility provides several predefined patterns for testing controller performance. These trajectories generate a time-sequenced `nav_msgs/Path` (lookahead path) for the MPC and a `geometry_msgs/PoseStamped` for LQR.

### A. Available Trajectory Types

| Mode | Description | Key Parameters |
| :--- | :--- | :--- |
| **HOVER** | Holds a fixed position at a specific altitude. | $z$ |
| **LIN_1D_X** | Linear motion along the X-axis. | $x_{min}, x_{max}, speed$ |
| **LIN_1D_Z** | Linear motion along the Z-axis. | $z_{min}, z_{max}, speed$ |
| **LIN_2D** | Linear motion along the X-Z plane. | $point_{start}, point_{end}, speed$ |
| **SINE** | A sinusoidal wave along the X-axis in the X-Z plane. | $amplitude, frequency, num\_wave, speed$ |
| **LIN_3D** | A linear point-to-point path in 3D space. | $point_{start}, point_{end}, speed$ |
| **HELIX** | A 3D helical upward path. | $radius, turns, height, speed$ |

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

## 10. Results & Summary
- **LQR**: Stable hover, but drifts slightly in wind.
- **LQI**: Zero steady-state error in 4 m/s wind.
- **MPC**: Superior tracking of complex trajectories (like Helix/Fig-8) with minimal lag due to its predictive nature.

## Author
- 65340500037 Pavaris Asawakijtananont
- 65340500058 Anuwit Intet
---