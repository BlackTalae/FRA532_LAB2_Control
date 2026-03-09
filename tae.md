# Linear Quadratic Regulator (LQR) Control for Quadrotor

This document details the design and development of the **LQR (Linear Quadratic Regulator)** controller for a quadrotor, based on the implementation in `LQR.py`.

## 1. Introduction
LQR is an optimal control method for Linear Time-Invariant (LTI) systems. It aims to find an optimal gain matrix $K$ that minimizes a defined cost function. In this project, LQR is used to maintain the quadrotor in a stable hover or to track a target position.

## 2. Dynamic Modeling & Linearization
The quadrotor is a nonlinear system, but for LQR control, we linearize it around the hover equilibrium.

### Full Nonlinear Dynamics (Simplified)
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

## 3. State-Space Representation
The system is expressed as: $\mathbf{\dot{x}} = A\mathbf{x} + B\mathbf{u}$

### State Vector ($\mathbf{x}$) - 12x1
$\mathbf{x} = [x, y, z, \phi, \theta, \psi, \dot{x}, \dot{y}, \dot{z}, \dot{\phi}, \dot{\theta}, \dot{\psi}]^T$

### Input Vector ($\mathbf{u}$) - 4x1
$\mathbf{u} = [F_{total}, \tau_{roll}, \tau_{pitch}, \tau_{yaw}]^T$

### Full Expanded State-Space Equation

The linearized quadrotor system in full $\mathbf{\dot{x}} = A\mathbf{x} + B\mathbf{u}$ form:

$$
\begin{bmatrix} 
\dot{x} \\ \dot{y} \\ \dot{z} \\ \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \\ \ddot{x} \\ \ddot{y} \\ \ddot{z} \\ \ddot{\phi} \\ \ddot{\theta} \\ \ddot{\psi} 
\end{bmatrix} = 
\begin{bmatrix} 
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & g & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & -g & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix} 
\begin{bmatrix} 
x \\ y \\ z \\ \phi \\ \theta \\ \psi \\ \dot{x} \\ \dot{y} \\ \dot{z} \\ \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} 
\end{bmatrix} + 
\begin{bmatrix} 
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
1/m & 0 & 0 & 0 \\
0 & 1/I_{xx} & 0 & 0 \\
0 & 0 & 1/I_{yy} & 0 \\
0 & 0 & 0 & 1/I_{zz}
\end{bmatrix} 
\begin{bmatrix} 
dF \\ \tau_{roll} \\ \tau_{pitch} \\ \tau_{yaw} 
\end{bmatrix}
$$

- $dF = F_{total} - mg$ is the thrust adjustment relative to hover.
- $\tau_{roll}, \tau_{pitch}, \tau_{yaw}$ are the control torques.

## 4. LQR Implementation Procedure

The following steps describe the process of implementing the LQR controller as seen in `LQR.py`:

### Step 1: Define Physical Parameters
Extract the quadrotor's physical constants from `robot_params.xacro`, including mass ($m$), gravity ($g$), and moments of inertia ($I_{xx}, I_{yy}, I_{zz}$).

### Step 2: Build the Linearized Model (A & B Matrices)
Construct the system matrix $A$ and input matrix $B$ by linearizing the quadrotor dynamics around the hover equilibrium point ($\phi, \theta \approx 0$).

### Step 3: Define Cost Matrices (Q & R)
- Select the **Q matrix** to penalize state errors. Higher values for specific states (like $z$ or $\psi$) make the controller more aggressive in correcting those errors.
- Select the **R matrix** to penalize control effort. Lower values allow the controller to use more power for faster response.

### Step 4: Solve the Algebraic Riccati Equation (ARE)
Solve the continuous-time Algebraic Riccati Equation to find the unique positive-definite matrix $P$:
$$A^T P + PA - PBR^{-1}B^T P + Q = 0$$
This is performed using `scipy.linalg.solve_continuous_are`.

### Step 5: Compute the Optimal Gain Matrix (K)
The optimal gain matrix $K$ is calculated as:
$$K = R^{-1} B^T P$$

### Step 6: Apply the Control Law
Calculate the control correction $\mathbf{u}_{corr}$ based on the state error:
$$\mathbf{u} = \mathbf{u}_{hover} + \mathbf{u}_{corr}$$
$$\mathbf{u}_{corr} = K \cdot (\mathbf{x}_{ref} - \mathbf{x}_{current})$$
Where $\mathbf{u}_{hover} = [mg, 0, 0, 0]^T$.

### Step 7: Motor Mixing & Actuation
Convert the control inputs $\mathbf{u} = [F, \tau_r, \tau_p, \tau_y]^T$ into individual motor squared velocities ($\omega_i^2$) using the inverse allocation matrix $\Gamma^{-1}$, then take the square root to get $\omega_i$.

---

## 5. Weight Design (Q & R Matrices)
LQR minimizes the cost function: $J = \int_{0}^{\infty} (\mathbf{x}^T Q \mathbf{x} + \mathbf{u}^T R \mathbf{u}) dt$

### State Weight Matrix (Q) - 12x12 Diagonal
Values from `LQR.py`:
- $Q = \text{diag}(120, 120, 900, 10, 10, 1500, 10, 10, 50, 1, 1, 10)$
- **Key Focus:** High penalty on $z$ error (900) and Yaw error (1500) for vertical and directional stability.

### Input Weight Matrix (R) - 4x4 Diagonal
Values from `LQR.py`:
- $R = \text{diag}(1, 1, 1, 0.001)$
- **Key Focus:** Very low penalty on Yaw torque ($R_{yaw}=0.001$), allowing the controller to use high control effort to fix yaw quickly.

## 7. Motor Mixing (Allocation Matrix)

The control inputs $\mathbf{u} = [F_{total}, \tau_{roll}, \tau_{pitch}, \tau_{yaw}]^T$ are allocated to the four motors using the relationship:
$$\mathbf{u} = \Gamma \cdot \begin{bmatrix} \omega_0^2 \\ \omega_1^2 \\ \omega_2^2 \\ \omega_3^2 \end{bmatrix}$$

### Allocation Matrix ($\Gamma$)
Based on the X-configuration and motor directions (0:CCW, 1:CCW, 2:CW, 3:CW) in `LQR.py`:

$$
\begin{bmatrix} 
F_{total} \\ \tau_{roll} \\ \tau_{pitch} \\ \tau_{yaw} 
\end{bmatrix} = 
\begin{bmatrix} 
k_F & k_F & k_F & k_F \\
-k_F \cdot L & k_F \cdot L & k_F \cdot L & -k_F \cdot L \\
-k_F \cdot L & k_F \cdot L & -k_F \cdot L & k_F \cdot L \\
-k_M & -k_M & k_M & k_M 
\end{bmatrix} 
\begin{bmatrix} 
\omega_0^2 \\ \omega_1^2 \\ \omega_2^2 \\ \omega_3^2 
\end{bmatrix}
$$

### Solving for Motor Speeds
To find the required motor speeds $\omega_i$, we use the inverse matrix $\Gamma^{-1}$:
$$\begin{bmatrix} \omega_0^2 \\ \omega_1^2 \\ \omega_2^2 \\ \omega_3^2 \end{bmatrix} = \Gamma^{-1} \cdot \begin{bmatrix} F_{total} \\ \tau_{roll} \\ \tau_{pitch} \\ \tau_{yaw} \end{bmatrix}$$

Individual motor equations (simplified):
1. $\omega_0 = \sqrt{\text{max}(0, \text{sol}_0)}$ (Front-Right)
2. $\omega_1 = \sqrt{\text{max}(0, \text{sol}_1)}$ (Rear-Left)
3. $\omega_2 = \sqrt{\text{max}(0, \text{sol}_2)}$ (Front-Left)
4. $\omega_3 = \sqrt{\text{max}(0, \text{sol}_3)}$ (Rear-Right)

Where $\text{sol}_i$ are the elements of the resulting vector from $\Gamma^{-1} \mathbf{u}$ and are clamped by $\omega_{max}$.

## 8. Moment of Inertia Calculation

The moments of inertia ($I_{xx}, I_{yy}, I_{zz}$) in `quadrotor_base.xacro` are essential for the LQR angular acceleration terms. For a quadrotor approximated as a solid rectangular prism (box) with mass $m$, width $w$, depth $d$, and height $h$:

### Formulas for a Solid Box
If the quadrotor's main body is modeled as a box with dimensions from the collision tag ($w=0.47, d=0.47, h=0.11$):

- **Roll Inertia ($I_{xx}$)**: $I_{xx} = \frac{1}{12} m (d^2 + h^2)$
- **Pitch Inertia ($I_{yy}$)**: $I_{yy} = \frac{1}{12} m (w^2 + h^2)$
- **Yaw Inertia ($I_{zz}$)**: $I_{zz} = \frac{1}{12} m (w^2 + d^2)$

### Specific Values from URDF
In `quadrotor_base.xacro`, the `base_link` inertial parameters are defined as:
- $m = 1.5$ kg
- $I_{xx} = 0.0347563$
- $I_{yy} = 0.07$
- $I_{zz} = 0.0977$

## 9. Composite Inertia Calculation (Multi-part Robot)

When a robot consists of multiple parts (links), the total inertia must be calculated relative to the global Center of Mass (CoM).

### Step 1: Total Mass and Global CoM
The total mass ($M$) and global CoM ($\mathbf{R}_{cm}$) are found by summing the masses ($m_i$) and positions ($\mathbf{r}_i$) of each link:
- $M = \sum m_i$
- $\mathbf{R}_{cm} = \frac{1}{M} \sum m_i \mathbf{r}_i$

### Step 2: Parallel Axis Theorem (Steiner's Theorem)
To find the inertia of each part relative to the global CoM, we use the **Parallel Axis Theorem**:
$$I_{total} = \sum_{i} \left( I_{i, cm} + m_i \cdot d_i^2 \right)$$
Where:
- $I_{i, cm}$ is the inertia of part $i$ about its own CoM.
- $d_i$ is the distance from the part's CoM to the global CoM along the axes.

### Application to `quadrotor_base.xacro`
The quadrotor has a `base_link` and 4 `rotor` links. To find the total $I_{zz}$:
1. **Rotor Positions**: Rotors are located at offsets ($x_i, y_i$).
2. **Shift Inertia**: For each rotor, the contribution to $I_{zz}$ is:
   $I_{zz, rotor, total} = I_{zz, rotor} + m_{rotor} (x_i^2 + y_i^2)$
3. **Summation**:
   $I_{zz, total} = I_{zz, base\_link} + \sum_{i=0}^{3} \left( I_{zz, rotor, i} + m_{rotor, i} (x_i^2 + y_i^2) \right)$

### Worked-out Example: Calculating Total $I_{zz}$
Using data from `quadrotor_base.xacro`:
- **Base Link**: $m_b = 1.5$ kg, $I_{zz,b} = 0.0977$
- **Rotors (4)**: $m_r = 0.005$ kg, $I_{zz,r} = 4.26 \times 10^{-5}$
- **Offsets (x, y)**:
  - Rotor 0: $(0.13, -0.22) \implies d^2 = 0.0653$
  - Rotor 1: $(-0.13, 0.20) \implies d^2 = 0.0569$
  - Rotor 2: $(0.13, 0.22) \implies d^2 = 0.0653$
  - Rotor 3: $(-0.13, -0.20) \implies d^2 = 0.0569$

**Calculations:**
1. **Rotor 0 contribution**: $4.26 \times 10^{-5} + 0.005(0.13^2 + 0.22^2) = 0.0003691$
2. **Rotor 1 contribution**: $4.26 \times 10^{-5} + 0.005(0.13^2 + 0.20^2) = 0.0003271$
3. **Total Rotor $I_{zz}$**: $2 \times (0.0003691 + 0.0003271) = 0.0013924$
4. **Final Robot $I_{zz}$**: $0.0977 + 0.0013924 = \mathbf{0.0990924}$ kg·m²

### Worked-out Example: Calculating Total $I_{zz}$
- **Base Link**: $m_b = 1.5$ kg, $I_{zz,b} = 0.0977$
- **Rotors (mass $m_r=0.005$)**: Total rotor $I_{zz}$ contribution: $\mathbf{0.0013924}$ 
- **Result $I_{zz, tot}$**: $0.0977 + 0.0013924 = \mathbf{0.0990924}$ kg·m²

### Worked-out Example: Calculating Total $I_{xx}$
Calculated using $I_{xx, tot} = I_{xx, b} + \sum [I_{xx, r} + m_r(y_i^2 + z_i^2)]$
- **Offsets**: $z=0.023$ for all. $y = [\pm 0.22, \pm 0.20]$.
- **Base Link**: $I_{xx,b} = 0.0347563$
- **Rotor Contribution (avg)**: $9.75 \times 10^{-7} + 0.005(0.21^2 + 0.023^2) \approx 0.0002246$
- **Result $I_{xx, tot}$**: $0.0347563 + 4 \times 0.0002246 \approx \mathbf{0.0356547}$ kg·m²

### Worked-out Example: Calculating Total $I_{yy}$
Calculated using $I_{yy, tot} = I_{yy, b} + \sum [I_{yy, r} + m_r(x_i^2 + z_i^2)]$
- **Offsets**: $z=0.023, x = \pm 0.13$ for all.
- **Base Link**: $I_{yy,b} = 0.07$
- **Rotor Contribution**: $4.17 \times 10^{-5} + 0.005(0.13^2 + 0.023^2) = 0.0001288$
- **Result $I_{yy, tot}$**: $0.07 + 4 \times 0.0001288 = \mathbf{0.0705152}$ kg·m²

This systematic summation ensures the LQR state-space model in `LQR.py` accurately represents the inertia of the entire assembled drone.
