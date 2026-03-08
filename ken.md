## Optimal Control for MPC
### 1. Problem Setup

Consider the discrete-time linear system

$$
x_{k+1} = A x_k + B u_k
$$

where

- $x_k \in \mathbb{R}^n$ is the state
- $u_k \in \mathbb{R}^m$ is the control input
- $A \in \mathbb{R}^{n \times n}$
- $B \in \mathbb{R}^{n \times m}$

In finite-horizon Model Predictive Control (MPC), at time $k$ we solve an optimization problem over a prediction horizon of length $N$.

We define

$$
x_0 := x(k)
$$


to minimize a quadratic cost while satisfying system dynamics and constraints.

---

### 2. Quadratic Cost Function

The standard finite-horizon quadratic cost is

$$
J = \sum_{i=0}^{N-1} \left( x_i^T Q x_i + u_i^T R u_i \right) + x_N^T P x_N
$$

where

- $Q \succeq 0$ is the stage state cost
- $R \succ 0$ is the stage input cost
- $P \succeq 0$ is the terminal state cost

This cost penalizes:

- large state deviations through $Q$
- large control effort through $R$
- terminal error through $P$

---

### 3. Finite-Horizon Prediction Model

Using the dynamics recursively,

$$
x_1 = A x_0 + B u_0
$$

$$
x_2 = A x_1 + B u_1 = A^2 x_0 + A B u_0 + B u_1
$$

$$
x_3 = A x_2 + B u_2 = A^3 x_0 + A^2 B u_0 + A B u_1 + B u_2
$$

In general,

$$
x_i = A^i x_0 + \sum_{j=0}^{i-1} A^{i-1-j} B u_j
$$

Now stack all predicted states into one vector:

$$
X =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_N
\end{bmatrix}
\in \mathbb{R}^{nN}
$$

as the current measured state, and optimize the future control sequence

$$
U =
\begin{bmatrix}
u_0 \\
u_1 \\
\vdots \\
u_{N-1}
\end{bmatrix}
\in \mathbb{R}^{mN}
$$

Then the stacked prediction model can be written as

$$
X = \mathcal{A} x_0 + \mathcal{B} U
$$

with

$$
\mathcal{A} =
\begin{bmatrix}
A \\
A^2 \\
\vdots \\
A^N
\end{bmatrix}
\in \mathbb{R}^{nN \times n}
$$

and

$$
\mathcal{B} =
\begin{bmatrix}
B & 0 & 0 & \cdots & 0 \\
AB & B & 0 & \cdots & 0 \\
A^2B & AB & B & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
A^{N-1}B & A^{N-2}B & A^{N-3}B & \cdots & B
\end{bmatrix}
\in \mathbb{R}^{nN \times mN}
$$

---

### 4. Rearranging the Cost into Matrix Form

To write the cost compactly, define the stacked weighting matrices

$$
\bar{Q} =
\begin{bmatrix}
Q      & 0      & \cdots & 0 \\
0      & Q      & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0      & 0      & \cdots & P
\end{bmatrix}
\in \mathbb{R}^{nN \times nN}
$$

and

$$
\bar{R} =
\begin{bmatrix}
R      & 0      & \cdots & 0 \\
0      & R      & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0      & 0      & \cdots & R
\end{bmatrix}
\in \mathbb{R}^{mN \times mN}
$$

Then the cost becomes

$$
J = X^T \bar{Q} X + U^T \bar{R} U
$$

Substitute the prediction model $X = \mathcal{A}x_0 + \mathcal{B}U$:

$$
J = (\mathcal{A}x_0 + \mathcal{B}U)^T \bar{Q} (\mathcal{A}x_0 + \mathcal{B}U) + U^T \bar{R}U
$$

Expand:

$$
J = x_0^T \mathcal{A}^T \bar{Q} \mathcal{A} x_0
+ 2 x_0^T \mathcal{A}^T \bar{Q} \mathcal{B} U
+ U^T \mathcal{B}^T \bar{Q} \mathcal{B} U
+ U^T \bar{R} U
$$

Group the terms in $U$:

$$
J = U^T H U + 2 F^T U + c
$$

where

$$
H = \mathcal{B}^T \bar{Q} \mathcal{B} + \bar{R}
$$

$$
F = \mathcal{B}^T \bar{Q} \mathcal{A} x_0
$$

$$
c = x_0^T \mathcal{A}^T \bar{Q} \mathcal{A} x_0
$$

Since $c$ does not depend on $U$, it can be ignored during optimization. Therefore the finite-horizon MPC problem becomes

$$
\min_U \quad U^T H U + 2 F^T U
$$

or equivalently, in standard quadratic programming form,

$$
\min_U \quad \frac{1}{2} U^T (2H) U + (2F)^T U
$$

---

### 5. Error-State Formulation and Minimization
Current quadratic cost function is minimizing state and control input, but we want to minimize error along the trajectory

$$
J = (X-X_{ref})^T\bar{Q}(X-X_{ref}) + U^T\bar{R}U
$$

Substitute the prediction model

$$
X = \mathcal{A}x_0 + \mathcal{B}U
$$

Then

$$
J =
(\mathcal{A}x_0 + \mathcal{B}U - X_{ref})^T
\bar{Q}
(\mathcal{A}x_0 + \mathcal{B}U - X_{ref})
+
U^T\bar{R}U
$$

Expand the quadratic term

$$
J =
(\mathcal{A}x_0-X_{ref})^T\bar{Q}(\mathcal{A}x_0-X_{ref})
+
2U^T\mathcal{B}^T\bar{Q}(\mathcal{A}x_0-X_{ref})
+
U^T\mathcal{B}^T\bar{Q}\mathcal{B}U
+
U^T\bar{R}U
$$

Group terms in \(U\)

$$
J =
U^T(\mathcal{B}^T\bar{Q}\mathcal{B}+\bar{R})U
+
2U^T\mathcal{B}^T\bar{Q}(\mathcal{A}x_0-X_{ref})
+
(\mathcal{A}x_0-X_{ref})^T\bar{Q}(\mathcal{A}x_0-X_{ref})
$$

Let

$$
H = \mathcal{B}^T\bar{Q}\mathcal{B}+\bar{R}
$$

$$
F = \mathcal{B}^T\bar{Q}(\mathcal{A}x_0-X_{ref})
$$

Then

$$
J = U^THU + 2U^TF + c
$$

---

#### Partial derivative w.r.t \(U\)

$$
\frac{\partial J}{\partial U}
=
2HU + 2F
$$

Set gradient to zero

$$
2HU + 2F = 0
$$

$$
HU + F = 0
$$

Optimal control sequence

$$
U^* = -H^{-1}F
$$

Substitute back

$$
U^* =
-
(\mathcal{B}^T\bar{Q}\mathcal{B}+\bar{R})^{-1}
\mathcal{B}^T\bar{Q}(\mathcal{A}x_0-X_{ref})
$$

$U^*$ is our optimal control input

> Special Thanks, reference : https://www.youtube.com/watch?v=6GSHAoLMsXs&list=PLg6FTHy3zJjzJ8Ddui6ZwQpdMZeoqG1ei

## Model Predictive Control Concept
MPC works by predicting future system behavior using a model, optimizing a sequence of control inputs over a finite horizon, and applying only the first input before repeating the process. Then update the model, using the state in current time step as the initial guess for the open loop prediction. The two key ideas involved are open-loop prediction and the distinction between prediction horizon and control horizon.

<p align="center">
  <img src="./images/mpc.png" width="75%"/>
</p>

**Prediction Horizon**
- The prediction horizon ​$N_{p}$  efines the number of future time steps over which the system states are predicted using the model
- The model simulates the future system behavior over ​$N_{p}$ steps assuming a sequence of future control inputs.
$$
x_{k+1}, x_{k+2} , x_{k+3} , \dots, x_{k+N_{p}}
$$
The optimizer minimizes a cost function defined over these predicted states.

**Control Horizon**
- The control horizon ​$N_{c}$ defines the number of independent control inputs optimized by the controller.
$$
N_{c} \le N_{p}
$$
- After ​$N_{c}$ , the control input is usually held constant for the remaining prediction horizon:
$$
u_{k+i} = u_{k+N_{c}-1}, i \ge N_{c}
$$