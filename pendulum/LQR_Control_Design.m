
% LQR_Control_Design.m
% ---------------------
%
% Designs a Linear Quadratic Regulator (LQR) for an inverted pendulum on a cart using state-space modeling.
% The script:
%   - Defines the system dynamics using physical parameters.
%   - Constructs state-space (A, B, C, D) matrices.
%   - Computes the LQR gain matrix K using a tunable cost function (Q, R).
%   - Tunes cost function in a defined manner
%   - Simulates the closed-loop response to a step input.
%   - Plots the system response (cart position, pendulum angle and velocities).
%   - Decomposes and visualizes each state’s contribution to the total control force.
%
% Assumptions:
%   - Pendulum is modeled as a rigid rod with mass concentrated at its end.
%   - Linear damping is applied to the cart.
%   - Small-angle approximations are not used (full dynamics are linearized around upright equilibrium).
%
% Physical Parameters:
%   M = cart mass [kg]
%   m = pendulum mass [kg]
%   l = pendulum length to center of mass [m]
%   b = damping coefficient [Ns/m]
%   g = gravity [m/s^2]
%   x = cart position
%   x_dot = cart velocity
%   phi = pendulum angle from upright position
%   phi_dot = angluar velocity
%
% Toolbox Requirements:
%   - Control System Toolbox: for `ss`, `eig`, `ctrb`, `lqr`, `lsim`
%
% References:
% [1] https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
%
% Author: Finn Cracknell
% Date: 30/06/2025

clear all; close all; clc;

% define system parameters
M = 10;
m = 0.2;
b = 0.1;
g = 9.81;
l = 0.15;
I = m*l^2;

% define denominator for the A and B matrices
p = I*(M+m)+M*m*l^2;

% define system dynamics matrix A (4 x 4)
A = [0      1              0           0;
     0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
     0      0              0           1;
     0 -(m*l*b)/p       m*g*l*(M+m)/p  0];

% define input matrix B (4 x 1)
B = [     0;
     (I+m*l^2)/p;
          0;
        m*l/p];

% define output matrix C (2 x 4)
C = [1 0 0 0; 
     0 0 1 0];

% define feedthrough matrix D (2 x 1)
D = [0;
     0];

% define state, input, and output names
states = {'x' 'x_dot' 'phi' 'phi_dot'};
inputs = {'u'};
outputs = {'x'; 'phi'};

% create state-space system for the open-loop plant
sys_ss = ss(A,B,C,D,'statename',states,'inputname',inputs,'outputname',outputs);

% get poles and controlability
poles = eig(A);
co = ctrb(sys_ss);

% check controllability is correctly equal to number of states)
controllability = rank(co);

% define cost function matrices (Q and R) for LQR design
Q = C'*C;
Q(1,1) = 5000;   % weight on x (cart position)
Q(2,2) = 100;    % weight on x_dot (cart velocity)
Q(3,3) = 200;    % weight on phi (pendulum angle)
Q(4,4) = 50;     % weight on phi_dot (anglular velocity)

R = 1;           % weight on control effect

% compute LQR gains
K = lqr(A,B,Q,R)

% define closed-loop system with state feedback
Ac = [(A-B*K)];
Bc = [B];
Cc = [C];
Dc = [D];

% redefine system for closed-loop
sys_cl = ss(Ac,Bc,Cc,Dc,'statename',states,'inputname',inputs,'outputname',outputs);

% simulate closed-loop response to step reference
figure;
t = 0:0.01:5;
r = 0.2 * ones(size(t));
[y, t, x] = lsim(sys_cl, r, t);

% plot x and theta on the first figure using plotyy
[AX, H1, H2] = plotyy(t, y(:,1), t, y(:,2), 'plot');
set(get(AX(1), 'Ylabel'), 'String', 'Cart Position x (m)');
set(get(AX(2), 'Ylabel'), 'String', 'Pendulum Angle \phi (rad)');
xlabel('Time (s)');
title('Step Response with LQR Control (Position and Angle)');

% overlay additional plots for x_dot and phi_dot
hold(AX(1), 'on'); 
hold(AX(2), 'on'); 

% extract state variables
x_pos   = x(:,1);
x_dot   = x(:,2);
phi     = x(:,3);
phi_dot = x(:,4);

% plot x_dot on left Y-axis
plot(AX(1), t, x_dot, 'r--', 'DisplayName', 'Cart Velocity $\dot{x}$');

% plot phi_dot on right Y-axis
plot(AX(2), t, phi_dot, 'm--', 'DisplayName', 'Pendulum Angular Velocity $\dot{\phi}$');
ylim(AX(2), [min(phi_dot), max(phi_dot)])

% simulate closed-loop system response
[y,t,x] = lsim(sys_cl,r,t);

% compute control input over time
u = -x * K';

% decompose x into individual state components
x1 = x(:,1); % cart position
x2 = x(:,2); % cart velocity
x3 = x(:,3); % pendulum angle
x4 = x(:,4); % pendulum angular velocity

% decompose K
k1 = K(1);
k2 = K(2);
k3 = K(3);
k4 = K(4);

% individual contributions to the control signal
u1 = -k1 * x1;
u2 = -k2 * x2;
u3 = -k3 * x3;
u4 = -k4 * x4;

% plot the full control signal and each component
figure;
plot(t, u, 'k-', 'LineWidth', 2); hold on;
plot(t, u1, 'r--', 'LineWidth', 1.5);
plot(t, u2, 'b--', 'LineWidth', 1.5);
plot(t, u3, 'g--', 'LineWidth', 1.5);
plot(t, u4, 'm--', 'LineWidth', 1.5);
legend('Total u(t)', '-k_1 x', '-k_2 x\_dot', '-k_3 \theta', '-k_4 \theta\_dot');
xlabel('Time (s)');
ylabel('Control Input (N)');
title('LQR Control Signal and State Contributions');
grid on;