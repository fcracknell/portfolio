"""
Swing-Up_and_LQR_Control_Clean.py
---------------------

Simulates the swing-up and LQR stabilization of an inverted pendulum on a cart using Pygame and Pymunk.

The pendulum is first swung up using an energy-based controller, then stabilized in the upright position using 
linear quadratic regulation (LQR). Real-time visualization, energy plots, and a screen recording GIF option are included.

Author: Finn Cracknell
Date: 30/06/2025

References:
[1] Energy-Based Swing-Up: https://coecsl.ece.illinois.edu/se420/ast_fur96.pdf
[2] Energy-Based Swing-Up: https://youtu.be/RhF2NMCYoiw?si=9hdegrJlNfaIlK40 
[3] LQR Control: https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
[4] LQR Control: https://www.youtube.com/watch?v=96hHEWN1sIM&ab_channel=JuliaHub 

Dependencies:
    - numpy
    - pygame
    - pymunk
    - matplotlib
    - pygame-screen-recorder (optional: only for gif creation)
"""

import pymunk
import pymunk.pygame_util
import pygame
from pymunk.vec2d import Vec2d

import matplotlib.pyplot as plt

import math
import numpy as np

import time

from pygame_screen_recorder import pygame_screen_recorder as pgr

pygame.init()

CONFIG = {
    'pixels_per_meter': 1000,
    'screen_length_px': 1300,
    'screen_height_px': 800,
    'pend_length_m': 0.15,
    'pend_mass_kg': 0.2,
    'friction_coefficient': 0.1, # cart-linear rail friction coefficient
    'cart_mass_kg': 10, # must be high compared to pendulum mass
    'gravity_N': 9.81,
    'time_step_s': 0.005,
    'max_acceleration_g': 1.5
}

# define commonly used variables
pixels_per_meter = CONFIG["pixels_per_meter"]
screen_length_px = CONFIG["screen_length_px"]
screen_height_px = CONFIG["screen_height_px"]
pend_length_px = CONFIG["pend_length_m"] * pixels_per_meter
pend_mass_kg = CONFIG["pend_mass_kg"]
cart_mass_kg = CONFIG["cart_mass_kg"]
gravity_px = CONFIG["gravity_N"] * pixels_per_meter

def get_state_variables(body_c, body_p, reference_matrix, x_prev, theta_prev, dt):
    """
    Computes the key state variables of the inverted pendulum-cart system.

    This function calculates the cart position and velocity, pendulum angle and angular velocity,
    and other derived quantities needed for control. It transforms the pendulum position into
    a coordinate frame relative to the cart and unwraps the angle to ensure continuity over time.

    Some of the outputs are redundant, this is raised as a future improvement.

    Parameters:
        body_c (pymunk.Body): The Pymunk body representing the cart.
        body_p (pymunk.Body): The Pymunk body representing the pendulum.
        reference_matrix (np.ndarray): Desired state reference for LQR control [x, x_dot, theta, theta_dot].
        x_prev (float): Previous time step cart position in meters.
        theta_prev (float): Previous time step pendulum angle in radians (for unwrapping).
        dt (float): Time step size (s) used for derivative estimation.

    Returns:
        tuple: (
            x (float): Cart position (m) in frame centered at bottom left of screen,
            x_dot (float): Cart velocity (m/s),
            theta (float): Pendulum angle (rad), positive is upright,
            theta_dot (float): Pendulum angular velocity (rad/s),
            theta_control (float): Negated angle used for LQR control input,
            theta_dot_control (float): Negated angular velocity for LQR,
            x_error_m (float): Cart position error relative to LQR reference x position (m),
            pend_x0 (float): Pendulum x-position in screen coordinates,
            theta_raw (float): Raw atan2 angle without unwrapping for control activation,
            pend_xc (float): Pendulum x-position relative to cart (screen units) for control activation
        )
    """
        
    # define coordinate frame 0 with origin at bottom left of screen
    #   --> positive x direction to the right
    #   --> positive y direction upwards

    # define cart position in coordinate frame 0
    cart_x0 = body_c.position[0]
    cart_y0 = screen_height_px - body_c.position[1]

    # define pendulumn center of mass position in coordinate frame 0
    pend_x0 = body_p.position[0]
    pend_y0 = screen_height_px - body_p.position[1]

    # define cart x-position relative to desired steady x-position state (in metres)
    x_error_m = (cart_x0/pixels_per_meter - reference_matrix[0][0])

    # define coordinate frame c with origin at cart position (as suggested in paper)
    #   --> positive x direction to the right
    #   --> positive y direction upwards

    # define pendulumn center of mass position in coordinate frame c
    pend_xc = pend_x0 - cart_x0
    pend_yc = pend_y0 - cart_y0

    # cart position in coordinate frame 0
    x = (cart_x0) / pixels_per_meter

    # cart velocity
    x_dot = ((x - x_prev)/dt)

    # get theta values
    theta_raw = math.atan2(pend_xc, pend_yc)
    theta_control = -math.atan2(pend_xc, pend_yc)
    theta_unwrapped = np.unwrap([theta_prev, theta_raw])[1]
    theta = theta_unwrapped
    
    # get angular velocity
    theta_dot = (theta - theta_prev)/dt
    theta_dot_control = -theta_dot

    return (x, x_dot, theta, theta_dot, theta_control, theta_dot_control, x_error_m, pend_x0, theta_raw, pend_xc)

def get_energy(angle, angular_vel, J_p, m_p, g, l):
    """
    Returns total mechanical energy of pendulum based on Equation (2) of [1].
    
    Parameters:
        angle (float): Pendulum angle in radians.
        angular_vel (float): Angular velocity in rad/s.
        
    Returns:
        float: Total mechanical energy (kinetic + gravitational) in SI units relative to upright position.
    """
    return 0.5*(angular_vel**2)*J_p + m_p*g*l*(1+np.cos(angle))

def get_friction_force(body_c, pixels_per_meter, friction_coefficient):
    """
    Computes the horizontal friction force acting on the cart due to velocity-based damping.

    Parameters:
        body_c (pymunk.Body): The Pymunk body representing the cart.
        pixels_per_meter (float): Conversion factor from meters to pixels.
        friction_coefficient (float): Linear damping coefficient (Ns/m).

    Returns:
        float: Friction force in pixel units to be applied to the cart body.
    """

    # friction force on cart
    velocity = body_c.velocity[0] / pixels_per_meter  # convert to m/s
    friction_force = -friction_coefficient * velocity
    friction_force_px = friction_force * pixels_per_meter

    return friction_force_px

def get_swing_up_force(theta, theta_dot, J_p, m_p, g, l, umax, m_c, pixels_per_meter):
    """
    Calculates the swing-up input force based on the energy of the pendulum.

    This function implements an energy-based controller to bring the pendulum to its upright
    position by injecting energy depending on the current state. The method
    is adapted from Equation (4) of [1], with a gain of 100 added as suggested by [2].
     
    This method is based on the difference between the current and target energy, scaled
    and clipped to a maximum allowable force constrained by a maximum allowable acceleration.

    Parameters:
        theta (float): Current pendulum angle (rad), with 0 at upright.
        theta_dot (float): Angular velocity of the pendulum (rad/s).
        J_p (float): Moment of inertia of the pendulum about the pivot.
        m_p (float): Mass of the pendulum (kg).
        g (float): Gravitational acceleration (m/s^2).
        l (float): Length from pivot to center of mass of the pendulum (m).
        umax (float): Maximum allowable force (in pixels) that can be applied to the cart.
        m_c (float): Mass of the cart (kg).
        pixels_per_meter (float): Conversion factor to translate meters to pixels and vice versa.

    Returns:
        float: The computed swing-up control force in pixels (clipped to ±umax).
    """

    # get current energy and reference energy
    E = get_energy(theta, theta_dot, J_p, m_p, g, l)
    E0 = get_energy(0, 0, J_p, m_p, g, l)

    # calculate input acceleration using gain from [2] and Equation (4) of [1]
    uE = 100*(E - E0)*np.sign(theta_dot*np.cos(theta))

    # edge case: if the product of theta_dot * cos(theta) is zero, treat the sign as positive
    if np.sign(theta_dot*np.cos(theta)) == 0:
            uE = 100*(E - E0)

    # convert acceleration to force based on cart mass and screen units
    input_force = m_c * uE * pixels_per_meter

    # make sure input force doesn't exceed maximum allowable acceleration
    input_force_px = sorted((-umax, input_force, umax))[1]

    return input_force_px

def get_LQR_control_force(x, x_dot, theta_control, theta_dot_control, reference_matrix, K_matrix, umax, pixels_per_meter):
    """
    Computes the LQR control force to stabilize the inverted pendulum at the desired steady state.

    This function uses a linear state-feedback controller (LQR) to calculate the control force
    based on the error from the desired reference state. The output force is clipped to respect
    the system’s maximum allowable input.

    Parameters:
        x (float): Current cart position (m).
        x_dot (float): Cart velocity (m/s).
        theta_control (float): Pendulum angle used for control (rad).
        theta_dot_control (float): Angular velocity used for control (rad/s).
        reference_matrix (np.ndarray): Desired system state as a 4×1 column vector.
        K_matrix (np.ndarray): State feedback gain matrix (1×4) computed from LQR design in MatLab.
        umax (float): Maximum allowable acceleration input (m/s^2).
        pixels_per_meter (float): Conversion factor to translate meters to pixels and vice versa.

    Returns:
        tuple:
            force_px (float): Control force in pixel units to apply to the cart.
            x_matrix (np.ndarray): Current 4×1 system state vector used in control calculation.
    """

    # get current statte variables
    x_matrix = np.array([
                        [x],
                        [x_dot],
                        [theta_control],
                        [theta_dot_control]
                        ])
    
    # compute error from steady state
    error = x_matrix - reference_matrix

    # get input force based on error and gain values, limiting by defined maximum acceleration
    control_force_ms2 = np.clip(-K_matrix @ error, -umax, umax)

    return(control_force_ms2[0][0] * pixels_per_meter, x_matrix)

def draw_scene(screen, trail, groove, space, body_p, body_c, c, shape_c, shape_p):
    """
    Renders the current state of the inverted pendulum-cart system to the Pygame screen.

    This function draws the full scene, including the cart, pendulum, linear rail, pivot constraint,
    and the pendulum trail showing the recent motion path.

    Parameters:
        screen (pygame.Surface): The Pygame surface to draw onto.
        trail (list of tuple): List of (x, y) positions for the pendulum bob trail.
        groove (pymunk.GrooveJoint): The constraint representing the horizontal track.
        space (pymunk.Space): The physics simulation space (used for coordinate transforms).
        body_p (pymunk.Body): The pendulum's physics body.
        body_c (pymunk.Body): The cart's physics body.
        c (pymunk.Constraint): The pin joint constraint between pendulum and cart.
        shape_c (pymunk.Segment): The shape representing the cart.
        shape_p (pymunk.Circle): The shape representing the pendulum bob.

    Returns:
        None
    """

    # drawing white background
    screen.fill('WHITE')

    # drawing pendulum bob trail
    for pos in trail:
        pygame.draw.circle(screen, (176,167,167), pos, 2)

    # draw horizontal rail constarint
    groove_start = groove.groove_a
    groove_end = groove.groove_b
    groove_start_world = space.static_body.local_to_world(groove_start)
    groove_end_world = space.static_body.local_to_world(groove_end)
    pygame.draw.line(screen, (128, 128, 128), groove_start_world, groove_end_world, 3)

    # draw pendulum shaft
    anchor1 = body_p.local_to_world(c.anchor_a)
    anchor2 = body_c.local_to_world(c.anchor_b)
    pygame.draw.line(screen, (128, 128, 128), anchor1, anchor2, 2)

    # draw cart
    cart_start = body_c.local_to_world(shape_c.a)
    cart_end = body_c.local_to_world(shape_c.b)
    pygame.draw.line(screen, (0, 0, 0), cart_start, cart_end, 8)

    # draw pendulum and pendulum outline
    pend_pos = int(body_p.position[0]), int(body_p.position[1])
    pygame.draw.circle(screen, (0, 0, 0), pend_pos, int(shape_p.radius+1))
    pygame.draw.circle(screen, (200, 0, 0), pend_pos, int(shape_p.radius))

def draw_data(screen, font, phase, time_list, x_error_m, theta):
    """
    Renders simulation data (phase, time, cart position, pendulum angle) on the screen.

    Parameters:
        screen (pygame.Surface): The Pygame surface to render the text on.
        font (pygame.font.Font): Font object used for rendering text.
        phase (str): Current phase of the simulation (e.g., "SWING-UP", "LQR CONTROL", etc.).
        time_list (list of float): List of time values; the last entry is used as current time.
        x_error_m (float): Cart position error from reference, in meters.
        theta (float): Current pendulum angle in radians.

    Returns:
        None
    """

    # define each line for each displayed line of text
    line_phase = font.render("Phase: " + phase, True, (0, 0, 0))
    line_time = font.render("Time = +{:.2f} s".format(time_list[-1]), True, (0, 0, 0))
    line_theta = font.render("Pendulum Angle: {:.2f} rad".format(theta), True, (0, 0, 0))
    line_x = font.render("Cart Position = {:.2f} m".format(x_error_m), True, (0, 0, 0))

    # draw each line of text, positioned appropriately in top left
    screen.blit(line_phase, (30, 15))
    screen.blit(line_theta, (30, 60))
    screen.blit(line_x, (30, 90))
    screen.blit(line_time, (30, 120))


def main():
    """
    Runs the full inverted pendulum simulation loop.

    Initializes the Pygame window, Pymunk physics space, cart-pendulum system, and control parameters.
    Handles the transition between swing-up, LQR control, and impulse response testing phases.
    Continuously updates physics, renders visuals, applies forces, and stores state variables.
    At the end of the simulation, generates plots for position, angle, energy, and control forces.

    Returns:
        None
    """

    # set font for dispaying data on the PyGame window
    font = pygame.font.SysFont("Arial", 24)

    # intitialise clock for more physically accurate visualisation runtime
    clock = pygame.time.Clock()

    # initialise screen
    size = screen_length_px, screen_height_px
    screen = pygame.display.set_mode(size)
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # # inititalise recorder for showcase gif (OPTIONAL)
    # recorder = pgr("simulation.gif")

    # initialise pymunk space
    space = pymunk.Space()
    space.gravity = 0, gravity_px

    # initialise cart
    body_c = pymunk.Body()
    body_c.position = Vec2d(*(screen_length_px - 50, screen_height_px/2))
    shape_c = pymunk.Segment(body_c, (pend_length_px / 5, 0), (-pend_length_px / 5, 0), 5)
    shape_c.mass = cart_mass_kg
    space.add(body_c, shape_c)

    # initialise linear rail
    groove = pymunk.GrooveJoint(space.static_body, body_c, (-screen_length_px*100, screen_height_px/2), (screen_length_px*10000, screen_height_px/2), (0,0))
    space.add(groove)

    # initialise pendulum
    body_p = pymunk.Body()
    body_p.position = Vec2d(*(screen_length_px - 50, screen_height_px/2 + pend_length_px))
    shape_p = pymunk.Circle(body_p, 10)
    shape_p.mass = pend_mass_kg
    space.add(body_p, shape_p)

    # initialise rigid rod constraint between cart and pendulum
    c: pymunk.Constraint = pymunk.PinJoint(body_p, body_c, (0, 0), (0, 0))
    space.add(c)

    # initialise variables for relevant equations in [1]
    l = pend_length_px / pixels_per_meter
    m_p = body_p.mass
    m_c = body_c.mass
    g = gravity_px / pixels_per_meter
    J_p = m_p * l**2 # moment of inertia for pendulum with massless rod with single mass concentrated at end of rod
    w_0 = math.sqrt( ( m_p * g * l ) / J_p)
    dt = CONFIG["time_step_s"]

    # define max acceleration based on possible motor design constraint
    max_accel = CONFIG["max_acceleration_g"] * g
    max_cart_force = m_c * max_accel * pixels_per_meter
    umax = max_cart_force

    # initialise list for trail of pendulum positions and define length of trail
    trail = []
    max_trail_length = 150

    # intitialise data collection lists
    time_list = [0]
    theta_list = [-np.pi]
    x_list = [body_c.position[0]/pixels_per_meter]
    x_dot_list = [0]
    theta_dot_list = [0]
    E_norm_list = [get_energy(-np.pi, 0, J_p, m_p, g, l)]
    swing_up_force_list = [0]
    control_force_list = []
    control_time_list = []
    F_x_list = []
    F_x_dot_list = []
    F_theta_list = []
    F_theta_dot_list = []

    # initialise phase Booleans
    start_control = False
    apply_impulse = True
    phase = "STANDBY"

    # define LQR control gains based off control.m (MatLab control analysis results)
    K_matrix = np.array([[-70.7107, -64.3191, 379.2017, 66.4718]]) # [K_1, K_2, K_3, K_4]

    # define LQR reference matrix to set stationary steady state position at the center of the screen
    reference_matrix = np.array([
                                [(screen_length_px / 2) / pixels_per_meter], # x (cart position relative to centre of screen in) [m]
                                [0], # x dot [m/s]
                                [0], # theta [rad]
                                [0]  # theta dot [rad/s]
                                ]) 

    # main simulation loop
    time_step_count = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # store time stamps of each simulation time step
        time_list.append(time_list[-1] + dt)

        # get state variables
        (x, x_dot, theta, theta_dot, theta_control, theta_dot_control, x_error_m, pend_x0, theta_raw, pend_xc) = get_state_variables(body_c, body_p, reference_matrix, x_list[-1], theta_list[-1], dt)

        # store state variables
        x_list.append(x)
        theta_list.append(theta)
        x_dot_list.append(x_dot)
        theta_dot_list.append(theta_dot)

        # add current pendulum position to trail
        trail.append((int(pend_x0), int(body_p.position[1])))
        # remove oldest point to keep trail length constant
        if len(trail) > max_trail_length:
            trail.pop(0)

        # store current energy
        E_norm_list.append(get_energy(theta, theta_dot, J_p, m_p, g, l))

        # apply cart/rail friction force based on cart velocity
        friction_force_px = get_friction_force(body_c, pixels_per_meter, CONFIG["friction_coefficient"])
        body_c.apply_force_at_local_point((friction_force_px, 0), (0, 0))
        
        # if the angle is sufficiently large and control hasn't started, initiate swing-up after 1 second has elapsed
        if theta % (2*np.pi) > 0.05 and not start_control and time_list[-1] > 1:

            # get and apply swing-up force on cart
            input_force_px = get_swing_up_force(theta, theta_dot, J_p, m_p, g, l, umax, m_c, pixels_per_meter)
            body_c.apply_force_at_local_point((input_force_px, 0), (0, 0))

            # store relevant data in SI units
            input_force_N = input_force_px / pixels_per_meter
            swing_up_force_list.append(input_force_N)
            phase = "SWING-UP"

        # if not swinging up, store a zero swing-up force
        else:
            swing_up_force_list.append(0)

        # activate control if angle is sufficiently small and the pendulum is moving away from a zero angle
        if abs(theta_raw) < 0.1 and np.sign(pend_xc) == np.sign(theta_dot) and not start_control:
            print(" * control activated * ")
            start_control = True

        if start_control == True: 

            # get control force and apply to cart
            (control_force_px, x_matrix) = get_LQR_control_force(x, x_dot, theta_control, theta_dot_control, reference_matrix, K_matrix, umax, pixels_per_meter)
            body_c.apply_force_at_local_point((control_force_px, 0), (0, 0))

            # store relevant data specific to control phase in SI units
            control_force_N = control_force_px / pixels_per_meter
            control_force_list.append(control_force_N)
            control_time_list.append(time_list[-1])
            F_x_list.append(-K_matrix[0][0] * x_matrix[0][0])
            F_x_dot_list.append(-K_matrix[0][1] * x_matrix[1][0])
            F_theta_list.append(-K_matrix[0][2] * x_matrix[2][0])
            F_theta_dot_list.append(-K_matrix[0][3] * x_matrix[3][0])
            phase = "LQR CONTROL"
        

        # apply 15 N impulse force once settled to test impulse response
        if time_list[-1] >= 6.0 and apply_impulse:
            apply_impulse = False
            body_p.apply_force_at_local_point((15 * pixels_per_meter, 0), (0, 0))

        if time_list[-1] >= 6.0:
            phase = "LQR CONTROL IMPULSE RESPONSE"

        if time_list[-1] >= 9.0:
            phase = "STEADY STATE"

        # draw background, pendulum, rod, cart, and linear rail constraint
        draw_scene(screen, trail, groove, space, body_p, body_c, c, shape_c, shape_p)

        # draw simulation phase and current data ontop of scene
        draw_data(screen, font, phase, time_list, x_error_m, theta)

        # # save screen as png for every 7 time-steps (OPTIONAL)
        # if time_step_count % 7 == 0:
        #     recorder.click(screen)

        pygame.display.update()
        space.step(dt)

        clock.tick(1/dt)
        time_step_count += 1

    # # save pictures and gif (OPTIONAL)
    # recorder.save()

    pygame.quit()

    # plot state variables against time
    fig, axs = plt.subplots(4)
    axs[0].plot(time_list, x_list)
    axs[0].set_title("$x$ [m] vs Time")
    axs[0].grid()
    axs[1].plot(time_list, x_dot_list)
    axs[1].set_title("$\\dot{x}$ [m/s] vs Time")
    axs[1].grid()
    axs[2].plot(time_list, theta_list)
    axs[2].set_title("$\\theta$ [rad] vs Time")
    axs[2].grid()
    axs[3].plot(time_list, theta_dot_list)
    axs[3].set_title("$\\dot{\\theta}$ [rad/s] vs Time")
    axs[3].grid()
    fig.tight_layout()

    # plot force applied to cart and energy during swing-up against time to imitate plots in [1]
    fig1, axs1 = plt.subplots(2)
    axs1[0].plot(time_list, swing_up_force_list)
    axs1[0].set_title("Swing Up Force [N] vs Time [s]")
    axs1[0].grid()
    axs1[1].plot(time_list, E_norm_list)
    axs1[1].set_title("Normalised Energy [J] vs Time [s]")
    axs1[1].grid()
    fig1.tight_layout()

    # plot force components based on controller gains and state variables against time
    fig2, ax_force = plt.subplots()
    ax_force.plot(control_time_list, control_force_list, label='Total Force')
    ax_force.plot(control_time_list, F_x_list, label='$x \\times K_1$')
    ax_force.plot(control_time_list, F_x_dot_list, label='$\\dot{x} \\times K_2$')
    ax_force.plot(control_time_list, F_theta_list, label='$\\theta \\times K_3$')
    ax_force.plot(control_time_list, F_theta_dot_list, label='$\\dot{\\theta} \\times K_4$')
    ax_force.set_title("Control Force Components vs Time")
    ax_force.set_ylabel("Component Force (N)")
    ax_force.set_xlabel("Time (s)")
    ax_force.legend(loc='upper left')
    ax_force.grid()
    fig2.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()