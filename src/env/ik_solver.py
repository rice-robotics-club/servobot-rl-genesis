import math

import numpy as np


# ik / fk for 4-bar linkage
def calculate_theta3(l0, l1, l2, l3, theta1, open_mode=True):
    """
    Calculates the output angle (theta3) of a four-bar linkage.

    Parameters:
    l0, l1, l2, l3: Lengths of the links (l0 is ground)
    theta1_deg: Input angle in degrees
    open_mode: Boolean to choose between the two assembly configurations
    """
    # Freudenstein Constants
    K1 = l0 / l1
    K2 = l0 / l3
    K3 = (l1**2 - l2**2 + l3**2 + l0**2) / (2 * l1 * l3)

    # Quadratic coefficients for the half-angle tangent substitution
    # (At^2 + Bt + C = 0)
    A = math.cos(theta1) - K1 - K2 * math.cos(theta1) + K3
    B = -2 * math.sin(theta1)
    C = K1 - (K2 + 1) * math.cos(theta1) + K3

    # Discriminant check
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return None  # Linkage cannot reach this position

    if open_mode:
        t = (-B + math.sqrt(discriminant)) / (2 * A)
    else:
        t = (-B - math.sqrt(discriminant)) / (2 * A)

    theta3 = 2 * math.atan(t)
    return math.degrees(theta3)


def solve_for_th_L12(z, y, L_0):
    L_mag = math.sqrt(z**2 + y**2)
    L_12 = math.sqrt(L_mag**2 - L_0**2)

    t_h = math.atan2(z, y) + math.atan2(L_12, L_0)
    t_h = math.degrees(t_h)
    return t_h, L_12


def wrap_angle(rad_angle):
    return (rad_angle + math.pi) % (2 * math.pi) - math.pi


def find_tb_tc(x_EE, z_EE, L_1, L_2):
    if math.sqrt(x_EE**2 + z_EE**2) > math.sqrt(L_1**2 + L_2**2):
        print("Leg position unreachable!", x_EE, z_EE)
        return None, None
    t_c = math.acos((x_EE**2 + z_EE**2 - L_1**2 - L_2**2) / (2 * L_1 * L_2))
    t_b = math.atan2(z_EE, x_EE)
    t_b = t_b - math.atan2(L_2 * math.sin(t_c), L_1 + L_2 * math.cos(t_c))
    # Wrap to (-pi, pi]
    wrapped_t_b = wrap_angle(t_b)
    return wrapped_t_b, t_c


def convert_tb_tc_to_ta_tl(
    t_b, t_c, l0, l1, l2, l3
):  # NOTE: l0-->l3 = 4-bar linkage links$
    t_3 = t_b + math.pi
    t_a = calculate_theta3(
        l0, l3, l2, l1, t_3, open_mode=False
    )  # degrees NOTE: l3 and l1 $
    if t_a is None:
        print("Linkage not solvable!")
        return None, None

    t_a = t_a + 180
    t_l = math.degrees(t_3 + t_c) + 180
    return t_a, t_l


def calculate_leg_ik(target_pos: np.array, L_0, L_1, L_2, l0, l1, l2, l3):
    """
    Given desired position of leg EE and lengths of leg, output angles of leg
    note: this does NOT consider configuration of leg
    assumptions:
    1. assuming leg is two linkages in series with output angles = angle at each linkage
    2. L1 != L2
    3. hip and upper_leg axes intersect
    input:
    1. Desired position of leg EE (w.r.t point of intersection between hip and upper_leg ax
        If leg is below the chassis (almost always), Z is negative.
    2. L_0, L_1, L_2 = lengths between the following joints: hip->upper_leg, upper_leg->lower_leg, lower_leg->EE
    output: list of solutions (usually 2) [[hip_angle, upper_angle, lower_angle], [...]]
    3. l0, l1, l2, l3 = linkage lengths, with l0 = ground, and l1
    """
    x, y, z = target_pos[0], target_pos[1], target_pos[2]

    t_h, L_12 = solve_for_th_L12(z, y, L_0)
    if t_h > math.pi / 2 or t_h < -math.pi / 2:
        print("Hip angle invalid! - IK not solved", t_h)
        return None, None, None

    if z < 0:
        # negate L_12 because converting a scalar val to signed val
        L_12 = -L_12
    t_b, t_c = find_tb_tc(x_EE=x, z_EE=L_12, L_1=L_1, L_2=L_2)
    if t_b is None or t_c is None:
        print("Leg angle invalid! - IK not solved")
        return None, None, None

    t_a, t_l = convert_tb_tc_to_ta_tl(t_b, t_c, l0, l1, l2, l3)
    if t_a is None or t_l is None:
        print("Linkage invalid! - IK not solved")
        return None, None, None

    return t_h, t_a, t_l


"""
NEW: Given servo angles t_a and t_l, extract the linkage angles or specific link angles
"""


def convert_ta_tl_to_tb_tc(t_a, t_l, l0, l1, l2, l3):
    """
    Inverse of convert_tb_tc_to_ta_tl.
    Given servo angles t_a and t_l, return the linkage angles t_b and t_c.

    The forward conversion is:
    - t_3 = t_b + pi
    - t_a = calculate_theta3(l0, l3, l2, l1, t_3, open_mode=False) + 180  (degrees)
    - t_l = degrees(t_3 + t_c) + 180

    This function inverts that to find t_b and t_c.

    Parameters:
    t_a, t_l: Servo angles in degrees
    l0, l1, l2, l3: 4-bar linkage lengths

    Returns:
    t_b, t_c: Linkage angles in radians
    """
    target_theta3_deg = t_a - 180
    target_theta3_rad = math.radians(target_theta3_deg)

    # Analytical inverse using Freudenstein's equation symmetry
    best_t_3 = None
    for mode in [False, True]:
        t_3_deg = calculate_theta3(l0, l1, l2, l3, target_theta3_rad, open_mode=mode)
        if t_3_deg is not None:
            # Verify the inverse
            t_3_rad = math.radians(t_3_deg)
            test_ta_deg = calculate_theta3(l0, l3, l2, l1, t_3_rad, open_mode=False)
            if test_ta_deg is not None and abs(test_ta_deg - target_theta3_deg) < 1.0:
                best_t_3 = t_3_rad
                break

    if best_t_3 is None:
        return None, None

    # Extract linkage angles
    # From t_3 = t_b + pi, solve for t_b
    t_b = best_t_3 - math.pi
    t_b = wrap_angle(t_b)

    # From t_l = degrees(t_3 + t_c) + 180, solve for t_c
    t_c = math.radians(t_l - 180) - best_t_3
    t_c = wrap_angle(t_c)

    return t_b, t_c


def convert_ta_tl_to_160_225(t_a, t_l, l0, l1, l2, l3):
    """
    Given servo angles t_a and t_l, extract the angle t_c between 160mm and 225m_top links.

    Parameters:
    t_a, t_l: Servo angles in degrees
    l0, l1, l2, l3: 4-bar linkage lengths

    Returns:
    t_c: Angle between 160mm and 225m_top links in radians
    """
    t_b, t_c = convert_ta_tl_to_tb_tc(t_a, t_l, l0, l1, l2, l3)

    if t_c is None:
        return None

    return t_c


def get_tc_limits(
    l0, l1, l2, l3, t_a_min=-180, t_a_max=180, t_l_min=-180, t_l_max=180, step=1
):
    """
    Find the minimum and maximum allowed angles for t_c (angle between 160mm and 225m_top).

    Samples across the valid servo angle ranges and tracks the extremes of t_c.
    If extrema occur at servo angle boundaries, uses the next safest value instead.

    Parameters:
    l0, l1, l2, l3: 4-bar linkage lengths
    t_a_min, t_a_max: Valid range for servo angle t_a in degrees
    t_l_min, t_l_max: Valid range for servo angle t_l in degrees
    step: Sampling step size in degrees (smaller = more thorough but slower)

    Returns:
    t_c_min, t_c_max: Min and max t_c angles in radians, or (None, None) if no valid configurations
    """
    tc_values = []  # List of (t_a, t_l, t_c) tuples

    # Sample across servo angle ranges
    t_a = t_a_min
    while t_a <= t_a_max:
        t_l = t_l_min
        while t_l <= t_l_max:
            t_c = convert_ta_tl_to_160_225(t_a, t_l, l0, l1, l2, l3)
            if t_c is not None:
                tc_values.append((t_a, t_l, t_c))
            t_l += step
        t_a += step

    if not tc_values:
        print("No valid t_c values found in the given servo angle ranges!")
        return None, None

    # Sort by t_c value to find extrema
    tc_values_sorted = sorted(tc_values, key=lambda x: x[2])

    # Find min t_c, excluding edge positions
    t_c_min = None
    for t_a, t_l, t_c in tc_values_sorted:
        if not (t_a == t_a_min or t_a == t_a_max or t_l == t_l_min or t_l == t_l_max):
            t_c_min = t_c
            break

    # If all min values are on edges, use the overall minimum but warn
    if t_c_min is None:
        if tc_values_sorted:
            t_c_min = tc_values_sorted[0][2]
            print(
                "Warning: t_c minimum is at servo angle boundary; result may be unreliable."
            )

    # Find max t_c, excluding edge positions
    t_c_max = None
    for t_a, t_l, t_c in reversed(tc_values_sorted):
        if not (t_a == t_a_min or t_a == t_a_max or t_l == t_l_min or t_l == t_l_max):
            t_c_max = t_c
            break

    # If all max values are on edges, use the overall maximum but warn
    if t_c_max is None:
        if tc_values_sorted:
            t_c_max = tc_values_sorted[-1][2]
            print(
                "Warning: t_c maximum is at servo angle boundary; result may be unreliable."
            )

    return t_c_min, t_c_max


# TESTS:

# Z
# |         (X pointing out)
# 0--- Y

#  L0
# ---+
#    |         <-- = L0
#    |
#    | L12

#  L0
# +---
# |            <-- = -L0
# |
# | L12

# print(calculate_leg_ik(target_pos=[-27,-3.5,-20], L_0=-3.5, L_1=20.7, L_2=15, l0=8.5, l1=6.2, l2=8.7, l3=16.63))
print(
    calculate_leg_ik(
        target_pos=[-5, 3.5, -10],
        L_0=3.5,
        L_1=20.7,
        L_2=15,
        l0=8.5,
        l1=6.2,
        l2=8.7,
        l3=16.63,
    )
)


print(solve_for_th_L12(z=0, y=5, L_0=3))
# print(solve_for_th_L12(z=-4, y=-3, L_0=-3)) # for legs on the right, since L0 points in opposite dir of Y, L_0 is negative

print(find_tb_tc(x_EE=0, z_EE=-7, L_1=4, L_2=3))
# print(find_tb_tc(x_EE=7, z_EE=0, L_1=4, L_2=3))
# print(find_tb_tc(x_EE=-7, z_EE=0, L_1=4, L_2=3))


print(calculate_theta3(l0=4, l1=2, l2=4, l3=2, theta1=np.pi / 2, open_mode=True))
