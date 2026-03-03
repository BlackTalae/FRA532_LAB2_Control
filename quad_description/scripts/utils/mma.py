import math
# Motor mixture model
def mma(thrust, roll, pitch, yaw, hover_bias=0):
    '''
    Input each param : [FR , HL , FL , HR] 
    '''
    
    # FL                 FR   
    #    .           .
    #        .   .
    #          o
    #        .   .
    #    .           .
    # HL                 HR


    # --- Position
    # FR , HL , FL , HR
    # --- Joint name
    # 0  , 1  , 2  , 3
    
    FR_vel = hover_bias + thrust - yaw - pitch - roll
    HL_vel = hover_bias + thrust - yaw + pitch + roll
    FL_vel = hover_bias + thrust + yaw - pitch + roll
    HR_vel = hover_bias + thrust + yaw + pitch - roll
    
    return FR_vel , HL_vel , FL_vel , HR_vel

def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def shortest_angular_distance(from_angle, to_angle):
    """
    Compute the shortest signed angular distance from from_angle to to_angle.
    Both angles should be in radians.
    Returns a value in [-pi, pi] representing the shortest rotation.
    """
    diff = to_angle - from_angle
    # Normalize to [-pi, pi]
    while diff > math.pi:
        diff -= 2.0 * math.pi
    while diff < -math.pi:
        diff += 2.0 * math.pi
    return diff

def thrust_to_omega(F, kf, Fmax):
    F = min(max(0.0, F), Fmax)
    return math.sqrt(F / kf)