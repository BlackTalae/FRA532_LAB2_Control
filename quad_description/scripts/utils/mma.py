# Motor mixture model
def mma(thrust, roll, pitch, yaw):
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
    
    FR_vel = thrust + yaw + pitch + roll
    HL_vel = thrust + yaw - pitch - roll
    FL_vel = thrust - yaw + pitch - roll
    HR_vel = thrust - yaw - pitch + roll
    
    return FR_vel , HL_vel , FL_vel , HR_vel