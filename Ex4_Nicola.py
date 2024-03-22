#-6.8e6
#10717
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.signal import find_peaks


#see website
#automatic option to do at 45 degree inital pos (w always 90 degree initial vel), at desired magnitudes for pos and v

#calculate eccentricity? or get vy from a given e
#put time in plot. period, eccentricity, energies? plot arrow for rocket. need different size and color for moon

# both circular and eccentric orbits - stable and repeatable?
# energy conservation - accuracy. what dt needed
# error


G = 6.67e-11# m^3*kg^-1*s^-2, Gravitational constant
Me = 5.97e24#kg,  Mass of Earth
m = 1 #kg, Mass of the Rocketship
Mm = 7.35e22 #kg, Mass of Moon
Ms = 1.99e30 #kg, Mass of the Sun

def circ_vel(x0): 
    """
    Function to calculate the (vy) velocity needed for a circular orbit, given an initial distance in x. 

    Parameters
    ----------
    x0 : The orbiting body's radius
    
    Returns
    -------
    vy:  y-component of the body's velocity

    """
    return np.sqrt((G*Me)/abs(x0))


Re_0 = [0, 0, 0] #m, Initial position (x, y, z) of the Earth

Rm_0 = [384.4e6, 0, 0] #m, Initial position (x, y, z) of the Moon

Rs_0 = [150e9, 0, 0] #m, Initial position (x, y, z) of the Sun

############################################################
#Part a functions

def f(v): 
    """
    First Runge-Kutta function for the case of a rocket orbiting the Earth 
    ----------
    Parameters
    ----------
    v : Three-dimensional velocity array of the orbiting rocket
    
    Returns
    -------
    v : Three-dimensional velocity array of the orbiting rocket

    """
    return v

def f_2(r): 
    """
    Second Runge-Kutta function for the case of a rocket orbiting the Earth 
    ----------
    Parameters
    ----------
    r : Three-dimensional position array of the orbiting rocket
    
    Returns
    -------
    dv/dt: Three-dimensional acceleration array of the orbiting rocket

    """
    dv_dt = -(G*Me*r)/(((r[0])**(2)+ (r[1])**(2) + (r[2])**(2))**(3/2))
    return dv_dt

#############################################################
#Part b functions

def f_2_with_moon(r): 
    """
    Second Runge-Kutta function for the case of a rocket orbiting the Earth and the Moon
    ----------
    Parameters
    ----------
    r : Three-dimensional position array of the orbiting rocket
    
    Returns
    -------
    dv/dt: Three-dimensional acceleration array of the orbiting rocket

    """
    rm = r - Rm_0
    re = r - Re_0

    dv_dt = -(G*Me*re)/((re[0]**(2)+ re[1]**(2) + re[2])**(3/2)) - (G*Mm*rm)/((rm[0]**(2)+ rm[1]**(2) + rm[2]**2)**(3/2))
    return dv_dt

#############################################################
#Part c functions

def f_2_e(re, rm): 
    """
    Second Runge-Kutta function for the gravitational effects fo the Moon on the Earth
    ----------
    Parameters
    ----------
    re : Three-dimensional position array of the orbiting Earth
    rm : Three-dimensional position array of the orbiting Moon
    
    Returns
    -------
    dv/dt: Three-dimensional acceleration array of the orbiting Earth

    """
    r = re - rm
    dv_dt = - (G*Mm*r)/(((r[0])**(2) + (r[1])**(2) + (r[2])**(2))**(3/2))
    return dv_dt

def f_2_m(re, rm): 
    """
    Second Runge-Kutta function for the gravitational effects fo the Earth on the Moon
    ----------
    Parameters
    ----------
    re : Three-dimensional position array of the orbiting Earth
    rm : Three-dimensional position array of the orbiting Moon
    
    Returns
    -------
    dv/dt: Three-dimensional acceleration array of the orbiting Moon

    """
    r = rm - re
    dv_dt = - (G*Me*r)/(((r[0])**(2) + (r[1])**(2) + (r[2])**(2))**(3/2))
    return dv_dt

#############################################################
#Part d functions

def f_2_e_with_Sun(re, rm): 
    """
    Second Runge-Kutta function for the gravitational effects of the Moon and the Sun on Earth
    ----------
    Parameters
    ----------
    re : Three-dimensional position array of the orbiting Earth
    rm : Three-dimensional position array of the orbiting Moon
    
    Returns
    -------
    dv/dt: Three-dimensional acceleration array of the orbiting Earth

    """
    r_e = re - rm
    r_s = re - Rs_0
    dv_dt = - G*Mm*r_e/(((r_e[0])**(2) + (r_e[1])**(2) + (r_e[2])**(2))**(3/2)) - (G*Ms*r_s)/(((r_s[0])**(2) + (r_s[1])**(2) + (r_s[2])**(2))**(3/2))
    return dv_dt

def f_2_m_with_Sun(re, rm): 
    """
    Second Runge-Kutta function for the gravitational effects of the Earth and the Sun on the Moon
    ----------
    Parameters
    ----------
    re : Three-dimensional position array of the orbiting Earth
    rm : Three-dimensional position array of the orbiting Moon
    
    Returns
    -------
    dv/dt: Three-dimensional acceleration array of the orbiting Moon
    """
    r_e = rm - re
    r_s = rm - Rs_0
    dv_dt = - G*Me*r_e/(((r_e[0])**(2) + (r_e[1])**(2) + (r_e[2])**(2))**(3/2)) - (G*Ms*r_s)/(((r_s[0])**(2) + (r_s[1])**(2) + (r_s[2])**(2))**(3/2))
    return dv_dt

#############################################################

def energy_a(r, v): 
    """
    Function for the calculaton of the rocket's kinetic, potential, and total energy, when it's orbiting the Earth.
    Used in part a). 
    ----------
    Parameters
    ----------
    r : Three-dimensional position array of the orbiting rocket
    v: Three-dimensional veolcity array of the orbiting rocket
    
    Returns
    -------
    TE: the total energy of the rocket 
    KE: the kinetic energy of the rocket
    PE: the potential energy of the rocket
    """

    KE = 0.5*m*((v[0])**2 + (v[1])**2 + v[2]**2)
    PE = (-G*Me*m)/(np.sqrt((r[0])**2 + (r[1])**2 + (r[2])**2))
    TE = KE + PE
    return TE, KE, PE

def energy_b(r, v):
    """
    Function for the calculaton of the rocket's kinetic, potential, and total energy, when it's orbiting the Earth and the Moon. 
    Used in part b). 
    ----------
    Parameters
    ----------
    r : Three-dimensional position array of the orbiting rocket
    v: Three-dimensional veolcity array of the orbiting rocket
    
    Returns
    -------
    TE: the total energy of the rocket 
    KE: the kinetic energy of the rocket
    PE: the potential energy of the rocket
    """

    M = Me + Mm

    KE = 0.5*m*((v[0])**2 + (v[1])**2 + v[2]**2)
    PE = (-G*M*m)/(np.sqrt((r[0])**2 + (r[1])**2 + (r[2])**2)) 
    TE = KE + PE

    return TE, KE, PE

def RK4_iteration(r_start_vals, v_start_vals,  dt, t_max, f, f_2, f_3 = 1, r2_start_vals = [0, 0, 0], v2_start_vals= [0, 0, 0], part_a = False, part_c = False, part_b = False, part_d = False):
    """
    Main functon for the Runge Kutta method. 
    Allows for a single solution (solution for the rocket orbiting around the Earth or around the Earth and the Moon), 
    or for two solutions (solution for the Earth and the Moon orbiting under eachother's influence or the Earth and the Moon orbiting under eachother's and a stationary Sun's infleunce.)

    Within the Runge-Kutta iteration, it includes an energy calculation at each timestep (for parts a and b), 
    a calculation of the orbital period for each part (where applicable), a section that tests for a crash into the Moon or the Earth (for parts a and b), 
    and error calculations.
    ----------
    Parameters
    ----------
    r_start_vals : Three-dimensional initial position array of the orbiting rocket (part a and b) or Earth (parts c and d)
    v_start_vals : Three-dimensional initial velocity array of the orbiting rocket (part a and b) or Earth (parts c and d)
    dt: the timestep
    t_max: Maximum time to calculate the orbit for
    f: First Runge-Kutta function (used in all parts)
    f_2: Second Runge Kutta function for the rocket (parts a and b) or the Earth (part c and d)
    f_3: Second RUnge Kutta function for the Moon (parts c and d)
    r2_start_vals : Three-dimensional initial position array of the orbiting Moon (parts c and d). 
    v2_start_vals : Three-dimensional initial velocity array of the orbiting Moon (parts c and d).
    part_a, part_b, part_c, part_d: To differentiate between those individual cases

    
    Returns
    -------
    r[:i+1, 0]: x1 array (Rocket or Earth)
    r[:i+1, 1]: y1 array (Rocket or Earth)
    r[:i+1, 2]: z1 array (Rocket or Earth)
    TE[:i+1]: array of total energies
    KE[:i+1]: array of kinetic energies
    PE[:i+1]: array of potential energies
    te_error: array of errors in the total energies
    te_global_error: the last element of the te_error array
    mean_orbit_error: The mean error of the orbital starting position
    period: the orbital period

    if part_c is True, excludes te_error, te_global_error, mean_orbit_error, and also returns:
    r2[:i+1, 0]: x2 array (Moon)
    r2[:i+1, 1]: y2 array (Moon)
    r2[:i+1, 2]: z2 array (Moon)
    
    """

    nr_steps = int(t_max/dt)

    #Initialisation of the position, velocity and energy arrays for (if present) two objects
    r = np.zeros((nr_steps, 3))
    v = np.zeros((nr_steps, 3))

    r_2 = np.zeros((nr_steps, 3))
    v_2 = np.zeros((nr_steps, 3))
    
    r[0, :]= r_start_vals
    v[0, :]= v_start_vals

    if part_c == True: 
        r_2[0, :]= r2_start_vals
        v_2[0, :]= v2_start_vals
    
    TE = np.zeros(nr_steps)
    PE = np.zeros(nr_steps)
    KE = np.zeros(nr_steps)

    #energy consideration only done for parts a and b
    if f_2 == f_2_with_moon:
        TE[0], KE[0], PE[0] = energy_b(r_start_vals, v_start_vals)
        
    else: 
        TE[0], KE[0], PE[0]  = energy_a(r_start_vals, v_start_vals)

    period = 0
    one_orbit1= False
    one_orbit2= False
    diff = np.zeros(nr_steps)

    #RUnge Kutta iteration
    for i in range(1, nr_steps):

        if part_c == True:
            r20 = r_2[i-1, :]
            v20 = v_2[i-1, :]

        r0 = r[i-1, :]
        v0 = v[i-1, :]

        if part_c == False: 

            k1_r = f(v0)
            k1_v = f_2(r0)

            k2_r  = f(v0 + dt*(k1_v/2))
            k2_v  = f_2(r0 + dt*(k1_r/2))

            k3_r  = f(v0 + dt*(k2_v/2))
            k3_v  = f_2(r0 + dt*(k2_r/2))

            k4_r  = f(v0 + dt*k3_v)
            k4_v  = f_2(r0 + dt*k3_r)

        elif part_c == True: 

            k1_r = f(v0)
            k1_v = f_2(r0, r20)

            k2_r  = f(v0 + dt*(k1_v/2))
            k2_v  = f_2(r0 + dt*(k1_r/2), r20 + dt*(k1_r/2) )

            k3_r  = f(v0 + dt*(k2_v/2))
            k3_v  = f_2(r0 + dt*(k2_r/2), r20 + dt*(k2_r/2))

            k4_r  = f(v0 + dt*k3_v)
            k4_v  = f_2(r0 + dt*k3_r, r20 + dt*k3_r)

            ######################

            k1_2r = f(v20)
            k1_2v = f_3(r0, r20)

            k2_2r  = f(v20 + dt*(k1_2v/2))
            k2_2v  = f_3(r0 + dt*(k1_2r/2), r20 + dt*(k1_2r/2) )

            k3_2r  = f(v20 + dt*(k2_2v/2))
            k3_2v  = f_3(r0 + dt*(k2_2r/2), r20 + dt*(k2_2r/2))

            k4_2r  = f(v20 + dt*k3_2v)
            k4_2v  = f_3(r0 + dt*k3_2r, r20 + dt*k3_2r)
            
            

        r[i] = r0 + (dt/6)*(k1_r + 2*k2_r + 2*k3_r + k4_r)

        v[i] = v0 + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

        #fills these up only if part c == True
        if part_c == True: 
            r_2[i] = r20 + (dt/6)*(k1_2r + 2*k2_2r + 2*k3_2r + k4_2r)

            v_2[i] = v20 + (dt/6)*(k1_2v + 2*k2_2v + 2*k3_2v + k4_2v)

        ############################################################################33
        
        #Energy calculation for parts a and b
        if f_2 == f_2_with_moon:
            TE[i], KE[i], PE[i] = energy_b(r[i], v[i])
        
        else: 
            TE[i], KE[i], PE[i] = energy_a(r[i], v[i])

        #to get the actual Time:
        if dt < 1:
            dt_ = 1/dt
        else: 
            dt_ = dt
        
        #for part b: breaks loop after rocket returns (and orbital calculation)
        if part_b == True and one_orbit1 == False: 
            if (np.sign(r[i, 1]) != np.sign(r[i-1, 1])) and r[i-1, 0] != r_start_vals[0] and r[i, 0] < Rm_0[0]/2:
                period = (i-1)*dt_
                
                one_orbit1 = True
        
        if one_orbit1 == True: 
            break
        
        #for part b: returns the closest possible distance from the moon without crashing
        if part_b == True and one_orbit2 == False:
            if (np.sign(r[i, 1]) != np.sign(r[i-1, 1])) and r[i-1, 0] != r_start_vals[0]:
                if v_start_vals[1] == 10567:
                    print(f"The closest possible distance from the Moon without crashing is {(distance_from_moon):.2f} m.")
                    one_orbit2 = True

        if one_orbit2 == True: 
            break

        #Orbital period calculation for part c
        if part_c == True and part_d == False and one_orbit1 == False: 
            if (np.sign(r_2[i, 1]) != np.sign(r_2[i-1, 1])) and r_2[i-1, 0] != Rm_0[0] and r_2[i, 0] > Re_0[0]:
        
                period = (i-1)*dt_
                
                one_orbit1 = True
    

        #Orbital period calculation for part c
        if part_d == True and one_orbit1 == False:
            if (np.sign(r[i, 1]) != np.sign(r[i-1, 1])) and r[i-1, 0] != 0.0 and r[i, 0] < Rs_0[0]:
                print(r[i, 0])
                period = (i-1)*dt_
                
                one_orbit1 = True
        
       

        
        #for the crash testing 
        seconds = (i-1)*dt_

        hours =seconds/3600
        days = seconds/86400

        def angle(r1, r2): 
            mag1 = np.linalg.norm(r1)
            mag2 = np.linalg.norm(r2)
            dot = np.dot(r1, r2)
            cos_theta = dot/(mag1*mag2)
            rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            deg = np.degrees(rad)

            return deg
        
        crashed = False
        distance_from_earth = np.linalg.norm(r[i] - Re_0)
        distance_from_moon = np.linalg.norm(r[i] - Rm_0)

        if r2_start_vals == [0, 0, 0]:
        #i.e if part a or b
            
            deg1 = angle(r[i-2], r[i-1])
            deg2 = angle(r[i-1], r[i])
            deg_diff = (deg2-deg1)/deg1

            
            #conditions to test whether rocket has crashed
            if deg_diff == np.inf or deg_diff == np.nan  and (distance_from_earth < 1e7 or distance_from_moon < 1e7) or r[i, 0] > Rm_0[0] + 1e8:

                if distance_from_earth < distance_from_moon:
                    print(f"Rocket crashed into Earth after {(seconds):.2f} Seconds, {(hours):.2f} Hours and {(days):.2f} Days.")

                else: 
                    print(f"Rocket crashed into the Moon after {(seconds):.2f} Seconds, {(hours):.2f} Hours and {(days):.2f} Days.")

                crashed = True
        

        if crashed == True: 
            break

        #for the orbital period of a) - see below
        diff[i] = np.linalg.norm(r[i] - r_start_vals)

    #gets error in the total energy for each timestep.
    te_error = abs(TE[:i+1] - TE[0])/abs(TE[0])
    
    #gets the last (accumulated) error of the total energy
    te_global_error = te_error[-1]

    #Orbital period calculation for a - done this way to get a list of period-times
    trough_indices, _ = find_peaks(-diff)
    if len(trough_indices) != 0:
        period2 = trough_indices[0]*dt_
    else: 
        period2 = 0
    
    if part_a == True:
        period = period2

    #gets error in a repeating position of the orbit (r_start_vals), and its mean
    diff = diff[trough_indices]
    orbit_error = diff/np.linalg.norm(r_start_vals)
    mean_orbit_error = np.mean(orbit_error)


    
    if part_c == False:
        r = r[:i+1]
        return  r[:i+1, 0], r[:i+1, 1], r[:i+1, 2], TE[:i+1], KE[:i+1], PE[:i+1], te_error, te_global_error, mean_orbit_error, period
    
    elif part_c == True: 
        r = r[:i+1]
        r_2 = r_2[:i+1]
        return  r[:i+1, 0], r[:i+1, 1], r[:, 2], r_2[:i+1, 0], r_2[:i+1, 1], r_2[:i+1, 2], TE[:i+1], KE[:i+1], PE[:i+1], period

#############################################################
#Plotting functions

def plot_2Dorbit(x_vals, y_vals, x2_vals = [], y2_vals = [], moon = False, Sun = False, zoom = False): 
    """
    Plots the 2D orbit(s)

    Parameters
    ----------
    x_vals : Three-dimensional position array of the orbiting rocket or Earth
    y_vals: Three-dimensional veolcity array of the orbiting rocket or Earth
    x2_vals : Three-dimensional position array of the orbiting Moon if present
    y2_vals: Three-dimensional veolcity array of the orbiting Moon if present
    moon, Sun: For different plotting in those cases
    zoom: To zoom into the Earth-Moon orbits of part d
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(x_vals, y_vals, c = "b")
    plt.plot(x2_vals, y2_vals, c = "r")

    if moon == False: 
        
        ax.set_xlim((x_vals.min())*1.15, (x_vals.max())*1.15)
        ax.set_ylim((y_vals.min())*1.15, (y_vals.max())*1.15)
        plt.plot([], [], c = "b", label = "Rocket")
        plt.plot(0, 0, "ko", label = "Earth")
        plt.title("2D Simulation of Rocket's orbit around Earth and Moon")
        plt.gca().set_aspect('equal', adjustable='box')

    elif moon == True: 
        
        if x2_vals == []:

            plt.plot(Re_0[0], Re_0[0], "ko",c= "b", label = "Earth" )
            plt.plot(Rm_0[0], Rm_0[1], "ko", c = "r", label = "Moon")
            plt.plot([], [], c = "darkblue", label = "Rocket")
            plt.title("2D Simulation of Rocket's orbit around Earth and Moon")

        else: 

            plt.plot(Re_0[0], Re_0[0], "ko",c= "b", label = "Earth" )
            plt.plot(Rm_0[0], Rm_0[1], "ko", c = "r", label = "Moon")
            plt.title("2D Simulation of Earth and Moon's orbits")

        if Sun == True: 
            
            plt.title("2D Orbit Simulation around Earth, Moon and Sun")

            #ax.set_xlim(0, 2.25*Rs_0[0])
            #ax.set_ylim(-1.25*Rs_0[0], 1.25*Rs_0[0])
            if zoom == True: 
                ax.set_xlim(0, 1.5e10)
                ax.set_ylim(0, 7e10)
            else: 
                plt.gca().set_aspect('equal', adjustable='box')     

            plt.plot(Rs_0[0], Rs_0[1], "ko",  c = "y", label = "Sun")
            
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend(fontsize = "small")
    plt.show()

def plot_3Dorbit(x_vals, y_vals, z_vals, x2_vals = [], y2_vals = [], z2_vals = [],moon = False, Sun = False, zoom = False): 
    """
    Plots the 3D orbit(s). 
    Parameters are the same as in plot_2Dorbit except for the inlusion of z-values. 

    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_vals, y_vals, z_vals, c = "b")
    ax.plot(x2_vals, y2_vals, z2_vals, c = "r")

    if moon == False: 

       
        ax.set_xlim((x_vals.min())*1.15, (x_vals.max())*1.15)
        ax.set_ylim((y_vals.min())*1.15, (y_vals.max())*1.15) 
        ax.set_zlim((y_vals.min())*1.15, (y_vals.max())*1.15) 

        plt.plot(0, 0, 0, "ko", label = "Earth")
        plt.plot([], [], [], c = "darkblue", label = "Rocket")
        plt.title("3D Orbit Simulation of Rocket around Earth")
        plt.gca().set_aspect('equal', adjustable='box')

    elif moon == True: 

        
        if x2_vals == []:
        #i.e part b

            ax.set_xlim((x_vals.min())*3, (x_vals.max())*1.07) 
            ax.set_ylim((y_vals.min())*1.15, (y_vals.max())*1.15)
            ax.set_zlim((y_vals.min())*1.15, (y_vals.max())*1.15)

            plt.plot(Re_0[0], Re_0[0], Re_0[2], "ko",c= "b", label = "Earth" )
            plt.plot(Rm_0[0], Rm_0[1], Rm_0[2], "ko", c = "r", label = "Moon")
            plt.plot([], [], [], c = "darkblue", label = "Rocket")
            plt.title("23 Simulation of Rocket's orbit around Earth and Moon")

        else: 
        #i.e part c and d
            

            ax.set_xlim((x2_vals.min())*1.15, (x2_vals.max())*1.15) 
            ax.set_ylim((y2_vals.min())*1.15, (y2_vals.max())*1.15)
            ax.set_zlim((x2_vals.min())*1.15, (x2_vals.max())*1.15)

            plt.plot([], [], [], c = "darkblue", label = "Earth")
            plt.plot([], [], [], c = "red", label = "Moon")
            plt.title("3D Simulation of Earth and Moon's orbits")

            if Sun == True: 

                if zoom == True: 

                    ax.set_xlim(0, 1.5e10)
                    ax.set_ylim(0, 7e10)     

                else: 

                    ax.set_xlim(-(abs(x2_vals.min()))*100, (x2_vals.max())*1.05) 
                    ax.set_ylim((y2_vals.min())*1.15, (y2_vals.max())*1.15)
                    plt.gca().set_aspect('equal', adjustable='box')

                plt.plot(Rs_0[0], Rs_0[1], Rs_0[2], "ko",  c = "y", label = "Sun")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    plt.legend()
    plt.show()

def plot_2Danimated_orbit(x_vals, y_vals, dt, x2_vals = [], y2_vals = [], moon = False, Sun = False, zoom = False): 
    """"
    Plots the animation of the 2D orbit. 
    Same parameters as plot_2Dorbit except the timestep dt is included (to get the correct time)

    """
    fig, ax = plt.subplots()

    if moon == False: 

        frame_factor = 1
       
        ax.set_xlim((x_vals.min())*1.15, (x_vals.max())*1.15)
        ax.set_ylim((y_vals.min())*1.15, (y_vals.max())*1.15) 
        plt.plot(0, 0, "ko", label = "Earth")
        plt.plot([], [], c = "darkblue", label = "Rocket")
        plt.title("2D Orbit Simulation of Rocket around Earth")
        plt.gca().set_aspect('equal', adjustable='box')

    elif moon == True: 

        
        if x2_vals == []:
        #i.e part b

            frame_factor = 7

            ax.set_xlim((x_vals.min())*3, (x_vals.max())*1.07) 
            ax.set_ylim((y_vals.min())*1.15, (y_vals.max())*1.15)
            #ax.set_xlim(-7000000, 7000000)
            #ax.set_ylim(-7000000, 7000000) 
            plt.plot(Re_0[0], Re_0[0], "ko",c= "b", label = "Earth" )
            plt.plot(Rm_0[0], Rm_0[1], "ko", c = "r", label = "Moon")
            plt.plot([], [], c = "darkblue", label = "Rocket")
            plt.title("2D Simulation of Rocket's orbit around Earth and Moon")

        else: 
        #i.e part c and d
            
            frame_factor = 50

            ax.set_xlim((x2_vals.min())*1.15, (x2_vals.max())*1.15) 
            ax.set_ylim((y2_vals.min())*1.15, (y2_vals.max())*1.15)
            plt.plot([], [], c = "darkblue", label = "Earth")
            plt.plot([], [], c = "red", label = "Moon")
            plt.title("2D Simulation of Earth and Moon's orbits")

            if Sun == True: 

                if zoom == True: 

                    frame_factor = 5
                    ax.set_xlim(0, 1.5e10)
                    ax.set_ylim(0, 7e10)   

                else: 

                    frame_factor = 30
                    ax.set_xlim(-(abs(x2_vals.min()))*100, (x2_vals.max())*1.05) 
                    ax.set_ylim((y2_vals.min())*1.15, (y2_vals.max())*1.15)
                    plt.gca().set_aspect('equal', adjustable='box')

                plt.plot(Rs_0[0], Rs_0[1], "ko",  c = "y", label = "Sun")
    
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()

    line, = ax.plot([], [])
    line2, = ax.plot([], [])
    circle, = ax.plot([], [], marker='o', markersize=6, color='darkblue')
    circle2, = ax.plot([], [], marker='o', markersize=6, color='red')

    x_limits = plt.gca().get_xlim()
    y_limits = plt.gca().get_ylim()
    time_text= ax.text(0.95*x_limits[0], 0.9*y_limits[1], "Time: 0 Hours, 0 Days", fontsize = "x-small")

    def animated_plot(): 
        def animate(frame):

            if dt < 1:
                seconds = frame*frame_factor/dt
            else: 
                seconds = frame*frame_factor*dt

            hours =seconds/3600
            days = seconds/86400
            
            
            line.set_data(x_vals[:frame*frame_factor], y_vals[:frame*frame_factor])
            line2.set_data(x2_vals[:frame*frame_factor], y2_vals[:frame*frame_factor])
            circle.set_data(x_vals[(frame)*frame_factor], y_vals[(frame)*frame_factor])
            time_text.set_text(f'Time: {(seconds):.2f} Seconds, {(hours):.2f} Hours, {(days):.2f} Days')
            if x2_vals != []:
                circle2.set_data(x2_vals[(frame)*frame_factor], y2_vals[(frame)*frame_factor])
                return line, line2, circle, circle2, time_text,
            else: 
                return line, circle, time_text,
            

        anim = animation.FuncAnimation(fig, animate, frames = len(x_vals), interval = 1, blit = True )
        return anim
    
    anim = animated_plot()

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.show()

def plot_energy(t, te, ke, pe):
    """
    Plots the toal, kinetic, and potential energy.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(t, te, c = "r", label = "Total Energy")
    plt.plot(t, ke, c = "g", label = "Kinetic Energy")
    plt.plot(t, pe, c = "b", label = "Potential Energy")
    plt.title('Energy over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Total Energy (J)')
    plt.legend()
    plt.show()

def plot_error(  te_error, title, xlabel, x = [] ):
    """
    Plots a given error. 
    """
    if x == []:
        x = np.linspace(0, len(te_error), len(te_error))
    plt.figure(figsize=(8, 6))
    plt.plot(x, te_error)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(title)
    plt.show()

def plot_moon_crash(x_vals, y_vals,label, x2_vals, y2_vals, label2, x3_vals,  y3_vals, label3, x4_vals, y4_vals, label4):
    """
    Plots four different pairs of x-y values corresponding to different vy velocity values, and zooms in on the Sun to analyse the crashing or non-crashing. 
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlim(Rm_0[0]-7e6, Rm_0[0]+7e6) 
    ax.set_ylim(Rm_0[1]-7e6, Rm_0[1]+7e6)

    plt.plot(Rm_0[0], Rm_0[1], "ko", c = "r", label = "Moon")
    plt.plot(x_vals, y_vals, c = "b", label = label)
    plt.plot(x2_vals, y2_vals, c = "k", label = label2)
    plt.plot(x3_vals, y3_vals, c = "g", label = label3)
    plt.plot(x4_vals, y4_vals, c = "m", label = label4)
    plt.legend()
    plt.show()


#########################################################################################################################################################
#Main program

MyInput = '0'
while MyInput != 'q':
    try:
        MyInput = input('Enter a choice, "a", "b", "c", "d", or "q" to quit: ')
        if MyInput not in ['a', 'b', 'c', 'd', 'q']:
            raise ValueError("Invalid choice. Please enter a valid option.")
    except ValueError as e:
        print(f"Error: {e}")
        continue

##################################################################################################
#Part a
    if MyInput == 'a':
        print('You have chosen part (a)')
        
        quit = False
        while True: 
            shape = input('Type "c" if you want a circular orbit, "e" if you want an elliptical orbit, or "q" to quit: ')
            if shape == "c": 
                factor = 1
                break
            elif shape == "e": 
                factor = 0.5
                break

            elif shape =="q": 
                quit = True
                break
            
            else:
                print("That's not a valid option, try again")

        if quit == True: 
            break

        #Initial position (x, y, z) and velocity (vx, vy, vz) values for the rocket.
        x0 = -500000
        r_start_vals = [x0, 0, 0]
        v_start_vals = [0, circ_vel(x0)*factor, 0]
        
        dt = 0.1
        te_global = []
        mean_orbit_err = []

       
        print("Calculating the rocket's orbit around Earth...")
        x, y, z, te, ke, pe, te_error, global_error,  mean_orbit_error, period = RK4_iteration(r_start_vals, v_start_vals,  dt = dt, t_max = 5000, f = f, f_2 = f_2, part_a = True)
        
        te_global.append(global_error)
        mean_orbit_err.append(mean_orbit_error)
        
        period_hours = period/3600
        period_days = period/86400
        
        print(f"\nOrbital period: {(period):.2f} Seconds, i.e {(period_hours):.2f} Hours or {(period_days):.2f} Days.\n" )


        #plotting options
        while True: 
            p_input = input('Type "plot1" if you want to plot the 2D orbit, "plot2" if you want to plot its animation, Type "plot3" if you want to plot it in 3D, "plot4" if you want to plot the rocket\'s energy over time, "plot5" if you want to plot the Energy conservation error over time, "plot6" if you want to see the influence of dt on the error,  or press Enter to skip: ')
            
            if p_input == "plot1":

                plot_2Dorbit(x, y)
                
            if p_input == "plot2":

                plot_2Danimated_orbit(x, y, dt)
                print("\nIf a blank plot appears, please rerun - matplotlib does that sometimes with the animation. Or, if you have VSCode, open it in VSCode\n")


            if p_input == "plot3":

                plot_3Dorbit(x, y, z)

            if p_input == "plot4": 

                t1 = np.linspace(0, len(te), len(te))
                plot_energy(t1[:1000], te[:1000], ke[:1000], pe[:1000])
            
            if p_input == "plot5":    

                plot_error(te_error, 'Energy conservation percentage error over time', xlabel = "Time (s)")
                plot_error(te_error[:1000], 'Zoomed in energy conservation percentage error over time', xlabel = "Time (s)")

            
            if p_input == "plot6":  

                dt_list = [0.1, 0.5, 1, 4, 5, 6, 7]
                for dt_ in dt_list[1:]:
                    x, y, z, te, ke, pe, te_error, global_error, mean_orbit_error, period = RK4_iteration(r_start_vals, v_start_vals,  dt = dt_, t_max = 5000, f = f, f_2 = f_2)
                    te_global.append(global_error)
                    mean_orbit_err.append(mean_orbit_error)
                
                plot_error(te_global, "Global energy conservation percentage error", "Timestep dt", dt_list)
                plot_error(mean_orbit_err, "Mean orbit deviation percentage error", "Timestep dt", dt_list)

            elif p_input == "":
                break
            else:
                print("That's not a valid option, try again")

    elif MyInput == 'b':
            
        print('You have chosen part (b)')
    
        quit = False
        while True: 
            shape = input('Type "8" if you want an eight-shaped slingshot, "0" if you want an elliptical slingshot, or "q" to quit: ')
            if shape == "8": 
                vy = 10555.75
                #vy = 10566 #moon crash
                #vy = 10560 #Earth crash
                break

            elif shape == "0": 
                vy = 10572
                #vy = 10570 #earth crash
                break

            elif shape =="q": 
                quit = True
                break
            
            else:
                print("That's not a valid option, try again")

        if quit == True: 
            break

        #Initial position (x, y, z) and velocity (vx, vy, vz) values for the rocket.
        x0 = -7000000
        r_start_vals = [x0, 0, 0]
        v_start_vals = [0, vy, 0]

        dt = 100
        te_global = []
        

        
        
        print("Calculating the rocket's orbit...")
        x, y, z, te, ke, pe, te_error, global_error,  mean_orbit_error, period  = RK4_iteration(r_start_vals, v_start_vals, dt = dt, t_max = 3000000, f = f, f_2 = f_2_with_moon, part_b=True)

        period_hours = period/3600
        period_days = period/86400
        print(f"\nReturn journey: {(period):.2f} Seconds, i.e {(period_hours):.2f} Hours or {(period_days):.2f} Days.\n" )
        
        #plotting options
        while True: 
            p_input = input('Type "plot1" if you want to plot the 2D orbit, "plot2" if you want to plot its animation, Type "plot3" if you want to plot it in 3D, "plot4" if you want to plot the rocket\'s energy over time, "plot5" if you want to plot the Energy conservation error over time, "plot6" if you want to see the influence of dt on the error, "plot7" to see the limits of a mooon crash or press Enter to skip: ')
            
            if p_input == "plot1":
                plot_2Dorbit(x, y, moon = True)
                
            if p_input == "plot2":
                plot_2Danimated_orbit(x, y, dt, moon = True)
                print("\nIf a blank plot appears, please rerun - matplotlib does that sometimes with the animation. Or, if you have VSCode, open it in VSCode\n")

            if p_input == "plot3":
                plot_3Dorbit(x, y, z, moon = True)

            if p_input == "plot4": 
                t = np.linspace(0, len(te), len(te))
                plot_energy(t, te, ke, pe)


            if p_input == "plot5":    

                plot_error(te_error, 'Energy conservation percentage error over time', xlabel = "Time (s)")
                plot_error(te_error[:1000], 'Zoomed in energy conservation percentage error over time', xlabel = "Time (s)")

            
            if p_input == "plot6":  

                dt_list = [25, 50, 100, 150, 200, 250]
                for dt_ in dt_list:
                    x, y, z, te, ke, pe, te_error, global_error, mean_orbit_error, period  = RK4_iteration(r_start_vals, v_start_vals, dt = dt_, t_max = 1000000, f = f, f_2 = f_2_with_moon, part_b=True)
                    te_global.append(global_error)
            
                
                plot_error(te_global, "Global energy conservation percentage error", "Timestep dt", dt_list)
                

            if p_input == "plot7":
                vy = 10566
                v_start_vals = [0, vy, 0]
                x, y, z, te, ke, pe, te_error, global_error, mean_orbit_error, period  = RK4_iteration(r_start_vals, v_start_vals, dt = dt, t_max = 1000000, f = f, f_2 = f_2_with_moon, part_b=True)

                vy2 = 10567
                v_start_vals = [0, vy2, 0]
                x2, y2, z, te, ke, pe, te_error, global_error, mean_orbit_error, period  = RK4_iteration(r_start_vals, v_start_vals, dt = dt, t_max = 1000000, f = f, f_2 = f_2_with_moon, part_b=True)

                vy3 = 10568
                v_start_vals = [0, vy3, 0]
                x3, y3, z, te, ke, pe, te_error, global_error, mean_orbit_error, period  = RK4_iteration(r_start_vals, v_start_vals, dt = dt, t_max = 1000000, f = f, f_2 = f_2_with_moon, part_b=True)

                vy4 = 10569
                v_start_vals = [0, vy4, 0]
                x4, y4, z, te, ke, pe, te_error, global_error, mean_orbit_error, period  = RK4_iteration(r_start_vals, v_start_vals, dt = dt, t_max = 1000000, f = f, f_2 = f_2_with_moon, part_b=True)
            
                plot_moon_crash(x, y, "vy: " + str(vy) + " m/s", x2, y2, "vy: " + str(vy2) + " m/s", x3, y3, "vy: " + str(vy3) + " m/s", x4, y4, "vy: " + str(vy4) + " m/s")

            elif p_input == "":
                break
            
            else:
                print("That's not a valid option, try again")

    elif MyInput == 'c':
                
        print('You have chosen part (c)')


        #Initial position (x, y, z) and velocity (vx, vy, vz) values for the rocket.
        quit = False
        while True: 
            e = input("Type 's' if you want the Earth to be stationary, type 'm' if you want it to move, or type 'q' to quit ")
            if e == "s": 
                vy_earth = 0
                break

            elif e == "m":
                vy_earth = 460
                break
            
            elif e =="q": 
                quit = True
                break

            else:
                print("That's not a valid option, try again") 

        if quit == True: 
            break

        quit = False
        while True: 
            a = input("Type 'Me' if you want the Moon to have the same mass as the Earth, type 'Mm' if you want them to be different, or type 'q' to quit ")
            if a == "Me": 
                Mm = Me
                break
            elif a == "Mm":
                break
            
            elif a =="q": 
                quit = True
                break
            else:
                print("That's not a valid option, try again") 

        if quit == True: 
            break

        earth_r_start_vals = [0, 0, 0]
        earth_v_start_vals = [0, vy_earth, 0]

        x0_moon = Rm_0[0]
        moon_r_start_vals=[x0_moon, 0, 0,] 
        moon_v_start_vals=[0, vy_earth + circ_vel(x0_moon), 0]

        dt = 100
        
        print("Calculating Earth and Moon's orbits...")
        xe, ye, ze, xm, ym, zm, te, ke, pe, period = RK4_iteration(earth_r_start_vals, earth_v_start_vals, dt = dt,  t_max= 10000000, f = f, f_2 = f_2_e, f_3 = f_2_m, r2_start_vals = moon_r_start_vals, v2_start_vals= moon_v_start_vals , part_c= True)
        
        period_hours = period/3600
        period_days = period/86400
        if e  == "s": 
            print(f"\nOrbital period of the Moon: {(period):.2f} Seconds, i.e {(period_hours):.2f} Hours or {(period_days):.2f} Days.\n" )

        #plotting options
        while True: 
            p_input = input('Type "plot1" if you want to plot the 2D orbit, "plot2" if you want to plot its animation,  "plot3" if you want to plot the orbit in 3D or press Enter to skip: ')
            
            if p_input == "plot1":
                plot_2Dorbit(xe, ye, xm, ym, moon = True)
                
            if p_input == "plot2":
                plot_2Danimated_orbit(xe, ye, dt, xm, ym, moon = True)
                print("\nIf a blank plot appears, please rerun - matplotlib does that sometimes with the animation. Or, if you have VSCode, open it in VSCode\n")

            if p_input == "plot3":
                plot_3Dorbit(xe, ye, ze, xm, ym, zm, moon= True)

            

            elif p_input == "":
                break
            else:
                print("That's not a valid option, try again")

    elif MyInput == 'd':
                    
        print('You have chosen part (d)')

        #Initial position (x, y, z) and velocity (vx, vy, vz) values for the rocket.
        vy_earth = 30e3
        earth_r_start_vals = [0, 0, 0]
        earth_v_start_vals = [0, vy_earth, 0]

        x0_moon = Rm_0[0]
        moon_r_start_vals=[x0_moon, 0, 0,] 
        moon_v_start_vals=[0, vy_earth + circ_vel(x0_moon), 0]

        dt = 1000
        
        print("Calculating Earth and Moon and Sun's orbits...")
        xe, ye, ze, xm, ym, zm, te, ke, pe, period= RK4_iteration(earth_r_start_vals, earth_v_start_vals, dt = dt, t_max= (1.04*3.154e7), f = f, f_2 = f_2_e_with_Sun, f_3 = f_2_m_with_Sun,  r2_start_vals=moon_r_start_vals, v2_start_vals=moon_v_start_vals, part_c= True, part_d = True)
        
        period_hours = period/3600
        period_days = period/86400
        print(f"\nOrbital period of the Earth: {(period):.2f} Seconds, i.e {(period_hours):.2f} Hours or {(period_days):.2f} Days.\n" )

        #plotting options
        while True: 
            p_input = input('Type "plot1" if you want to plot the 2D orbit, "plot2" if you want to plot its animation, "plot3" if you want to zoom in, "plot4" if you want the animated zoom-in, "plot5" if you want to plot the orbit in 3D or press Enter to skip: ')
            
            if p_input == "plot1":
                plot_2Dorbit(xe, ye, xm, ym, moon = True, Sun = True)
                
            if p_input == "plot2":
                plot_2Danimated_orbit(xe, ye, dt, xm, ym, moon = True, Sun = True)
                print("If a blank plot appears, please rerun - matplotlib does that sometimes with the animation.")

            if p_input == "plot3":
                plot_2Dorbit(xe, ye, xm, ym, moon = True, Sun = True, zoom = True)

            if p_input == "plot4":
                plot_2Danimated_orbit(xe, ye, dt, xm, ym, moon = True, Sun = True, zoom = True)
                print("If a blank plot appears, please rerun - matplotlib does that sometimes with the animation.")
            
            if p_input == "plot5":
                plot_3Dorbit(xe, ye, ze, xm, ym, zm, moon= True, Sun = True)


            elif p_input == "":
                break
            else:
                print("That's not a valid option, try again")


    