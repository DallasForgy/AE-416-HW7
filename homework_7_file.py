########################################################################
#
#  .py - Create surface panelization for a NACA4 Series Airfoil
#
#
#  Input list: 
#  N/A
#
#  Output list:
#  cl       -  Lift Coefficient  
#  cd       -  Drag Coefficient  
#  cm       -  Moment Coefficient  
#  x        -  Vector of x coordinates of the surface nodes
#  y        -  Vector of y coordinates of the surface nodes
#  cp       -  Pressure Coefficient Distribution
#
#  Written by: Matthew Clarke                                          #
#              Department of Aerospace Engineering                     #
#              University of Illinois, Urbana-Champaign                #
#              maclarke@illinois.edu                                   #
#                                                                      #
#  Last Modified: Wed Aug 30 2023
########################################################################

import matplotlib.pyplot as plt 
from os.path import dirname, realpath, sep, pardir
import sys
import numpy as np

sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")  

from hess_smith             import hess_smith
from naca_4series_generator import naca_4series_generator  
from solve_thwaites_BL      import solve_thwaites_BL
from solve_heads_BL         import solve_heads_BL
from michel_criterion       import michel_criterion  

def main():  
    naca4           =  '3310'
    alpha           = 3.0
    npanel          = 200
    Re_L            = 5e6

    # ---------------------------------------------------------------------------   
    # STEP 2: generate airfoil surface panelization 
    # ---------------------------------------------------------------------------
    # NACA 4 Series Generator  
    x, y = naca_4series_generator(naca4, npanel)
    # Call Hess Smith  
    cl_wo, cd_wo, cm_wo, cp_wo, xbar_wo, ybar_wo, vt, cos_theta, sin_theta = hess_smith(x, y, alpha)
    # use tangential velocity to determine where upper and lower surfaces  
    stagnation_point = np.argmin(np.abs(vt))
    # slice arrays for x,y and vt. Note that 
    # 1) on the botton surface, you have to reverse the direction 
    #    e.g. x_bot            = x[:stagnation_point][::-1]
    x_bot = x[:stagnation_point][::-1]
    y_bot = y[:stagnation_point][::-1]
    x_top = x[stagnation_point:][::1]
    y_top = y[stagnation_point:][::1]
    # 2) on the bottom surface you have to multiply the velocity distribution by -1.
    #    e.g. Ve_bot           = -vt[:stagnation_point][::-1]   
    Ve_bot = -vt[:stagnation_point][::-1]
    Ve_top = vt[stagnation_point:][::1]
    
    
    # create data structures for the boundary layer properties using "np.zeros(len(Ve_bot))". Using np.cumsum to compute the arc length, s, along the surface of the airfoil  
    x_bl_bot = np.zeros(len(Ve_bot))
    x_bl_top = np.zeros(len(Ve_top))
    for i in range(1, len(Ve_bot)):
        x_bl_bot[i] = x_bl_bot[i-1] + np.sqrt((x_bot[i]-x_bot[i-1])**2 + (y_bot[i]-y_bot[i-1])**2)
    for i in range(1, len(Ve_top)):
        x_bl_top[i] = x_bl_top[i-1] + np.sqrt((x_top[i]-x_top[i-1])**2 + (y_top[i]-y_top[i-1])**2)

    # create data structures for dVe on the upper and lower surface
    x_bl_top[0] = 1E-6
    x_bl_bot[0] = 1E-6
    dVe_bot = np.gradient(Ve_bot, x_bl_bot)
    dVe_top = np.gradient(Ve_top, x_bl_top)
    
    
    # compute 
    # e.g. nu:  nu =  1 /Re_L
    nu = 1 / Re_L
    
    # initialize the first value in the arclenght to a non-zero value i.e. 1E-6. We did this in HW 4. 
    # e.g. x_bl_top[0] = 1E-6 
    
    # --------------------------------------------------------------------------------------------------------------------------------------------    
    # Top Surface 
    # --------------------------------------------------------------------------------------------------------------------------------------------
    # compute theta_0 on the top surface  
    theta_0 = 0.664*x_bl_bot[0] / np.sqrt(Re_L*x_bl_bot[0])
    
    # run the thwaites code for the upper surface . Note that l is the last value:  x_bl_top[-1] 
    x_i_lam_top, H_lam_top,delta_star_lam_top,delta_lam_top,cf_lam_top,theta_lam_top,Re_x_lam_top,Re_theta_lam_top = solve_thwaites_BL(nu, x_bl_top, Re_L, x_bl_top, Ve_top, dVe_top, theta_0)
    # Find transition points for top and bottom surface 
    # i.e. transition_index_top,transition_flag_top = michel_criterion(Re_theta_lam_top,Re_x_lam_top) 
    transition_index_top,transition_flag_top = michel_criterion(Re_theta_lam_top,Re_x_lam_top) 
    if transition_flag_top:
        print(f'There is a transition on the top airfoil at {round(x_i_lam_top[transition_index_top]/x_i_lam_top[-1], 3)}')
        # Get initial conditons for turbulent boundary layer
        x_i_trans = x_i_lam_top[transition_index_top:]
        H_trans = H_lam_top[transition_index_top]
        delta_star_trans = delta_star_lam_top[transition_index_top]
        delta_trans = delta_lam_top[transition_index_top]
        cf_trans = cf_lam_top[transition_index_top]
        theta_trans = theta_lam_top[transition_index_top]
        Re_x_trans = Re_x_lam_top[transition_index_top]
        Re_theta_trans = Re_theta_lam_top[transition_index_top]
        Ve_i_trans = Ve_top[transition_index_top:]
        dVe_i_trans = dVe_top[transition_index_top:]
        # run the head code on the top surface  
        x_i_turb, H_turb,delta_star_turb,delta_turb,cf_turb,theta_turb,Re_x_turb,Re_theta_turb = solve_heads_BL(delta_trans,theta_trans,delta_star_trans,cf_trans,H_trans,Re_L,x_i_trans,Ve_i_trans,dVe_i_trans,x_bl_top[-1],nu)
        # Concatenate vectors
        x_i_top = np.concatenate([x_i_lam_top[:transition_index_top], x_i_turb])
        H_top = np.concatenate([H_lam_top[:transition_index_top], H_turb])
        delta_star_top = np.concatenate([delta_star_lam_top[:transition_index_top], delta_star_turb])
        delta_top = np.concatenate([delta_lam_top[:transition_index_top], delta_turb])
        cf_top = np.concatenate([cf_lam_top[:transition_index_top], cf_turb])
        theta_top = np.concatenate([theta_lam_top[:transition_index_top], theta_turb])
        Re_x_top = np.concatenate([Re_x_lam_top[:transition_index_top], Re_x_turb])
        Re_theta_top = np.concatenate([Re_theta_lam_top[:transition_index_top], Re_theta_turb])
    else: 
        print('There is no transition on the top of the airfoil.')
        x_i_top = x_i_lam_top
        H_top = H_lam_top
        delta_star_top = delta_star_lam_top
        delta_top = delta_lam_top
        cf_top = cf_lam_top
        theta_top = theta_lam_top
        Re_x_top = Re_x_lam_top
        Re_theta_top = Re_theta_lam_top
       
    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Bottom Surface
    # --------------------------------------------------------------------------------------------------------------------------------------------
    # compute theta_0 on the bottom surface
    theta_0 = 0.664*x_bl_bot[0] / np.sqrt(Re_L*x_bl_bot[0])

    # run the thwaites code for the bottom surface. Note that l is the last value: x_bl_bot[-1]
    x_i_lam_bot, H_lam_bot,delta_star_lam_bot,delta_lam_bot,cf_lam_bot,theta_lam_bot,Re_x_lam_bot,Re_theta_lam_bot = solve_thwaites_BL(nu, x_bl_bot, Re_L, x_bl_bot, Ve_bot, dVe_bot, theta_0)
    # Find transition points for bottom surface
    # i.e. transition_index_bot,transition_flag_bot = michel_criterion(Re_theta_lam_bot,Re_x_lam_bot)
    transition_index_bot,transition_flag_bot = michel_criterion(Re_theta_lam_bot,Re_x_lam_bot)
    if transition_flag_bot:
        print(f'There is a transition on the bottom airfoil at {round(x_i_lam_bot[transition_index_bot]/x_i_lam_bot[-1], 3)}')
        # Get initial conditons for turbulent boundary layer
        x_i_trans = x_i_lam_bot[transition_index_bot:]
        H_trans = H_lam_bot[transition_index_bot]
        delta_star_trans = delta_star_lam_bot[transition_index_bot]
        delta_trans = delta_lam_bot[transition_index_bot]
        cf_trans = cf_lam_bot[transition_index_bot]
        theta_trans = theta_lam_bot[transition_index_bot]
        Re_x_trans = Re_x_lam_bot[transition_index_bot]
        Re_theta_trans = Re_theta_lam_bot[transition_index_bot]
        Ve_i_trans = Ve_bot[transition_index_bot:]
        dVe_i_trans = dVe_bot[transition_index_bot:]
        # run the head code on the bottom surface
        x_i_turb, H_turb,delta_star_turb,delta_turb,cf_turb,theta_turb,Re_x_turb,Re_theta_turb = solve_heads_BL(delta_trans,theta_trans,delta_star_trans,cf_trans,H_trans,Re_L,x_i_trans,Ve_i_trans,dVe_i_trans,x_bl_bot[-1],nu)
        # Concatenate vectors
        x_i_bot = np.concatenate([x_i_lam_bot[:transition_index_bot], x_i_turb])
        H_bot = np.concatenate([H_lam_bot[:transition_index_bot], H_turb])
        delta_star_bot = np.concatenate([delta_star_lam_bot[:transition_index_bot], delta_star_turb])
        delta_bot = np.concatenate([delta_lam_bot[:transition_index_bot], delta_turb])
        cf_bot = np.concatenate([cf_lam_bot[:transition_index_bot], cf_turb])
        theta_bot = np.concatenate([theta_lam_bot[:transition_index_bot], theta_turb])
        Re_x_bot = np.concatenate([Re_x_lam_bot[:transition_index_bot], Re_x_turb])
        Re_theta_bot = np.concatenate([Re_theta_lam_bot[:transition_index_bot], Re_theta_turb])
    else:
        print('There is no transition on the bottom of the airfoil.')
        x_i_bot = x_i_lam_bot
        H_bot = H_lam_bot
        delta_star_bot = delta_star_lam_bot
        delta_bot = delta_lam_bot
        cf_bot = cf_lam_bot
        theta_bot = theta_lam_bot
        Re_x_bot = Re_x_lam_bot
        Re_theta_bot = Re_theta_lam_bot

        
        
    # ------------------------------------------------------------------------------------------------------
    # Compute effective surface of airfoil with boundary layer  
    # ------------------------------------------------------------------------------------------------------
    # concatenate the bottom and the top. Remember to "flip" back the bottom surface using [::-1].
    # e.g. DELTA          = np.concatenate([delta_bot[::-1],delta_top])
    DELTA = np.concatenate([delta_bot[::-1], delta_top])
        
    
    # MASK any nans 
    mask           = np.isnan(DELTA)
    DELTA[mask]    = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), DELTA[~mask])
    
    
    # update surface of airfoil with boudary layer
    x_bar = (x[:-1] + x[1:])/2
    x_bar_new = x_bar + cos_theta*DELTA
    y_bar = (y[:-1] + y[1:])/2
    y_bar_new = y_bar + sin_theta*DELTA

    
    # recompute the hess-smith using the updated x and y values
    panel_idx = np.arange(len(DELTA))
    node_idx  = np.linspace(0, len(DELTA) - 1, len(x))
    DELTA_nodes     = np.interp(node_idx, panel_idx, DELTA)
    cos_theta_nodes = np.interp(node_idx, panel_idx, cos_theta)
    sin_theta_nodes = np.interp(node_idx, panel_idx, sin_theta)
    x_new = x - sin_theta_nodes * DELTA_nodes
    y_new = y + cos_theta_nodes * DELTA_nodes
    cl_w, cd_w, cm_w, cp_w, xbar_w, ybar_w, vt_w, cos_theta_w, sin_theta_w = hess_smith(x_new, y_new, alpha)

    print(f'Without boundary layer: cl = {cl_wo:.4f}, cm = {cm_wo:.4f}')
    print(f'With boundary layer:    cl = {cl_w:.4f},  cm = {cm_w:.4f}')

    # plot results 
    plt.plot(xbar_w, cp_w, label='Boundary Layer Included')
    plt.plot(xbar_wo, cp_wo, label='Boundary Layer Excluded')
    plt.gca().invert_yaxis()
    plt.xlabel(r'$\bar x$')
    plt.ylabel(r'$c_p$')
    plt.legend()
    plt.savefig('Coefficient Of Pressure Graph')
    plt.show()

    plt.plot(x, y, label='Original Airfoil', linestyle='--', alpha=0.8)
    plt.plot(x_new, y_new, label='Airfoil with boundary layer', linestyle=':', alpha=0.8)
    plt.xlabel(r'$\bar x$')
    plt.ylabel(r'$\bar y$')
    plt.axis('equal')
    plt.legend()
    plt.savefig('Airfoil Shapes')
    plt.show()

    # Boundary layer property plots
    # Thicknesses on one plot
    plt.figure()
    plt.plot(x_bl_top, theta_top,      label=r'$\theta$ Top')
    plt.plot(x_bl_bot, theta_bot,      label=r'$\theta$ Bot')
    plt.plot(x_bl_top, delta_star_top, label=r'$\delta^*$ Top')
    plt.plot(x_bl_bot, delta_star_bot, label=r'$\delta^*$ Bot')
    plt.plot(x_bl_top, delta_top,      label=r'$\delta$ Top')
    plt.plot(x_bl_bot, delta_bot,      label=r'$\delta$ Bot')
    plt.xlabel(r'$s$')
    plt.ylabel('Thickness')
    plt.legend()
    plt.savefig('thicknesses')
    plt.show()

    bl_vars = [
        (H_top,          H_bot,          r'$H$',               'H'),
        (np.concatenate([np.array([cf_top[1]]),cf_top[1:]]),         np.concatenate([np.array([cf_bot[1]]),cf_bot[1:]]),         r'$c_f$',             'cf'),
        (Ve_top,         Ve_bot,         r'$V_e$',             'Ve'),
        (dVe_top,        dVe_bot,        r'$dV_e/dx$',         'dVe'),
        (Re_x_top,       Re_x_bot,       r'$Re_x$',            'Rex'),
        (Re_theta_top,   Re_theta_bot,   r'$Re_\theta$',       'Retheta'),
    ]

    for var_top, var_bot, ylabel, fname in bl_vars:
        plt.figure()
        plt.plot(x_bl_top/x_bl_top[-1], var_top, label='Top Surface')
        plt.plot(x_bl_bot/x_bl_bot[-1], var_bot, label='Bottom Surface')
        plt.xlabel(r'$x$')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(fname)
        plt.show()
    

    return  
    
if __name__ == '__main__':
    main()
    plt.show()

