import torch 
import matplotlib.pyplot as plt 
from collections import deque 
from tqdm import tqdm
from scipy.integrate import odeint
import numpy as np 



  


def derive(y,t,m1,m2,l_1,l_2,g):
    #formula's from https://www.myphysicslab.com/pendulum/double-pendulum-en.html
    theta1, dot_theta1, theta2, dot_theta2 = y

    dotdot_theta1 =(-g*(2*m1+m2)*np.sin(theta1)-m2*g*np.sin(theta1-2*theta2)-2*np.sin(theta1-theta2)*m2*((dot_theta2**2)*l_2+(dot_theta1**2)*l_1*np.cos(theta1-theta2)))/(l_1*(2*m1+m2-m2*np.cos(2*theta1-2*theta2)))
    dotdot_theta2 = (2*np.sin(theta1-theta2)*((dot_theta1**2)*l_1*(m1+m2)+g*(m1+m2)*np.cos(theta1)+(dot_theta2**2)*l_2*m2*np.cos(theta1-theta2)))/(l_2*(2*m1+m2-m2*np.cos(2*theta1-2*theta2)))
    return dot_theta1, dotdot_theta1 , dot_theta2, dotdot_theta2 #return derivative for all y 

def coordinates(y,l_1,l_2):
    theta1,dot_theta1,theta2,dot_theta2 = y.T

    x1 =  l_1 * np.sin(theta1)
    y1 = -l_1* np.cos(theta1)
    
    x2 = x1 + l_2*np.sin(theta2)
    y2 = y1 - l_2*np.cos(theta2) 

    return (x1,y1,x2,y2)

def dot_coordinates(y,l_1,l_2):
    theta1,dot_theta1,theta2,dot_theta2 = y.T
    dot_x1 = l_1*dot_theta1*np.cos(theta1)
    dot_y1 = l_1*dot_theta1*np.sin(theta1) 
    dot_x2 = dot_x1 + l_2*dot_theta2*np.cos(theta2)
    dot_y2 = dot_y1 + l_2*dot_theta2*np.sin(theta2)

    return (dot_x1,dot_y1,dot_x2,dot_y2)


def kinetic_energy(y,l_1,l_2,m1,m2):
    dot_x1,dot_y1,dot_x2,dot_y2 = dot_coordinates(y,l_1,l_2)
    return (1/2)*m1*(dot_x1**2 + dot_y1**2) + (1/2)*m2*(dot_x2**2+dot_y2**2)         

def potential_energy(y,l_1,l_2,m1,m2,g):
    x1,y1,x2,y2 = coordinates(y,l_1,l_2)
    return m1*g*y1 + m2*g*y2

def legrandian(y,l_1,l_2,m1,m2,g):
    return kinetic_energy(y,l_1,l_2,m1,m2) - potential_energy(y,l_1,l_2,m1,m2,g) 

def total_energy(y,l_1,l_2,m1,m2,g):
    return kinetic_energy(y,l_1,l_2,m1,m2) + potential_energy(y,l_1,l_2,m1,m2,g) 

if __name__ == "__main__":
    #hyperparam
    m1 = 1
    m2 = 1
    l_1 = 1
    l_2 = 1
    g = 9.81
    time_steps = 1000
    end_time = 100
    buffer_size = 200
    #initial 
    y0 = np.array([3,0,3,0]) #theta1,dot_theta1,theta2,dot_theta2
    t = np.linspace(0,end_time,time_steps)

    #calculate
    y = odeint(derive,y0,t,args=(m1,m2,l_1,l_2,g)) #theta1,dot_theta1,theta2,dot_theta2
    
    x1,y1,x2,y2 = coordinates(y,l_1,l_2)
    dot_x1,dot_y1,dot_x2,dot_y2 = dot_coordinates(y,l_1,l_2)
    K = kinetic_energy(y,l_1,l_2,m1,m2)
    P = potential_energy(y,l_1,l_2,m1,m2,g)
    E = total_energy(y,l_1,l_2,m1,m2,g)

  
    #visualize
    fig,axis = plt.subplots(2)
    kenetic_energy_buffer = deque(maxlen=buffer_size)
    potential_energy_buffer = deque(maxlen=buffer_size)
    total_energy_buffer = deque(maxlen=buffer_size)

    plt.ion()
    for i in tqdm(range(len(y)),ascii=True):
        kenetic_energy_buffer.append(K[i])
        potential_energy_buffer.append(P[i])
        total_energy_buffer.append(E[i])
            
        axis[0].cla()
        axis[1].cla()
        axis[0].set_xlim([-2,2])
        axis[0].set_ylim([-2,2])
        axis[0].set_box_aspect(1)
        axis[0].plot((0, x1[i]), (0, y1[i]),marker="o")            
        axis[0].plot((x1[i],x2[i]),(y1[i],y2[i]),marker="o")
        axis[1].plot(torch.arange(len(kenetic_energy_buffer)),kenetic_energy_buffer,color="r")
        axis[1].plot(torch.arange(len(potential_energy_buffer)),potential_energy_buffer,color="g")
        axis[1].plot(torch.arange(len(total_energy_buffer)),total_energy_buffer,color="b")
        plt.pause(0.01)    
