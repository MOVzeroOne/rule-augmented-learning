import torch 
import matplotlib.pyplot as plt 
from collections import deque 
from tqdm import tqdm
from scipy.integrate import odeint
import numpy as np 


class pendulum():
    def __init__(self,m1 = 1,m2 = 1,l_1 = 1,l_2 = 1,g = 9.81,time_steps = 1000,end_time = 100):
        self.m1 = m1
        self.m2 = m2
        self.l_1 = l_1
        self.l_2 = l_2
        self.g = g
        self.time_steps = time_steps
        self.end_time = end_time
        self.t = torch.linspace(0,self.end_time,self.time_steps)

    def derive(self,y,t,m1,m2,l_1,l_2,g):
        #formula's from https://www.myphysicslab.com/pendulum/double-pendulum-en.html
        theta1, dot_theta1, theta2, dot_theta2 = y

        dotdot_theta1 =(-g*(2*m1+m2)*np.sin(theta1)-m2*g*np.sin(theta1-2*theta2)-2*np.sin(theta1-theta2)*m2*((dot_theta2**2)*l_2+(dot_theta1**2)*l_1*np.cos(theta1-theta2)))/(l_1*(2*m1+m2-m2*np.cos(2*theta1-2*theta2)))
        dotdot_theta2 = (2*np.sin(theta1-theta2)*((dot_theta1**2)*l_1*(m1+m2)+g*(m1+m2)*np.cos(theta1)+(dot_theta2**2)*l_2*m2*np.cos(theta1-theta2)))/(l_2*(2*m1+m2-m2*np.cos(2*theta1-2*theta2)))
        return dot_theta1, dotdot_theta1 , dot_theta2, dotdot_theta2 #return derivative for all y 

    def sample(self,y0:torch.tensor):
        """
        y0: torch.array([theta1,dot_theta1,theta2,dot_theta2])
        y0 can also be a pytorch array as long as it doesnt require grads
        """
        y = torch.tensor(odeint(self.derive,y0,self.t,args=(self.m1,self.m2,self.l_1,self.l_2,self.g)),dtype=torch.float) #theta1,dot_theta1,theta2,dot_theta2  
        
        input = y[:-1,:]
        target =  y[1:,:]
        
        return input,target

    def coordinates(self,y):
        theta1,dot_theta1,theta2,dot_theta2 = y.T

        x1 =  self.l_1 * torch.sin(theta1)
        y1 = -self.l_1* torch.cos(theta1)
        
        x2 = x1 + self.l_2*torch.sin(theta2)
        y2 = y1 - self.l_2*torch.cos(theta2) 

        return (x1,y1,x2,y2)

    def dot_coordinates(self,y):
        theta1,dot_theta1,theta2,dot_theta2 = y.T
        dot_x1 = self.l_1*dot_theta1*torch.cos(theta1)
        dot_y1 = self.l_1*dot_theta1*torch.sin(theta1) 
        dot_x2 = dot_x1 + self.l_2*dot_theta2*torch.cos(theta2)
        dot_y2 = dot_y1 + self.l_2*dot_theta2*torch.sin(theta2)

        return (dot_x1,dot_y1,dot_x2,dot_y2)

    def kinetic_energy(self,y):
        dot_x1,dot_y1,dot_x2,dot_y2 = self.dot_coordinates(y)
        return (1/2)*self.m1*(dot_x1**2 + dot_y1**2) + (1/2)*self.m2*(dot_x2**2+dot_y2**2)         

    def potential_energy(self,y):
        x1,y1,x2,y2 = self.coordinates(y)
        return self.m1*self.g*y1 + self.m2*self.g*y2

    def legrandian(self,y):
        return self.kinetic_energy(y) - self.potential_energy(y) 

    def total_energy(self,y):
        return self.kinetic_energy(y) + self.potential_energy(y) 

    def info(self,y:torch.tensor):
        """
        y: torch.tensor([[theta1,dot_theta1,theta2,dot_theta2],[theta1,dot_theta1,theta2,dot_theta2],..])
        where each [theta1,dot_theta1,theta2,dot_theta2] represents a different time step
        """
        x1,y1,x2,y2 = self.coordinates(y) 
        dot_x1,dot_y1,dot_x2,dot_y2 = self.dot_coordinates(y)
        K = self.kinetic_energy(y)
        P = self.potential_energy(y)
        E = self.total_energy(y)

        return x1,y1,x2,y2,dot_x1,dot_y1,dot_x2,dot_y2,K,P,E 

class dampend_pendulum(pendulum):
    def __init__(self,dampening_l1=1,dampening_l2=1,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dampening_l1 = dampening_l1
        self.dampening_l2 = dampening_l2
    
    def derive(self,y,t,m1,m2,l_1,l_2,g):
        #formula's from https://www.myphysicslab.com/pendulum/double-pendulum-en.html
        #with naive dampening
        theta1, dot_theta1, theta2, dot_theta2 = y

        dotdot_theta1 =(-g*(2*m1+m2)*np.sin(theta1)-m2*g*np.sin(theta1-2*theta2)-2*np.sin(theta1-theta2)*m2*((dot_theta2**2)*l_2+(dot_theta1**2)*l_1*np.cos(theta1-theta2)))/(l_1*(2*m1+m2-m2*np.cos(2*theta1-2*theta2)))
        dotdot_theta2 = (2*np.sin(theta1-theta2)*((dot_theta1**2)*l_1*(m1+m2)+g*(m1+m2)*np.cos(theta1)+(dot_theta2**2)*l_2*m2*np.cos(theta1-theta2)))/(l_2*(2*m1+m2-m2*np.cos(2*theta1-2*theta2)))
        return dot_theta1, dotdot_theta1-dot_theta1*self.dampening_l1 , dot_theta2, dotdot_theta2-dot_theta2*self.dampening_l2 #return derivative for all y 

if __name__ == "__main__":
    """
    testing with visualization 
    """
    buffer_size= 500
    #pen = pendulum()
    pen = dampend_pendulum(0.9,0.3)
    y,_ = pen.sample(torch.tensor([2,0,3,0],dtype=torch.float))

    x1,y1,x2,y2,dot_x1,dot_y1,dot_x2,dot_y2,K,P,E = pen.info(y)

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