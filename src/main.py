import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from enviroments import dampend_pendulum
from torch.distributions import Beta
from collections import deque 
from matplotlib.widgets import Slider 
from tqdm import tqdm 
import matplotlib.gridspec as gridspec

"""
based upon:
https://arxiv.org/pdf/2106.07804.pdf
https://ai.googleblog.com/2022/01/controlling-neural-networks-with-rule.html

 The conventional approach to implementing rules incorporates them by including them in the calculation of the loss. 
 There are three limitations of this approach that we aim to address: 
 (i) rule strength needs to be defined before learning (thus the trained model cannot operate flexibly based on how much the data satisfies the rule); 
 (ii) rule strength is not adaptable to target data at inference if there is any mismatch with the training setup; 
 and (iii) the rule-based objective needs to be differentiable with respect to learnable parameters (to enable learning from labeled data). 

"""

class mish(nn.Module):
    """
    activation funciton
    Mish: A Self Regularized Non-Monotonic Ativation Function (https://arxiv.org/abs/1908.08681)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x * torch.tanh(F.softplus(x))
    


class mlp(nn.Module):
    def __init__(self,struct:list,act_f:nn.Module = mish) -> None:
        super().__init__()
        self.struct = struct 
        self.act_f = act_f
        self.layers = self._build()
    
    def _build(self) -> nn.Sequential:
        layers = []
        for index,(i,j) in enumerate(zip(self.struct[:-1],self.struct[1:])):
            layers.append(nn.Linear(i,j))
            if(not (index == len(self.struct)-2)):#if not last layer
                layers.append(self.act_f())
        return nn.Sequential(*layers)

    def forward(self,x:torch.tensor) -> torch.tensor:
        return self.layers(x) 


class network(nn.Module):
    """
    rule_encoder,data_encoder, decision block parameters are structs:
    [num_neurons,num_neurons,etc..]
    where num_neurons is an integer
    """
    def __init__(self,rule_encoder,data_encoder,decision_block):
        super().__init__()
        self.rule_encoder = mlp(rule_encoder)
        self.decision_block = mlp(decision_block)
        self.data_encoder = mlp(data_encoder)

    def forward(self,x,alpha):
        x += 0.0001* torch.randn_like(x)
        h_rule = self.rule_encoder(x)
        h_data = self.data_encoder(x)

        h = torch.cat((h_rule*(alpha),h_data*(1-alpha)),dim=1)
        
        return self.decision_block(h)


def energy_loss(y_pred,x_input,pendulum):
    """
    under the assumption that there is no batch dimension (we only have a single sequence)
    """
    energy_t0 = pendulum.total_energy(x_input) #t0
    energy_t1 = pendulum.total_energy(y_pred)

    loss = nn.ReLU()(energy_t1-energy_t0) #energy of the previous step is equal or lower. 
    return loss
    

class visualizer():
    def __init__(self,network,pendulum,y0=torch.tensor([2,0,3,0],dtype=torch.float),buffer_size=500,initial_alpha=0.5,alpha_min=0,alpha_max=1):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha = initial_alpha
        self.buffer_size = buffer_size
        self.y0 = torch.cat((torch.rand(1)*torch.pi*2,torch.randn(1)*0.1,torch.rand(1)*torch.pi*2,torch.randn(1)*0.1),dim=0)
        self.network = network
        self.pendulum = pendulum

        plt.ion()
        self.setup()
        self.run()

    def run(self):
        self.calculations()
        with torch.no_grad():
            while(True):
                self.t = self.t % len(self.y_pendulum) 
                self.kenetic_energy_buffer_pendulum.append(self.K_pen[self.t])
                self.potential_energy_buffer_pendulum.append(self.P_pen[self.t])
                self.total_energy_buffer_pendulum.append(self.E_pen[self.t])

                self.kenetic_energy_buffer_network.append(self.K_net[self.t])
                self.potential_energy_buffer_network.append(self.P_net[self.t])
                self.total_energy_buffer_network.append(self.E_net[self.t])
                self.energy0_buffer.append(self.E0[self.t])
                self.energy1_buffer.append(self.E_net[self.t])

                self.axis[0].cla()
                self.axis[1].cla()
                self.axis[2].cla()
                self.axis[3].cla()
                self.axis[0].set_xlim([-2,2])
                self.axis[0].set_ylim([-2,2])
                self.axis[0].set_box_aspect(1)
                self.axis[1].set_xlim([-2,2])
                self.axis[1].set_ylim([-2,2])
                self.axis[1].set_box_aspect(1)
                #pendulum 
                self.axis[1].plot((0, self.x1_pen[self.t]), (0, self.y1_pen[self.t]),marker="o")            
                self.axis[1].plot((self.x1_pen[self.t],self.x2_pen[self.t]),(self.y1_pen[self.t],self.y2_pen[self.t]),marker="o")
                self.axis[2].plot(torch.arange(len(self.kenetic_energy_buffer_pendulum)),self.kenetic_energy_buffer_pendulum,color="r",linestyle='dashed',label="kenetic_energy_target")
                self.axis[2].plot(torch.arange(len(self.potential_energy_buffer_pendulum)),self.potential_energy_buffer_pendulum,color="g",linestyle='dashed',label="potential_energy_target")
                self.axis[2].plot(torch.arange(len(self.total_energy_buffer_pendulum)),self.total_energy_buffer_pendulum,color="b",linestyle='dashed',label="total_energy_target")
                #network
                self.axis[0].plot((0, self.x1_net[self.t]), (0, self.y1_net[self.t]),marker="o")            
                self.axis[0].plot((self.x1_net[self.t],self.x2_net[self.t]),(self.y1_net[self.t],self.y2_net[self.t]),marker="o")
                self.axis[2].plot(torch.arange(len(self.kenetic_energy_buffer_network)),self.kenetic_energy_buffer_network,color="r",label="kenetic_energy_pred")
                self.axis[2].plot(torch.arange(len(self.potential_energy_buffer_network)),self.potential_energy_buffer_network,color="g",label="potential_energy_pred")
                self.axis[2].plot(torch.arange(len(self.total_energy_buffer_network)),self.total_energy_buffer_network,color="b",label="total_energy_pred")
                self.axis[2].legend(loc='upper left')
                #energy 
                self.axis[3].plot(torch.arange(len(self.energy0_buffer)),self.energy0_buffer,label="E0")
                self.axis[3].plot(torch.arange(len(self.energy1_buffer)),self.energy1_buffer,label="E1",linestyle='dashed')
                self.axis[3].legend(loc='upper left')
                
                plt.pause(0.01)


                self.t += 1
    
    def setup(self):
        gs = gridspec.GridSpec(2, 2)
        self.fig = plt.figure()
        ax1 = self.fig.add_subplot(gs[0, 0]) # row 0, col 0
        ax2 = self.fig.add_subplot(gs[0, 1]) # row 0, col 1
        ax3 = self.fig.add_subplot(gs[1, 0]) # row 1, col 0 
        ax4 = self.fig.add_subplot(gs[1, 1]) # row 1, col 1
        self.axis = [ax1,ax2,ax3,ax4]

        self.fig.subplots_adjust(bottom=0.25)
        self.axAlpha = self.fig.add_axes([0.15, 0.1, 0.65, 0.03])
        self.alpha_slider = Slider(
            ax=self.axAlpha,
            label='α',
            valmin=self.alpha_min,
            valmax=self.alpha_max,
            valinit=self.alpha,
        )

        self.alpha_slider.on_changed(self.callback_alpha_slider)  
    
    def calculations(self):
        input,self.y_pendulum = self.pendulum.sample(self.y0) 

        _,_,_,_,_,_,_,_,_,_,self.E0 = self.pendulum.info(input)
                
        self.x1_pen,self.y1_pen,self.x2_pen,self.y2_pen,self.dot_x1_pen,self.dot_y1_pen,self.dot_x2_pen,self.dot_y2_pen,self.K_pen,self.P_pen,self.E_pen = self.pendulum.info(self.y_pendulum)
        with torch.no_grad():
            y_network = self.network(input,self.alpha)
            self.x1_net,self.y1_net,self.x2_net,self.y2_net,self.dot_x1_net,self.dot_y1_net,self.dot_x2_net,self.dot_y2_net,self.K_net,self.P_net,self.E_net = self.pendulum.info(y_network)

        self.kenetic_energy_buffer_pendulum = deque(maxlen=self.buffer_size)
        self.potential_energy_buffer_pendulum = deque(maxlen=self.buffer_size)
        self.total_energy_buffer_pendulum = deque(maxlen=self.buffer_size)

        self.kenetic_energy_buffer_network = deque(maxlen=self.buffer_size)
        self.potential_energy_buffer_network = deque(maxlen=self.buffer_size)
        self.total_energy_buffer_network = deque(maxlen=self.buffer_size)
        self.energy0_buffer = deque(maxlen=self.buffer_size) 
        self.energy1_buffer = deque(maxlen=self.buffer_size)
        self.t = 0 

    def callback_alpha_slider(self,alpha):
        self.y0 = torch.cat((torch.rand(1)*torch.pi*2,torch.randn(1)*0.1,torch.rand(1)*torch.pi*2,torch.randn(1)*0.1),dim=0) 
        self.alpha = alpha
        self.calculations()




if __name__ == "__main__":
    epochs = 1000 #500 
    batch_size = 40 #20

    net = network([4,100,100],[4,100,100],[200,100,4])

    optimizer = optim.Adam(net.parameters(),lr=0.0001)
    
    pendulum = dampend_pendulum(time_steps = 500,end_time = 20,dampening_l1=0.1,dampening_l2=0.1)
    #pendulum = dampend_pendulum(dampening_l1=0.1,dampening_l2=0.1)
    
    alpha_disribution = Beta(torch.tensor([0.1]), torch.tensor([0.1])) #Beta(β, β) β = 0.1 
 
    plt.ion()
    loss_deque = deque(maxlen=500)
   
    for _ in tqdm(range(epochs),ascii=True):
        optimizer.zero_grad()
        total_loss = 0 
    
        for _ in range(batch_size):
            y0 = torch.cat((torch.rand(1)*torch.pi*2,torch.randn(1)*0.1,torch.rand(1)*torch.pi*2,torch.randn(1)*0.1),dim=0) #theta1,dot_theta1,theta2,dot_theta2 
            """
            theta1 : 0,2pi
            dot_theta1 : any real number
            theta2: 0,2pi
            dot_theta2 : any real number
            """
            input,target = pendulum.sample(y0)
            sequence_length = input.size(0)
            alpha = alpha_disribution.sample()
            #train
            pred = net(input,alpha)
            loss_task = ((1-alpha)*torch.sum(((pred-target)**2))/sequence_length)/batch_size
            loss_rule = alpha*(torch.sum(energy_loss(pred,input,pendulum))/sequence_length)/batch_size
        
            loss = loss_rule + loss_task

            loss.backward()
            total_loss += loss.detach()

        optimizer.step()
        loss_deque.append(loss.detach())

        plt.cla()
        plt.plot(torch.arange(len(loss_deque)),loss_deque)
        plt.pause(0.01)
    visualizer(net,pendulum)
    plt.show()