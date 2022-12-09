import torch 
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt 
from enviroments import pendulum, dampend_pendulum

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



class mlp(nn.Module):
    def __init__(self,struct:list,act_f:nn.Module = nn.ReLU) -> None:
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
        h_rule = self.rule_encoder(x)
        h_data = self.data_encoder(x)

        h = h_data*(alpha) + h_rule*(1-alpha)
        
        return self.decision_block(h)

if __name__ == "__main__":
    net = network([4,100,100],[4,100,100],[100,100,4])

    optimizer = optim.Adam(net.parameters(),lr=0.01)


    