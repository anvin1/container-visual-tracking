import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
from collections import deque
import random
import time
import matplotlib as plt
class ActorNetwork(nn.Module):
    def __init__(self, npy_path,num_actions=4):
        super(ActorNetwork, self).__init__()
        
        if npy_path is not None:
            self.data_dict = np.load(npy_path, allow_pickle=True, encoding='latin1').item()

        else:
            self.data_dict = None
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.data_dict['conv1'][0].shape[3], kernel_size=7, stride=2)
        self.conv1.weight.data = torch.from_numpy(np.transpose(self.data_dict['conv1'][0]))
        self.conv1.bias.data = torch.from_numpy(self.data_dict['conv1'][1])
        
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=self.data_dict['conv2'][0].shape[3], kernel_size=5, stride=2)
        self.conv2.weight.data = torch.from_numpy(np.transpose(self.data_dict['conv2'][0]))
        self.conv2.bias.data = torch.from_numpy(self.data_dict['conv2'][1])
        
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=self.data_dict['conv3'][0].shape[3], kernel_size=3, stride=1)
        self.conv3.weight.data = torch.from_numpy(np.transpose(self.data_dict['conv3'][0]))
        self.conv3.bias.data = torch.from_numpy(self.data_dict['conv3'][1])
        
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=self.data_dict['conv4'][0].shape[3], kernel_size=3, stride=1)
        self.conv4.weight.data = torch.from_numpy(np.transpose(self.data_dict['conv4'][0]))
        self.conv4.bias.data = torch.from_numpy(self.data_dict['conv4'][1])

        self.fc1 = nn.Linear(1024 , 512)    
        self.fc2 = nn.Linear(512, num_actions)
        for param in [self.conv1.parameters(), self.conv2.parameters(), self.conv3.parameters(), self.conv4.parameters()]:
            for p in param:
                p.requires_grad = False
    def forward(self,rgb_g, rgb_l):
            feature_maps = {}
            rgb_g=rgb_g-128.0
            rgb_l=rgb_l-128.0
            x_g = F.relu(self.conv1(rgb_g))
            x_g = F.local_response_norm(x_g, size=5)
            x_g = F.max_pool2d(x_g, kernel_size=3, stride=2)

            x_g = F.relu(self.conv2(x_g))
            x_g = F.local_response_norm(x_g, size=5)
            x_g = F.max_pool2d(x_g, kernel_size=3, stride=2)

            x_g = F.relu(self.conv3(x_g))
            x_g = F.relu(self.conv4(x_g))
            x_g = x_g.reshape(-1,512)

            x_l = F.relu(self.conv1(rgb_l))
            x_l = F.local_response_norm(x_l, size=5)
            x_l = F.max_pool2d(x_l, kernel_size=3, stride=2)
            
            x_l = F.relu(self.conv2(x_l))
            x_l = F.local_response_norm(x_l, size=5)
            x_l = F.max_pool2d(x_l, kernel_size=3, stride=2)
            
            x_l = F.relu(self.conv3(x_l))
            x_l= F.relu(self.conv4(x_l))
            x_l = x_l.reshape(-1,512)

            x=torch.cat((x_g, x_l), dim=1)

        
            x =F.relu(self.fc1(x))
            x = F.tanh(self.fc2(x))
            return x

class CriticNetwork(nn.Module):
    def __init__(self, npy_path,num_actions=4):
        super(CriticNetwork, self).__init__()
        
        if npy_path is not None:
            self.data_dict = np.load(npy_path, allow_pickle=True, encoding='latin1').item()

        else:
            self.data_dict = None
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.data_dict['conv1'][0].shape[3], kernel_size=7, stride=2)
        self.conv1.weight.data = torch.from_numpy(np.transpose(self.data_dict['conv1'][0]))
        self.conv1.bias.data = torch.from_numpy(self.data_dict['conv1'][1])
        
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=self.data_dict['conv2'][0].shape[3], kernel_size=5, stride=2)
        self.conv2.weight.data = torch.from_numpy(np.transpose(self.data_dict['conv2'][0]))
        self.conv2.bias.data = torch.from_numpy(self.data_dict['conv2'][1])
        
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=self.data_dict['conv3'][0].shape[3], kernel_size=3, stride=1)
        self.conv3.weight.data = torch.from_numpy(np.transpose(self.data_dict['conv3'][0]))
        self.conv3.bias.data = torch.from_numpy(self.data_dict['conv3'][1])
        
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=self.data_dict['conv4'][0].shape[3], kernel_size=3, stride=1)
        self.conv4.weight.data = torch.from_numpy(np.transpose(self.data_dict['conv4'][0]))
        self.conv4.bias.data = torch.from_numpy(self.data_dict['conv4'][1])

        self.fc1 = nn.Linear(1024 , 512)
        self.fc2 = nn.Linear(512+num_actions, 1)

    def forward(self,rgb_g, rgb_l,action):

            rgb_g=rgb_g-128.0
            rgb_l=rgb_l-128.0
            x_g = F.relu(self.conv1(rgb_g))
            x_g = F.local_response_norm(x_g, size=5)
            x_g = F.max_pool2d(x_g, kernel_size=3, stride=2)
            
            x_g = F.relu(self.conv2(x_g))
            x_g = F.local_response_norm(x_g, size=5)
            x_g = F.max_pool2d(x_g, kernel_size=3, stride=2)
            
            x_g = F.relu(self.conv3(x_g))
            x_g = F.relu(self.conv4(x_g))
            x_g = x_g.reshape(-1,512)

            x_l = F.relu(self.conv1(rgb_l))
            x_l = F.local_response_norm(x_l, size=5)
            x_l = F.max_pool2d(x_l, kernel_size=3, stride=2)
            
            x_l = F.relu(self.conv2(x_l))
            x_l = F.local_response_norm(x_l, size=5)
            x_l = F.max_pool2d(x_l, kernel_size=3, stride=2)
            
            x_l = F.relu(self.conv3(x_l))
            x_l= F.relu(self.conv4(x_l))
            x_l = x_l.reshape(-1,512)

            x=torch.cat((x_g, x_l), dim=1)
          
            x =F.relu(self.fc1(x))
            x = torch.cat((x, action), dim=1)
            x = self.fc2(x)
            return x


class DDPG:
    def __init__(self, num_actions, buffer_capacity=10000, batch_size=128, gamma=0.99, tau=0.001, epsilon=0.5, epsilon_decay=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon

        self.Actor = ActorNetwork("vggm1-4.npy",num_actions=num_actions).to(self.device)
        self.Critic = CriticNetwork("vggm1-4.npy",num_actions=num_actions).to(self.device)

        self.TargetActor = ActorNetwork("vggm1-4.npy",num_actions=num_actions).to(self.device)
        self.TargetCritic = CriticNetwork("vggm1-4.npy",num_actions=num_actions).to(self.device)

        self.actor_optimizer_l2 = optim.Adam(self.Actor.parameters(), lr=1e-4)
        self.actor_target_optimizer_l2 = optim.Adam(self.TargetActor.parameters(),lr=1e-4)

        self.memory = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.episode = 0
        self.epsilon_decay = epsilon_decay

        self.actor_optimizer = optim.Adam(self.Actor.parameters(), lr=1e-6)
        self.critic_optimizer = optim.Adam(self.Critic.parameters(), lr=1e-5)

        self.update_target(self.TargetActor, self.Actor, 1.0)
        self.update_target(self.TargetCritic, self.Critic, 1.0)

    def update_target(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

   

    def action(self, state1, state2):
        state1 = torch.FloatTensor(state1).to(self.device)
        state2 = torch.FloatTensor(state2).to(self.device)
        
        action = self.Actor(rgb_g=state1,rgb_l=state2)
        action = action.detach().cpu().numpy().flatten()

        return action

    def train(self):
        
        if len(self.memory) < self.batch_size:
            return
        imo_g,imo_l, action, reward,  imo_g_,imo_l_, done= self.memory.sample_batch(self.batch_size)
        
        imo_g = torch.FloatTensor(imo_g).to(self.device).squeeze()
        imo_l= torch.FloatTensor(imo_l).to(self.device).squeeze()
        imo_g_ = torch.FloatTensor(imo_g_).to(self.device).squeeze()
        imo_l_ = torch.FloatTensor(imo_l_).to(self.device).squeeze()

        action = torch.FloatTensor(action).to(self.device).squeeze()
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
  
        Q_vals = self.Critic(imo_g,imo_l,action)
        
        with torch.no_grad():
            next_actions = self.TargetActor(imo_g_,imo_l_)
            next_Q_vals = self.TargetCritic(imo_g_,imo_l_,next_actions)
            target_Q_vals = reward + self.gamma * next_Q_vals * (1 - done)
        
        critic_loss = F.mse_loss(Q_vals, target_Q_vals)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.Critic(imo_g,imo_l,self.Actor(imo_g,imo_l)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
      
        if self.episode % 10000 == 0:
            self.update_target(self.TargetActor, self.Actor, self.tau)
            self.update_target(self.TargetCritic, self.Critic, self.tau)
          
            print("Target networks updated")
       
    def L2_train(self,L2Buffer):

        states1, states2, actions = L2Buffer.sample_batch(32)

        states1 = torch.FloatTensor(states1).to(self.device).squeeze()
        states2 = torch.FloatTensor(states2).to(self.device).squeeze()
        actions = torch.FloatTensor(actions).to(self.device).squeeze()


        actor_actions = self.Actor(rgb_g=states1, rgb_l=states2)
        actor_loss1 = F.mse_loss(actor_actions, actions)
        
        self.actor_optimizer_l2.zero_grad()
        actor_loss1.backward()
        self.actor_optimizer_l2.step()

        target_actor_actions = self.TargetActor(rgb_g=states1, rgb_l=states2)
        actor_loss2 = F.mse_loss(target_actor_actions, actions)

        self.actor_target_optimizer_l2.zero_grad()
        actor_loss2.backward()
        self.actor_target_optimizer_l2.step()

    def load(self, filename):
        self.Actor.load_state_dict(torch.load(filename + "_actor_model.pth"))
        self.Critic.load_state_dict(torch.load(filename + "_critic_model.pth"))
        self.TargetActor.load_state_dict(torch.load(filename + "_target_actor_model.pth"))
        self.TargetCritic.load_state_dict(torch.load(filename + "_target_critic_model.pth"))

class ReplayBuffer:
    def __len__(self):
        return len(self.buffer)
    
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, statea,statec, action, reward,  next_statea,next_statec, done):
        self.buffer.append((statea,statec, action, reward,  next_statea,next_statec, done))

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        imo_g,imo_l, action, reward,  imo_g_,imo_l_, done=  zip(*batch)
        
        imo_g = np.array( imo_g)
        imo_l = np.array(imo_l)
        imo_g_=np.array(imo_g_)
        imo_l_=np.array(imo_l_)
        action=np.array(action)
     
        return imo_g,imo_l, action, reward,  imo_g_,imo_l_, done

class L2Buffer:
    def __init__(self):
        self.buffer = []

    def add(self, batch_g, batch_l, expert_action):
       

        for i in range(len(batch_g)):
                self.buffer.append((batch_g[i], batch_l[i], expert_action[i]))

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  
        states1, states2, actions = zip(*batch)
        
        states1 = np.array(states1)
        states2 = np.array(states2)
        actions = np.array(actions)
        

        return states1, states2, actions

    
    def __len__(self):
        return len(self.buffer)



