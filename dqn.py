import ray
import argparse
import gym
import collections
import random
import socket
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        n = max(n, len(self.buffer))
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()


@ray.remote(num_cpus=8, num_gpus=1)
class Runner(object):
    def __init__(self, actor_id, lr, buffer_limit):
        self.env = env = gym.make('CartPole-v1')
        self.q = Qnet()
        self.q_target = Qnet()
        self.q_target.load_state_dict(self.q.state_dict())
        self.memory = ReplayBuffer(buffer_limit)

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

        
    def start(self, n_epis, epsilon):
        for n_epi in range(n_epis):
            s = self.env.reset()
            done = False

            while not done:
                a = self.q.sample_action(torch.from_numpy(s).float(), epsilon)
                s_prime, r, done, info = self.env.step(a)
                done_mask = 0.0 if done else 1.0
                self.memory.put((s,a,r/100.0,s_prime,done_mask))
                s = s_prime

                if done:
                    break


    def compute_gradient(self, params, batch_size, gamma):
        self.q.load_state_dict(params['q'])
        self.q_target.load_state_dict(params['q_target'])
        s, a, r, s_prime, done_mask = self.memory.sample(batch_size)

        q_out = self.q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()

        ret = []
        for paramName, paramValue in self.q.named_parameters():
            ret.append(paramValue.grad)

        return socket.gethostbyname(socket.gethostname()), ret

        
def train(config):


    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=config.lr)

    agent_ids = [Runner.remote(i, config.lr, config.buffer_limit) for i in range(config.num_workers)]

    params = {}
    params['q'] = q.state_dict()
    params['q_target'] = q_target.state_dict()

    for iter_i in range(config.n_iters):
        print(iter_i, 'iteration start')
        epsilon = max(0.01, 0.08 - 0.01*(iter_i/200))

        ray.get([agent_id.start.remote(config.n_episodes, epsilon) for agent_id in agent_ids])

        params = {}
        params['q'] = q.state_dict()
        params['q_target'] = q_target.state_dict()

        gradient_ids = [agent_id.compute_gradient.remote(params, config.batch_size, config.gamma) for agent_id in agent_ids]

        losses = []
        optimizer.zero_grad()
        while len(gradient_ids):
            done_id, gradient_ids = ray.wait(gradient_ids)

            worker_ip, gradients = ray.get(done_id[0])
            print('worker_ip: ', worker_ip)
            for p, grad in zip(q.parameters(), gradients):
                if p.grad == None:
                    p.grad = grad
                else:
                    p.grad += grad
            
        for p in q.parameters():
            p.grad /= config.num_workers
        optimizer.step()

        if iter_i % 2:
            q_target.load_state_dict(q.state_dict())

        m_name = './models/model' + str(iter_i) + '.pt'
        
        torch.save(q.state_dict(), m_name)


if __name__ == '__main__':

    ray.init(address='auto')

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--n_iters', default=10000, type=int)
    parser.add_argument('--n_episodes', default=50, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--gamma', default=0.98, type=float)
    parser.add_argument('--buffer_limit', default=5000, type=int)

    config = parser.parse_args()

    train(config)