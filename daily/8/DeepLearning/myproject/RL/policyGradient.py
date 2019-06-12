import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np

class Policy(nn.Module):
    def __init__(self,n_states,n_actions):
        '''
        定义3层神经网络 n_states-->32-->64-->n_actions
        :param n_states: 
        :param n_actions: 
        '''
        super().__init__()
        self.n_states=n_states
        self.n_actions=n_actions

        self.fc1=nn.Linear(self.n_states,32)
        self.fc2=nn.Linear(32,64)
        self.fc3=nn.Linear(64,self.n_actions)
    def forward(self,x):
        '''
        计算policy,保存到self.policy_prob:(N,A)
        返回logit
        :param x: (N,n_states)输入的状态
        :return: 
        '''
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))  #logit (N,action)

        prob=F.softmax(x,1)
        self.policy_prob=prob.data.cpu().numpy()
        return x


def update_policy(states,actions,Gs,policy_model,optimizer):
    '''
    
    :param states:tensor(N,n_states) the state
    :param actions:tensor(N,) the action which take
    :param Gs: tensor(N,):estimated return
    :param policy_model:NN
    :return: 
    '''
    # 计算loss=avg -(Gs-bs)logPr(a|s)

    #(N,A)
    log_policy=F.log_softmax(policy_model(states))
    #评估当前(s,a)
    N=log_policy.size(0)
    log_action=log_policy[torch.arange(N),actions]

    #回报越大,越是应该关注这组(a,s),bs这里给出0.1
    weights=Gs-0.1

    loss=-weights*log_action
    loss=torch.mean(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
def tensor(x,type=np.float32):return torch.tensor(type(x))

def reward2Return(rewards,gamma=0.9):
    '''
    本函数复制把reward转成 return
        
    Return(t)=reward[t]+gamma*Return(t+1)
    
    :param rewards: list of T element
    :return: 
    '''
    R=[0]*(len(rewards)+1)

    for t in range(len(rewards)-1,-1,-1):
        R[t]=rewards[t]+gamma*R[t+1]
    return np.asarray(R[:-1])


def lanch_session(env,Tmax=1000,policy_model=None,optimizer=None,device='cuda'):
    '''
    
    :param env: 
    :param Tmax: 
    :param policy_model: 
    :return: 
    '''
    states=[]
    actions=[]
    rewards=[]


    s=env.reset()
    for _ in range(Tmax):
        #根据policy_model,选择一个动作
        policy_model(tensor([s]))
        p=policy_model.policy_prob
        a=np.random.choice(policy_model.n_actions,p=p)

        news,r,done,_=env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        if done:
            break
        s=news
    total_reward = sum(rewards)
    #更新policy_model
    states=tensor(states).to(device)
    actions=tensor(actions,np.int32).to(device)
    Gs=tensor(reward2Return(rewards)).to(device)

    update_policy(states,actions,Gs,optimizer)
    return total_reward

def save_model(modelpath,model,optimizer):
    state ={}
    state['optimizer'] = optimizer.state_dict()
    state['model'] = model.state_dict()
    torch.save(state,modelpath)
    print('save model to path ',modelpath)

if __name__ == '__main__':

    device='cuda'
    taskname='CartPole-v0'
    modelpath='policy_models/policy_%d.pt'%taskname

    env=gym.make(taskname)
    n_states,n_actions=env.observation_space.shape[0],env.action_space.n

    policy_model=Policy(n_states,n_actions).to(device)
    optimizer=optim.Adam(policy_model.parameters())


    bestreward=-1
    for i in range(1000):
        reward_episode=[lanch_session(env,policy_model=policy_model,optimizer=optimizer)
        for _ in range(1000)]
        avg_reward=np.mean(avg_reward)

        print('step {},avg reward {:.2f}'.format(i,avg_reward))
        if bestreward<avg_reward:
            bestreward=avg_reward
            save_model(modelpath,policy_model,optimizer)
        if avg_reward>300:
            break

