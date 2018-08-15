# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from copy import deepcopy

from envs import IPD

class Hp():
    def __init__(self):
        self.lr_out = 0.2
        self.lr_in = 0.3
        self.gamma = 0.96
        self.n_update = 50
        self.len_rollout = 100
        self.batch_size = 200
        self.seed = 42

hp = Hp()

ipd = IPD(hp.len_rollout, hp.batch_size)

def phi(x1,x2):
    return [x1*x2, x1*(1-x2), (1-x1)*x2,(1-x1)*(1-x2)]

def true_objective(theta1, theta2, ipd):
    p1 = torch.sigmoid(theta1)
    p2 = torch.sigmoid(theta2[[0,1,3,2,4]])
    p0 = (p1[0], p2[0])
    p = (p1[1:], p2[1:])
    # create initial laws, transition matrix and rewards:
    P0 = torch.stack(phi(*p0), dim=0).view(1,-1)
    P = torch.stack(phi(*p), dim=1)
    R = torch.from_numpy(ipd.payout_mat).view(-1,1).float()
    # the true value to optimize:
    objective = (P0.mm(torch.inverse(torch.eye(4) - hp.gamma*P))).mm(R)
    return -objective

def act(batch_states, theta):
    batch_states = torch.from_numpy(batch_states).long()
    probs = torch.sigmoid(theta)[batch_states]
    m = Bernoulli(1-probs)
    actions = m.sample()
    log_probs_actions = m.log_prob(actions)
    return actions.numpy().astype(int), log_probs_actions

def get_gradient(objective, theta):
    # create differentiable gradient for 2nd orders:
    grad_objective = torch.autograd.grad(objective, (theta), create_graph=True)[0]
    return grad_objective

def step(theta1, theta2):
    # evaluate progress and opponent modelling
    (s1, s2), _ = ipd.reset()
    score1 = 0
    score2 = 0
    freq1 = np.ones(5)
    freq2 = np.ones(5)
    n1 = 2*np.ones(5)
    n2 = 2*np.ones(5)
    for t in range(hp.len_rollout):
        s1_ = deepcopy(s1)
        s2_ = deepcopy(s2)
        a1, lp1 = act(s1, theta1)
        a2, lp2 = act(s2, theta2)
        (s1, s2), (r1, r2),_,_ = ipd.step((a1, a2))
        # cumulate scores
        score1 += np.mean(r1)/float(hp.len_rollout)
        score2 += np.mean(r2)/float(hp.len_rollout)

        # count actions
        for i,s in enumerate(s1_):
            freq1[int(s)] += (s2==3).astype(float)[i]
            freq1[int(s)] += (s2==1).astype(float)[i]
            n1[int(s)] += 1

        for i,s in enumerate(s2_):
            freq2[int(s)] += (s1==3).astype(float)[i]
            freq2[int(s)] += (s1==1).astype(float)[i]
            n2[int(s)] += 1

    # infer opponent's parameters
    theta1_ = -np.log(n1/freq1 - 1)
    theta2_ = -np.log(n2/freq2 - 1)
    theta1_ = torch.from_numpy(theta1_).float().requires_grad_()
    theta2_ = torch.from_numpy(theta2_).float().requires_grad_()

    return (score1, score2), theta1_, theta2_

class Agent():
    def __init__(self, theta=None):
        # init theta and its optimizer
        self.theta = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.theta_optimizer = torch.optim.Adam((self.theta,),lr=hp.lr_out)

    def theta_update(self, objective):
        self.theta_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        self.theta_optimizer.step()

    def in_lookahead(self, other_theta):
        other_objective = true_objective(other_theta, self.theta, ipd)
        grad = get_gradient(other_objective, other_theta)
        return grad

    def out_lookahead(self, other_theta):
        objective = true_objective(self.theta, other_theta, ipd)
        self.theta_update(objective)

def play(agent1, agent2, n_lookaheads):
    print("start iterations with", n_lookaheads, "lookaheads:")
    joint_scores = []

    # init opponent models
    theta1_ = torch.zeros(5, requires_grad=True)
    theta2_ = torch.zeros(5, requires_grad=True)

    for update in range(hp.n_update):

        #print(theta1_)
        #print(agent1.theta)

        for k in range(n_lookaheads):
            # estimate other's gradients from in_lookahead:
            grad2 = agent1.in_lookahead(theta2_)
            grad1 = agent2.in_lookahead(theta1_)
            # update other's theta
            theta2_ = theta2_ - hp.lr_in * grad2
            theta1_ = theta1_ - hp.lr_in * grad1

        # update own parameters from out_lookahead:
        agent1.out_lookahead(theta2_)
        agent2.out_lookahead(theta1_)

        # evaluate:
        score, theta1_, theta2_ = step(agent1.theta, agent2.theta)
        joint_scores.append(0.5*(score[0] + score[1]))

        # print
        if update%10==0 :
            p1 = [p.item() for p in torch.sigmoid(agent1.theta)]
            p2 = [p.item() for p in torch.sigmoid(agent2.theta)]
            print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]) , 'policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p1[0], p1[1], p1[2], p1[3], p1[4]),' (agent2) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))

    return joint_scores

# plot progress:
if __name__=="__main__":

    colors = ['b','c','m','r']

    for i in range(4):
        torch.manual_seed(hp.seed)
        scores = np.array(play(Agent(), Agent(), i))
        plt.plot(scores, colors[i], label=str(i)+" lookaheads")

    plt.legend()
    plt.xlabel('rollouts')
    plt.ylabel('joint score')
    plt.show()
