'''
author: lcy
This file provides the python interfaces of the routers simulations environment
this file do the interactions between ryu+mininet
'''
import time

from gym import spaces
from torch.cuda import graph

if __name__ == "__main__":
    from utils import weight_choice
else:
    from net_env.utils import weight_choice

import torch
import random
import numpy as np
import os
import sys
import glob
import heapq
import copy
import json
import socket
import argparse

import networkx as nx

# 论文4种不同类型的流量

# 在本文的实验中
# 1.不考虑QoS需求，将所有流量类型设置为一样
# 2.为不同类型流量提供不同的带宽、延迟、丢包保证（丢包很难保证，只能通过保证延迟和带宽间接保证）

flow_type = {
    "latency-sensitive": 0,
    "throughput-sensitive": 1,
    "latency-throughput-sensitive": 2,
    "latency-loss-sensitive": 3
}

request_times = {
    "10": [[10], [10], [10], [10]],
    "20": [[20], [20], [20], [20]],
    "30": [[30], [30], [30], [30]],
    "40": [[40], [40], [40], [40]],
    "50": [[50], [50], [50], [50]]
}


# 流量请求 源、目的、开始时间、结束时间、流量请求大小、流量类型
class Request:
    def __init__(self, s, t, start_time, end_time, demand, rtype):
        # use open time interval: start_time <= time < end_time
        self.s = s
        self.t = t
        self.start_time = start_time
        self.end_time = end_time
        self.demand = demand
        self.rtype = rtype

    '''
        deprecated function
    '''

    def to_json(self):
        data = {
            'src': int(self.s),
            'dst': int(self.t),
            'time': int(self.end_time - self.start_time),
            'rtype': int(self.rtype),
            'demand': int(self.demand),
        }
        return json.dumps(data)

    def __lt__(self, other):
        return self.end_time < other.end_time

    def __str__(self):
        return ("s: %d t: %d\nstart_time: %d\nend_time: %d\ndemand: %d\nrtype: %d"
                % (self.s, self.t, self.start_time, self.end_time, self.demand, self.rtype))


class NetEnv:
    '''
    must run setup before using other methods
    '''

    def __init__(self, args):

        # 获取设定的参数
        self.args = args

        # 保存当前网络拓扑图
        self.graph = nx.DiGraph()

        # 多个智能体的状态 每个智能体对应一个空间
        if self.args.agent_mode == "multi_agent":
            # 多个智能体状态空间
            self._observation_spaces = []
            # 多个智能体动作空间
            self._action_spaces = []

        # 单个智能体的状态
        elif self.args.agent_mode == "single_agent":
            # 单智能体 状态空间（针对离散空间DQN、PPO、A2C）
            self._single_observation_spaces = None
            # 单智能体 动作空间 K条候选路径
            self._single_action_spaces = None

        # 延迟数据自适应归一化处理
        self._delay_discounted_factor = 0.99
        # 丢包数据自适应归一化处理
        self._loss_discounted_factor = 0.99

        # 初始化时 需要和 mininet以及controller建立连接
        # set up communication to remote hosts(mininet host)
        # # 和Mininet平台建立连接
        # MININET_HOST_IP = '127.0.0.1'  # Testbed server IP
        # MININET_HOST_PORT = 5000
        # # 这里设置BUFFER_SIZE 和 DRL-OR-S 设置的不一样
        # self.BUFFER_SIZE = 1024
        # self.mininet_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.mininet_socket.connect((MININET_HOST_IP, MININET_HOST_PORT))

        # 和控制器建立连接
        # 计算好路由路径之后，向控制器发送包含路由路径path的消息，由控制器下发流表，
        # 流表匹配项为（ipv4_src, src_port, ipv4_dst, dst_port）
        # CONTROLLER_IP = '127.0.0.1'
        # CONTROLLER_PORT = 3999
        # self.controller_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.controller_socket.connect((CONTROLLER_IP, CONTROLLER_PORT))

    # 多智能体的状态空间
    @property
    def observation_spaces(self):
        return self._observation_spaces

    @property
    def single_observation_spaces(self):
        return self._single_observation_spaces

    @property
    def action_spaces(self):
        return self._action_spaces

    @property
    def single_action_spaces(self):
        return self._single_action_spaces

    '''
    @param:
        toponame: a string indicate the toponame for the environment
    @retval:
        node_num: an int
        observation_spaces: a [node_name] shape list shows the observation spaces for each node, now are Box(k)
        action_spaces: a [node_name] shape list shows the observation spaces for each node, now are Discrete(l)
    '''

    def setup(self, toponame, demand_matrix_name="test"):

        self._time_step = 0
        self._request_heapq = []

        # self._type_num = self.args.service_type_num
        #
        # if self._type_num == 4:
        #     self._type_dist = np.array([0.2, 0.3, 0.3, 0.2])
        #
        # elif self._type_num == 2:
        #     self._type_dist = np.array([0.5, 0.5])
        #
        # elif self._type_num == 1:
        #     self._type_num = np.array([1])

        # 4种类型流量
        # 每种类型流量的分布概率为0.2 0.3 0.3 0.2
        self._type_num = 4
        self._type_dist = np.array([0.2, 0.3, 0.3, 0.2])  # TO BE CHECKED BEFORE EXPERIMENT

        # load topo info(direct paragraph)
        if toponame == "test":
            self._node_num = 4
            self._edge_num = 8
            self._observation_spaces = []
            self._action_spaces = []
            # topology info
            # 边的连接关系, 下标是每个节点，对表
            self._link_lists = [[3, 1], [0, 2], [1, 3], [2, 0]]
            self._shr_dist = [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]]
            self._link_capa = [[0, 1000, 0, 5000], [1000, 0, 5000, 0], [0, 5000, 0, 5000],
                               [5000, 0, 5000, 0]]  # link capacity (Kbps)
            self._link_usage = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # usage of link capacity
            self._link_losses = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # link losses, x% x is int
            # request generating setting
            # bandwidth aware usually need as large bandwidth as they can, delay-bandwidth aware may have several stage of bandwidth demand
            self._request_demands = [[100], [500], [500], [100]]
            self._request_times = [[10], [10], [10], [10]]  # simple example test
            # 节点 和 agent对应关系
            self._node_to_agent = [0, 1, 2, 3]  # i:None means drl-agent not deploys on node i; i:j means drl-agent j deploys on node i
            # agent和节点对应关系
            self._agent_to_node = [0, 1, 2, 3]  # indicate the ith agent's node
            self._agent_num = 4
            # 流量矩阵
            self._demand_matrix = [0, 1, 1, 1,
                                   1, 0, 1, 1,
                                   1, 1, 0, 1,
                                   1, 1, 1, 0]

            if self.args.agent_mode == "multi_agent":

                # 针对每个节点生成智能体的状态空间
                # 包含部分状态和条件状态，
                # 部分状态：
                # 源节点one-hot编码 和 目标节点 one-hot编码，服务类型(1) 和 最大速率
                # 拓扑信息包括：邻居节点到不同目的节点的最短距离 链路丢包率
                # 资源利用信息：邻居节点到不同目的节点最大可用带宽 链路剩余容量

                for i in self._agent_to_node:
                    # onehot src, onehot dst, neighbour shr dist to each node
                    # low and high for Box isn't essential
                    self._observation_spaces.append(spaces.Box(0., 1.,
                                                               [1 + self._node_num * len(self._link_lists[i]) +
                                                                self._node_num * len(self._link_lists[i]) +
                                                                self._node_num ** 2 + self._node_num ** 2 +
                                                                self._type_num +
                                                                self._node_num * 2],
                                                               dtype=np.float32))  # maximum observation space
                    # self._observation_spaces.append(spaces.Box(0., 1., [1 + len(self._link_lists[i]) + len(self._link_lists[i]) + self._type_num + self._node_num * 2], dtype=np.float32)) # minimum observation space
                    self._action_spaces.append(spaces.Discrete(2))

            # 单智能体的状态空间：
            # 网络总体延迟、丢包、链路利用率、最短距离矩阵(还是当前距离矩阵)
            # 源节点的 one-hot编码， 目的节点的 one-hot编码
            elif self.args.agent_mode == "single_agent":
                self._single_observation_spaces = spaces.Box(0., 1.,
                                                             [self._node_num ** 2 +
                                                              self._node_num ** 2 +
                                                              self._node_num * 2], dtype=np.float32)
                # 两条候选路径
                self._single_action_spaces = spaces.Discrete(2)

        elif toponame in ["Abi", "GEA", "GBN"]:

            self._load_topology(toponame, demand_matrix_name)

        else:
            print(f"{toponame} not supported")
            raise NotImplementedError

        # 单智能体
        if self.args.agent_mode == "multi_agent":
            # 智能体数量 节点数量 状态空间 动作空间 服务类型
            return self._agent_num, self._node_num, self._observation_spaces, self._action_spaces, self._type_num


        elif self.args.agent_mode == "single_agent":
            # 单智能体状态空间 单智能体动作空间
            return self._single_observation_spaces, self._single_action_spaces

    '''
    reset the environment
    @retval:
        states: [torch.tensor([x, y, ...]), ...]
    '''

    def reset(self):
        # 时间步
        self._time_step = 0

        # 流量请求
        self._request_heapq = []

        # 链路使用量
        self._link_usage = [([0.] * self._node_num) for i in range(self._node_num)]

        # 延迟归一化
        self._delay_normal = [([1.] * self._node_num) for i in range(self._node_num)]

        # 丢包归一化
        self._loss_normal = [([1.] * self._node_num) for i in range(self._node_num)]

        self._update_state()

        return self._request, self._states

    '''
    interact with controller and mininet to install path rules and generate request
    @retval:
        metrics of path including throughput, packet loss 
    '''

    def sim_interact(self, request, path):
        # install path in controller
        data_js = {}
        # TCP/UDP 报文头包括源端口和目的端口
        # 流表匹配项（源IP、目的IP、）
        data_js['path'] = path
        data_js['ipv4_src'] = "10.0.0.%d" % (request.s + 1)
        data_js['ipv4_dst'] = "10.0.0.%d" % (request.t + 1)
        data_js['src_port'] = self._time_step % 10000 + 10000
        data_js['dst_port'] = self._time_step % 10000 + 10000
        msg = json.dumps(data_js)
        self.controller_socket.send(msg.encode())
        self.controller_socket.recv(self.BUFFER_SIZE)

        # communicate to testbed
        data_js = {}
        data_js['src'] = int(request.s)
        data_js['dst'] = int(request.t)
        data_js['src_port'] = int(self._time_step % 10000 + 10000)
        data_js['dst_port'] = int(self._time_step % 10000 + 10000)
        data_js['rtype'] = int(request.rtype)
        data_js['demand'] = int(request.demand)
        data_js['rtime'] = int(request.end_time - request.start_time)
        msg = json.dumps(data_js)
        self.mininet_socket.send(msg.encode())
        # get the feedback
        msg = self.mininet_socket.recv(self.BUFFER_SIZE)
        data_js = json.loads(msg)
        return data_js

    '''
    use action to interact with the environment
    @param: 
        actions:[torch.tensor([x]), ... the next hop action for each agent
        gfactors: list:shape[node_num] shows the reward factor for the global and local rwd
    @retval:
        states: [torch.tensor([x, y, ...]), ...]
        rewards: [torch.tensor([r]), ...]
        path: [start, x, y, z, ..., end]
    '''

    def step(self, actions, gfactors, simenv=True):
        # update

        # 根据 agent 采取的行动 生成路由路径
        path = [self._request.s]
        count = 0
        capacity = 1e9
        pre_node = None
        circle_flag = 0
        link_flag = [[0] * self._node_num for i in range(self._node_num)]
        node_flag = [0] * self._node_num
        node_flag[self._request.s] = 1
        while count < self._node_num:
            curr_node = path[count]
            agent_ind = self._node_to_agent[curr_node]

            # 如果当前节点由智能体控制
            if agent_ind is not None:
                next_hop = self._link_lists[curr_node][actions[agent_ind][0].item()]

            # 如果当前节点不由智能体控制
            else:
                temp = [None, 1e9]
                for i in self._link_lists[curr_node]:
                    if i != pre_node and self._shr_dist[i][self._request.t] < temp[1]:
                        temp = [i, self._shr_dist[i][self._request.t]]
                next_hop = temp[0]

            # 产生了路由环路，计算最短路由路径
            if link_flag[curr_node][next_hop] == 1:
                circle_flag = 1
                path = self.calcSHR(self._request.s, self._request.t)
                count = len(path) - 1
                break

            else:
                link_flag[curr_node][next_hop] = 1

            # delete ring
            if node_flag[next_hop] == 1:
                while path[count] != next_hop:
                    node_flag[path[count]] = 0
                    path.pop()
                    count -= 1
            else:
                path.append(next_hop)
                node_flag[next_hop] = 1
                count += 1
            if next_hop == self._request.t:
                break
            pre_node = curr_node

        # extra safe learning part to congestion prevention

        # for link failure
        for i in range(len(path) - 1):
            if self._link_capa[path[i]][path[i + 1]] == 0:
                circle_flag = 1
                path = self.calcSHR(self._request.s, self._request.t)
                count = len(path) - 1
                break
        # a looser fallback policy trigger

        threshold = 1.2  # 2 for heavy load
        for i in range(len(path) - 1):
            if self._link_usage[path[i]][path[i + 1]] > threshold * self._link_capa[path[i]][path[i + 1]]:
                circle_flag = 1  # indicating safe learning approach being used
                path = self.calcBCSHR(self._request.s, self._request.t, self._request.demand)
                if path == None:
                    path = self.calcWP(self._request.s, self._request.t)
                count = len(path) - 1
                break

        # update sim network state
        capacity = 1e9
        for i in range(len(path) - 1):
            capacity = min(capacity,
                           max(0, self._link_capa[path[i]][path[i + 1]] - self._link_usage[path[i]][path[i + 1]]))
            self._link_usage[path[i]][path[i + 1]] += self._request.demand
        capacity = max(capacity, 0)
        self._request.path = copy.copy(path)

        # interact with sim-env(ryu + mininet)
        if simenv:
            ret_data = self.sim_interact(self._request, path)

            # for reliability test
            if 'change' in ret_data:
                self.change_env(ret_data['change'])

            delay = ret_data['delay']
            throughput = ret_data['throughput']
            loss_rate = ret_data['loss']
            delay_cut = min(1000, delay)  # since 1000 is much larger than common delay


            # 延迟、丢包、数据归一化处理
            self._delay_normal[self._request.s][self._request.t] = (self._delay_discounted_factor * self._delay_normal[self._request.s][self._request.t]
                                                                    + (1 - self._delay_discounted_factor) * delay_cut)
            delay_scaled = delay_cut / self._delay_normal[self._request.s][self._request.t]
            delay_sq = - delay_scaled ** 1

            self._loss_normal[self._request.s][self._request.t] = (self._loss_discounted_factor * self._loss_normal[self._request.s][self._request.t]
                                                                   + (1 - self._loss_discounted_factor) * loss_rate)
            loss_scaled = loss_rate / (0.01 + self._loss_normal[self._request.s][self._request.t])
            loss_sq = - loss_scaled ** 1

            throughput_log = np.log(0.5 + throughput / self._request.demand)  # avoid nan

        else:
            delay = 0.
            throughput = 0.
            loss_rate = 0.
            delay_scaled = 0.
            delay_sq = 0.
            loss_sq = 0.
            throughput_log = 0.

        # calc global rwd of the generated path
        # 根据产生的路径计算奖励
        # 延迟流
        if self._request.rtype == 0:
            global_rwd = 1 * delay_sq

        # 吞吐流
        elif self._request.rtype == 1:
            global_rwd = 0. * (delay_sq) + 1 * throughput_log - 0. * min(self._request.demand / (capacity + 1), 1)

        # 延迟、吞吐流
        elif self._request.rtype == 2:
            global_rwd = 0.5 * delay_sq + 0.5 * throughput_log

        # 延迟、丢包流
        else:
            global_rwd = 0.5 * delay_sq + 0.5 * loss_sq

        # avoid unsafe route
        # fall back policy penalty
        # 产生路由环路时，给予全局负的奖励，惩罚当前做出的决策
        if circle_flag == 1:
            global_rwd -= 5

        rewards = []

        # 计算全局奖励 和 每个agent局部奖励的和
        for i in range(self._agent_num):
            # add a dist reward for each node
            # local rwd = delta dist for the action each node do
            ind = self._agent_to_node[i]
            if ind == self._request.t:
                local_rwd = 0.
            else:
                action_hop = self._link_lists[ind][actions[i][0].item()]
                local_rwd = (self._shr_dist[ind][self._request.t] - self._shr_dist[action_hop][self._request.t] - 1) / \
                            self._shr_dist[ind][self._request.t]
            rewards.append(torch.tensor([0.1 * (gfactors[i] * global_rwd + (1 - gfactors[i]) * local_rwd)]))

        delta_dist = count - self._shr_dist[self._request.s][self._request.t]
        delta_demand = min(capacity, self._request.demand) / self._request.demand
        throughput_rate = throughput / self._request.demand
        rtype = self._request.rtype

        # generate new state
        self._time_step += 1
        self._update_state()

        return self._states, rewards, path, delta_dist, delta_demand, circle_flag, rtype, global_rwd, delay, throughput_rate, loss_rate

    def get_k_path(self, source, target, k_path_num=5):
        path = list(nx.shortest_simple_paths(self.graph, source, target))[:k_path_num]
        while len(path) < k_path_num:
            extend_path = list(nx.shortest_simple_paths(self.graph, source, target))[:k_path_num - len(path)]
            path.extend(extend_path)
        return path

    def step_single_agent(self, action, simenv=True):
        k_path_list = self.get_k_path(self._request.s, self._request.t)
        path = k_path_list[action]


        # update sim network state
        # 更新网络状态
        capacity = 1e9
        for i in range(len(path) - 1):
            capacity = min(capacity,
                           max(0, self._link_capa[path[i]][path[i + 1]] - self._link_usage[path[i]][path[i + 1]]))
            self._link_usage[path[i]][path[i + 1]] += self._request.demand
        capacity = max(capacity, 0)
        self._request.path = copy.copy(path)

        if simenv:

            ret_data = self.sim_interact(self._request, path)

            delay = ret_data['delay']
            throughput = ret_data['throughput']
            loss_rate = ret_data['loss']
            delay_cut = min(1000, delay)  # since 1000 is much larger than common delay

            # 对延迟数据进行自适应归一化
            self._delay_normal[self._request.s][self._request.t] = self._delay_discounted_factor * \
                                                                   self._delay_normal[self._request.s][
                                                                       self._request.t] + (1 - self._delay_discounted_factor) * delay_cut
            delay_scaled = delay_cut / self._delay_normal[self._request.s][self._request.t]
            delay_sq = - delay_scaled ** 1

            # 对丢包数据进行自适应归一化
            self._loss_normal[self._request.s][self._request.t] = self._loss_discounted_factor * \
                                                                  self._loss_normal[self._request.s][
                                                                      self._request.t] + (1 - self._loss_discounted_factor) * loss_rate
            loss_scaled = loss_rate / (0.01 + self._loss_normal[self._request.s][self._request.t])
            loss_sq = - loss_scaled ** 1

            throughput_log = np.log(0.5 + throughput / self._request.demand)  # avoid nan

        else:
            delay = 0.
            throughput = 0.
            loss_rate = 0.
            delay_scaled = 0.
            delay_sq = 0.
            loss_sq = 0.
            throughput_log = 0.

            # calc global rwd of the generated path
        if self._request.rtype == 0:
            global_rwd = 1 * delay_sq

        elif self._request.rtype == 1:
            global_rwd = 0. * delay_sq + 1 * throughput_log - 0. * min(self._request.demand / (capacity + 1), 1)

        elif self._request.rtype == 2:
            global_rwd = 0.5 * delay_sq + 0.5 * throughput_log

        else:
            global_rwd = 0.5 * delay_sq + 0.5 * loss_sq

        return global_rwd







    '''
    giving current agent and it's action, return next agent ind and the nodes in the path
    agent = None means that path termiatied before meet a agent(or agent on the t node)
    @retval
        agent: the index of next agent
        path: the path node from 
    '''

    def next_agent(self, agent, action):
        curr_node = self._agent_to_node[agent]
        path = []  # not include current agent's node
        pre_node = curr_node
        action_hop = self._link_lists[curr_node][action[0].item()]
        curr_node = action_hop
        while (1):
            path.append(curr_node)
            if curr_node == self._request.t:
                return None, path
            if self._node_to_agent[curr_node] != None:
                break
            temp = [None, 1e9]
            for i in self._link_lists[curr_node]:
                if i != pre_node and self._shr_dist[i][self._request.t] < temp[1]:
                    temp = [i, self._shr_dist[i][self._request.t]]
            pre_node = curr_node
            curr_node = temp[0]
        return self._node_to_agent[path[-1]], path

    '''
    return first agent index and the nodes in the path
    @retval
        agent: the index of next agent
        path: the path node from 
    '''

    def first_agent(self):

        # 加入第一个节点，当前路径的源节点
        path = [self._request.s]
        pre_node = None
        while (1):
            curr_node = path[-1]
            # 当前节点是目的节点，返回路径
            if curr_node == self._request.t:
                return None, path

            # 如果当前节点不是由agent控制，结束循环，返回当前的节点和
            if self._node_to_agent[curr_node] != None:
                break

            temp = [None, 1e9]
            for i in self._link_lists[curr_node]:
                if i != pre_node and self._shr_dist[i][self._request.t] < temp[1]:
                    temp = [i, self._shr_dist[i][self._request.t]]
            path.append(temp[0])
            pre_node = curr_node

        return self._node_to_agent[path[-1]], path

    '''
    generate requests and update the state of environment
    '''

    def _update_state(self):

        # update env request heapq
        # 对于已经完成的流量请求释放资源， 这里基于timestep同步时间
        while len(self._request_heapq) > 0 and self._request_heapq[0].end_time <= self._time_step:
            request = heapq.heappop(self._request_heapq)
            path = request.path
            if path is not None:
                # 更新已经被占用链路利用资源
                for i in range(len(path) - 1):
                    self._link_usage[path[i]][path[i + 1]] -= request.demand

        # generate new request
        nodelist = range(self._node_num)
        # uniform sampling
        # 等概率从节点列表采样2个节点
        # s, t = random.sample(nodelist, 2)
        # sampling according to demand matrix

        # 根据流量矩阵随机选择两个节点产生流量
        ind = weight_choice(self._demand_matrix)
        # ind 对应流量矩阵一个点
        # s 对应行，t 对应列
        s = ind // self._node_num
        t = ind % self._node_num

        start_time = self._time_step

        # 不区分流量类型
        if self.args.service_type_num == 1:
            # [[100], [1500], [1500], [500]] kbps
            rtype = -1 #不区分类型
            demand = random.choice(self._request_demands)[0]


        elif self.args.service_type_num == 4:
            # 从四种流量类型随机选择一个 分布概率分别为[0.2, 0.3, 0.3, 0.2]
            rtype = np.random.choice(list(range(self._type_num)), p=self._type_dist)
            # 这里 request_demands已经确定了， 不管 random choice都是一样的
            demand = random.choice(self._request_demands[rtype])

        # 这里由于每种类型的流量持续时间一样所以不存在随机选择
        # 从 10 - 50
        end_time = start_time + random.choice(self._request_times[rtype])
        print("start_time:", start_time, "end_time:", end_time)

        self._request = Request(s, t, start_time, end_time, demand, rtype)
        heapq.heappush(self._request_heapq, self._request)

        # calc wp dist for each node pair
        # 计算源和目的最大带宽（瓶颈链路带宽）
        self._wp_dist = []
        for i in range(self._node_num):
            self._wp_dist.append([])
            for j in range(self._node_num):
                if j == i:
                    self._wp_dist[i].append(1e6)  # not aware of i -> i
                elif j in self._link_lists[i]:
                    self._wp_dist[i].append(self._link_capa[i][j] - self._link_usage[i][j])
                else:
                    self._wp_dist[i].append(- 1e6)  # testing for heavy load

        for k in range(self._node_num):
            for i in range(self._node_num):
                for j in range(self._node_num):
                    if self._wp_dist[i][j] < min(self._wp_dist[i][k], self._wp_dist[k][j]):
                        self._wp_dist[i][j] = min(self._wp_dist[i][k], self._wp_dist[k][j])

                        # generate the output state of environment

        # common state for each agent
        link_usage_info = []
        for j in range(self._node_num):
            for k in range(self._node_num):
                link_usage_info.append(self._link_capa[j][k] - self._link_usage[j][k])

        link_loss_info = []
        for j in range(self._node_num):
            for k in range(self._node_num):
                link_loss_info.append(self._link_losses[j][k] / 100)  # input link loss x indicating x%

        if self.args.agent_mode == "multi_agent":

            self._states = []
            for i in self._agent_to_node:
                # generate src and dst one hot state
                type_state = torch.tensor(list(np.eye(self._type_num)[self._request.rtype]))
                src_state = torch.tensor(list(np.eye(self._node_num)[self._request.s]))
                dst_state = torch.tensor(list(np.eye(self._node_num)[self._request.t]))
                one_hot_state = torch.cat([type_state, src_state, dst_state], 0)

                # generate neighbors shr distance state
                neighbor_dist_state = []
                for j in self._link_lists[i]:
                    neighbor_dist_state += self._shr_dist[j]
                    # neighbor_dist_state.append(self._shr_dist[j][self._request.t]) #for less state
                neighbor_dist_state = torch.tensor(neighbor_dist_state, dtype=torch.float32)

                # generate link edge state
                link_usage_state = torch.tensor(link_usage_info, dtype=torch.float32)

                # generate link loss state
                # 这里丢包是初始化链路的丢包状态，不是实时的丢包状态
                link_loss_state = torch.tensor(link_loss_info, dtype=torch.float32)

                # 带宽需求
                # generate demand and time state
                extra_info_state = torch.tensor([self._request.demand], dtype=torch.float32)

                # generate neighbors widest path state
                neighbor_wp_state = []
                for j in self._link_lists[i]:
                    neighbor_wp_state += self._wp_dist[j]
                    # neighbor_wp_state.append(self._wp_dist[j][self._request.t]) #for less state
                neighbor_wp_state = torch.tensor(neighbor_wp_state, dtype=torch.float32)

                concat_state = torch.cat(
                    [extra_info_state, neighbor_wp_state, neighbor_dist_state, link_usage_state, link_loss_state,
                     one_hot_state], 0)  # full state
                # concat_state = torch.cat([extra_info_state, neighbor_wp_state, neighbor_dist_state, one_hot_state], 0)  # less state
                self._states.append(concat_state)

        elif self.args.agent_mode == "single_agent":

            # 源节点 src one-hot编码
            src_state = torch.tensor(list(np.eye(self._node_num)[self._request.s]))

            # 目标节点 dst one-hot编码
            dst_state = torch.tensor(list(np.eye(self._node_num)[self._request.t]))

            # generate link edge state
            link_usage_state = torch.tensor(link_usage_info, dtype=torch.float32)

            # generate link loss state
            link_loss_state = torch.tensor(link_loss_info, dtype=torch.float32)

            concat_state = torch.cat([src_state, dst_state, link_usage_state, link_loss_state])

            self._states = concat_state
    '''
    load the topo and setup the environment
    '''

    def _load_topology(self, toponame, demand_matrix_name):
        # Abi和GEA拓扑路径
        data_path = os.path.dirname(os.path.realpath(__file__)) + "/inputs/"
        # 拓扑连接关系矩阵
        topofile = open(data_path + toponame + "/" + toponame + ".txt", "r")
        # 需求矩阵
        demandfile = open(data_path + toponame + "/" + demand_matrix_name, "r")
        self._demand_matrix = list(map(int, demandfile.readline().split()))
        # the input file is undirected graph while here we use directed graph
        # node id for input file indexed from 1 while here from 0
        # 节点数量和边数量
        self._node_num, edge_num = list(map(int, topofile.readline().split()))
        # 这里当作有向边 （src dst）和 (dst, src)
        self._edge_num = edge_num * 2
        self._observation_spaces = []
        self._action_spaces = []

        # build the link list
        # 获得每个节点的邻居节点
        self._link_lists = [[] for i in range(self._node_num)]  # neighbor for each node
        self._link_capa = [([0] * self._node_num) for i in range(self._node_num)]
        self._link_usage = [([0] * self._node_num) for i in range(self._node_num)]
        self._link_losses = [([0] * self._node_num) for i in range(self._node_num)]

        for i in range(edge_num):
            u, v, _, c, loss = list(map(int, topofile.readline().split()))
            # since node index range from 1 to n in input file
            # 添加每个节点的相邻节点
            self._link_lists[u - 1].append(v - 1)
            self._link_lists[v - 1].append(u - 1)

            # undirected graph to directed graph
            # 添加链路容量
            self._link_capa[u - 1][v - 1] = c
            self._link_capa[v - 1][u - 1] = c

            # 添加链路丢包
            self._link_losses[u - 1][v - 1] = loss
            self._link_losses[v - 1][u - 1] = loss

            self.graph.add_edge(u-1, v-1, capa=c, loss=loss)
            self.graph.add_edge(v-1, u-1, capa=c, loss=loss)

        # input agent index
        is_agent = list(map(int, topofile.readline().split()))
        self._agent_to_node = []
        self._node_to_agent = [None] * self._node_num
        for i in range(self._node_num):
            if is_agent[i] == 1:
                self._node_to_agent[i] = len(self._agent_to_node)
                self._agent_to_node.append(i)
        self._agent_num = len(self._agent_to_node)

        # calculate shortest path distance
        # 初始化两个点之间的距离：
        # 1. 自己到自己的距离为0
        # 2. 自己到邻居的距离为1
        # 3. 没有直接连接边的距离为1e6
        self._shr_dist = []
        for i in range(self._node_num):
            self._shr_dist.append([])
            for j in range(self._node_num):
                if j == i:
                    self._shr_dist[i].append(0)
                elif j in self._link_lists[i]:
                    self._shr_dist[i].append(1)
                else:
                    self._shr_dist[i].append(1e6)  # inf

        # 计算每两个点之间的最短距离
        for k in range(self._node_num):
            for i in range(self._node_num):
                for j in range(self._node_num):
                    if self._shr_dist[i][j] > self._shr_dist[i][k] + self._shr_dist[k][j]:
                        self._shr_dist[i][j] = self._shr_dist[i][k] + self._shr_dist[k][j]

        # generate observation spaces and action spaces
        if self.args.agent_mode == "multi_agent":
            # 对于由智能体控制的节点生成状态空间和动作空间
            for i in self._agent_to_node:
                # state: extra_state + neighbor_wp + neighbor shr(or least delay) + linkusage + link_losses + onehot type state + onehot src + dst state
                self._observation_spaces.append(spaces.Box(0., 1.,

                                                           [1 + self._node_num * len(self._link_lists[i]) +
                                                            self._node_num * len(self._link_lists[i]) +
                                                            self._node_num ** 2 + self._node_num ** 2 +
                                                            self._type_num + self._node_num * 2],
                                                           dtype=np.float32))  # maximum observation state
                # self._observation_spaces.append(spaces.Box(0., 1., [1 + len(self._link_lists[i]) + len(self._link_lists[i]) + self._type_num + self._node_num * 2], dtype=np.float32)) # only use neighbor state space
                self._action_spaces.append(spaces.Discrete(len(self._link_lists[i])))

        elif self.args.agent_mode == "single_agent":
            # 单智能体的状态空间：
            # 网络总体延迟、丢包、链路利用率、最短距离矩阵(还是当前距离矩阵)
            # 源节点的 one-hot编码， 目的节点的 one-hot编码
            self._single_observation_spaces = spaces.Box(0., 1.,
                                                         [self._node_num ** 2 +
                                                          self._node_num ** 2 +
                                                          self._node_num * 2], dtype=np.float32)
            # k条候选路径
            self._single_action_spaces = spaces.Discrete(self.args.k_path_num)


        # TO BE CHECKED BEFORE EXPERIMENT
        # setup flow generating step
        if toponame == "Abi":

            self._request_demands = [[100], [1500], [1500], [500]]
            # 这里可以修改成字典
            self._request_times = [[50], [50], [50], [50]]  # heavy load
            # self._request_times = [[10], [10], [10], [10]] # light load
            # self._request_times = [[20], [20], [20], [20]]
            # self._request_times = [[30], [30], [30], [30]] # mid load
            # self._request_times = [[40], [40], [40], [40]]


        elif toponame == "GEA":

            # 请求时间为15s
            self._request_demands = [[100], [1500], [1500], [500]]
            self._request_times = [[15], [15], [15], [15]]

        elif toponame == "GBN":

            # 请求时间为15s
            self._request_demands = [[100], [1500], [1500], [500]]
            self._request_times = [[15], [15], [15], [15]]

    '''
        changing network status:
        link failure
        demand changing
    '''
    # 自适应测试： 链路故障测试， 流量大小测试
    def change_env(self, msg):
        if msg == "link_failure":
            # Abi link failure 1-5
            self._link_capa[0][4] = 0
            self._link_capa[4][0] = 0

            # calculate shortest path distance for new topology
            self._shr_dist = []
            for i in range(self._node_num):
                self._shr_dist.append([])
                for j in range(self._node_num):
                    if j == i:
                        self._shr_dist[i].append(0)
                    elif (j in self._link_lists[i]) and (self._link_capa[i][j] > 0):
                        self._shr_dist[i].append(1)
                    else:
                        self._shr_dist[i].append(1e6)  # inf
            for k in range(self._node_num):
                for i in range(self._node_num):
                    for j in range(self._node_num):
                        if (self._shr_dist[i][j] > self._shr_dist[i][k] + self._shr_dist[k][j]):
                            self._shr_dist[i][j] = self._shr_dist[i][k] + self._shr_dist[k][j]

        elif msg == "demand_change":
            # from light load to mid load, may be for heavy load in the future
            self._request_times = [[30], [30], [30], [30]]

        else:
            raise NotImplementedError

    '''
    calculating the Widest Path from s to t
    '''

    def calcWP(self, s, t):
        fat = [-1] * self._node_num
        WP_dist = [- 1e6] * self._node_num
        flag = [False] * self._node_num
        WP_dist[t] = 1e6
        flag[t] = True
        cur_p = t
        while flag[s] == False:
            for i in self._link_lists[cur_p]:
                # bandwidth = max(self._link_capa[i][cur_p] - self._link_usage[i][cur_p], 0)
                bandwidth = self._link_capa[i][cur_p] - self._link_usage[i][cur_p]  # for testing
                if min(bandwidth, WP_dist[cur_p]) > WP_dist[i]:
                    WP_dist[i] = min(bandwidth, WP_dist[cur_p])
                    fat[i] = cur_p
            cur_p = -1
            for i in range(self._node_num):
                if flag[i]:
                    continue
                if cur_p == -1 or WP_dist[i] > WP_dist[cur_p]:
                    cur_p = i
            flag[cur_p] = True

        path = [s]
        cur_p = 0
        while path[cur_p] != t:
            path.append(fat[path[cur_p]])
            cur_p += 1
        return path

    '''
    calculating the Shortest Path from s to t
    '''

    def calcSHR(self, s, t):
        path = [s]
        cur_p = 0
        while path[cur_p] != t:
            tmp_dist = 1e6
            next_hop = None
            # 当前节点path[cur_p]，找到当前节点邻居节点距离目标节点距离最小的点
            # 这里同时考虑了当前节点和邻居节点的带宽要大于0
            # 例如当前链路有一条链路为瓶颈链路，发送流量过大时会占用全部带宽
            for i in self._link_lists[path[cur_p]]:
                if self._shr_dist[i][t] < tmp_dist and self._link_capa[path[cur_p]][i] > 0:
                    # 只考虑距离不考虑可用带宽
                    # if self._shr_dist[i][t] < tmp_dist :
                    next_hop = i
                    tmp_dist = self._shr_dist[i][t]
            path.append(next_hop)
            cur_p += 1
        return path

    '''
    calculating the Bandwidth-Constrained Shortest Path
    '''

    def calcBCSHR(self, s, t, demand):
        fat = [-1] * self._node_num
        SHR_dist = [1e6] * self._node_num
        flag = [False] * self._node_num
        SHR_dist[t] = 0
        flag[t] = True
        cur_p = t

        while flag[s] == False:
            for i in self._link_lists[cur_p]:
                # bandwidth = max(self._link_capa[i][cur_p] - self._link_usage[i][cur_p], 0)
                bandwidth = self._link_capa[i][cur_p] - self._link_usage[i][cur_p]
                if bandwidth >= demand and SHR_dist[i] > SHR_dist[cur_p] + 1:
                    SHR_dist[i] = SHR_dist[cur_p] + 1
                    fat[i] = cur_p
            cur_p = -1
            for i in range(self._node_num):
                if flag[i]:
                    continue
                if cur_p == -1 or SHR_dist[i] < SHR_dist[cur_p]:
                    cur_p = i
            if SHR_dist[cur_p] < 1e6:
                flag[cur_p] = True
            else:
                break

        if not flag[s]:
            return None
        path = [s]
        cur_p = 0
        while path[cur_p] != t:
            path.append(fat[path[cur_p]])
            cur_p += 1
        return path

    '''
    method = "SHR"/"WP"/"DS"(diff-serv)
    '''

    def step_baseline(self, method):
        # 最短路由优先
        if method == "SHR":
            path = self.calcSHR(self._request.s, self._request.t)

        # 最大带宽
        elif method == "WP":
            path = self.calcWP(self._request.s, self._request.t)

        # 区分服务
        elif method == "DS":

            # 流量类型0和3针对延迟有要求，计算最短路径
            if self._request.rtype == 0 or self._request.rtype == 3:
                path = self.calcSHR(self._request.s, self._request.t)

            # 流量类型1 对带宽有要求，计算最大带宽路径
            elif self._request.rtype == 1:
                path = self.calcWP(self._request.s, self._request.t)

            # 计算带宽约束的最短路径
            else:
                path = self.calcBCSHR(self._request.s, self._request.t, self._request.demand)
                if path == None:
                    path = self.calcWP(self._request.s, self._request.t)

        elif method == 'QoS':
            path = self.calcBCSHR(self._request.s, self._request.t, self._request.demand)
            if path == None:
                path = self.calcWP(self._request.s, self._request.t)

        else:
            raise NotImplementedError

        self._request.path = copy.copy(path)

        # update link usage according to the selected path
        capacity = 1e9

        # 根据计算的路径更新链路容量
        for i in range(len(path) - 1):
            capacity = min(capacity,
                           max(0, self._link_capa[path[i]][path[i + 1]] - self._link_usage[path[i]][path[i + 1]]))
            self._link_usage[path[i]][path[i + 1]] += self._request.demand

        count = len(path) - 1

        # install rules for path and generate service request in sim-env(ryu + mininet)
        ret_data = self.sim_interact(self._request, path)

        delay = ret_data['delay']
        throughput = ret_data['throughput']
        loss_rate = ret_data['loss']

        delta_dist = count - self._shr_dist[self._request.s][self._request.t]
        delta_demand = min(capacity, self._request.demand) / self._request.demand
        throughput_rate = throughput / self._request.demand
        rtype = self._request.rtype

        max_link_util = 0.
        for i in range(self._node_num):
            for j in range(self._node_num):
                if self._link_capa[i][j] > 0:
                    max_link_util = max(max_link_util, self._link_usage[i][j] / self._link_capa[i][j])

        print("max link utility:", max_link_util)
        # generate new state
        self._time_step += 1
        self._update_state()
        return rtype, delta_dist, delta_demand, delay, throughput_rate, loss_rate


if __name__ == "__main__":

    # 可以遍历所有的传统方法，计算对应的值
    toponame = sys.argv[1] # 拓扑名称
    method = sys.argv[2]  # baseline method can be SHR | WP | DS | QoS
    request_time = int(sys.argv[3]) # request times 10、20、30、40、50
    num_step = int(sys.argv[4]) # 测试时间长度 默认10000个 flow request


    # 根据拓扑名称加载流量矩阵
    demand_matrix = None

    if toponame == "Abi":
        demand_matrix = "Abi_500.txt"

    elif toponame == "GEA":
        demand_matrix = "GEA_500.txt"

    elif toponame == "GBN":
        demand_matrix = "GBN_500.txt"

    # setup env
    args = None
    envs = NetEnv(args)

    # 智能体个数 节点个数 状态空间 动作空间 服务类型
    num_agent, num_node, observation_spaces, action_spaces, num_type = envs.setup(toponame, demand_matrix)
    envs.reset()

    # open log file
    log_dir = "../log/%s_%s_%d_simenv_heavyload_1-5loss/" % (toponame, method, num_step)
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.log'))
        for f in files:
            os.remove(f)

    log_dist_files = []
    log_demand_files = []
    log_delay_files = []
    log_throughput_files = []
    log_loss_files = []


    for i in range(num_type):
        log_dist_file = open("%s/dist_type%d.log" % (log_dir, i), "w", 1)
        log_dist_files.append(log_dist_file)
        log_demand_file = open("%s/demand_type%d.log" % (log_dir, i), "w", 1)
        log_demand_files.append(log_demand_file)
        log_delay_file = open("%s/delay_type%d.log" % (log_dir, i), "w", 1)
        log_delay_files.append(log_delay_file)
        log_throughput_file = open("%s/throughput_type%d.log" % (log_dir, i), "w", 1)
        log_throughput_files.append(log_throughput_file)
        log_loss_file = open("%s/loss_type%d.log" % (log_dir, i), "w", 1)
        log_loss_files.append(log_loss_file)

    start_time = time.time()

    for i in range(num_step):
        print("step:", i)
        rtype, delta_dist, delta_demand, delay, throughput_rate, loss_rate = envs.step_baseline(method)
        print(delta_dist, file=log_dist_files[rtype])
        print(delta_demand, file=log_demand_files[rtype])
        print(delay, file=log_delay_files[rtype])
        print(throughput_rate, file=log_throughput_files[rtype])
        print(loss_rate, file=log_loss_files[rtype])

    end_time = time.time()
    print(f"cost time:{end_time - start_time}")