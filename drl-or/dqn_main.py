import argparse
from net_env.my_simenv import NetEnv
import torch
from dqn.dqn_agent import DQNAgent

def get_args():

    parser = argparse.ArgumentParser(description='RL')


    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")

    # 智能体的模式 单智能体或者多智能体
    # 多智能体情况下每个智能体选择下一跳的节点 组成路径 path
    # 单智能体情况下智能体从K条候选路径中选择一条 path
    parser.add_argument('--agent_mode', type=str, default="single_agent",
                        help='agent mode(single agent or multi agent)')
    # 单智能体的动作空间
    parser.add_argument('--k_path_num', type=int, default=5,
                        help='k candidate path for src to route all paths')

    # 是否区分不同类型的流量，当type为1时不区分流量，主要是针对强化学习的奖励函数设计
    parser.add_argument('--service_type_num', type=int, default=1,
                        help='service type for flows')
    # 拓扑环境名称 Abi、GEA、GBN
    parser.add_argument('--env_name', type=str, default="Abi",
                        help='environment name')
    # 流量矩阵
    parser.add_argument("--demand-matrix", default='test.txt',
                        help='demand matrix input file name (default:test.txt)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # 模型加载路径
    parser.add_argument("--model-load-path", default=None,
                        help='load model parameters from the model-load-path')
    # 模型保存路径
    parser.add_argument("--model-save-path", default=None,
                        help='save model parameters at the model-save-path')


    args = parser.parse_args()
    # 不使用cuda 并且 cuda 可用
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def main():

    args = get_args()

    print(f"agent_mode: {args.agent_mode} ")
    print(f"k_path_num: {args.k_path_num} ")
    print(f"service type num: {args.service_type_num}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # for reproducible
    # 检查 cuda 是否可用
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env = NetEnv(args)

    if args.agent_mode == "single_agent":
        observation_space, action_space = env.setup(args.env_name, args.demand_matrix)
        print(f"agent observation_space {observation_space}, action spaces {action_space}")
        state_size = observation_space.shape[0]
        action_size = action_space.n
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        # 初始化环境
        request, obses = env.reset()

        print(f"request s:{request.s}, request t:{request.t}, obses:{obses.shape}")
        action = agent.choose_action(obses)
        env.step_single_agent(action, simenv=True)

    elif args.agent_mode == "multi_agent":
        num_agent, num_node, observation_spaces, action_spaces, num_type = env.setup(args.env_name, args.demand_matrix)
        print(f"num_agent:{num_agent}, num_node:{num_node}, observation space:{observation_spaces}"
              f"action spaces:{action_spaces}, num_type:{num_type}")

    # 针对离散空间可以选择的 DRL算法： DQN、PPO、A2C
    # 针对网络状态特征提取可以使用 GNN（要对节点特征embedding） 或者是 MLP (构建成一唯向量的格式)
    # 初始化 dqn 智能体


    # 预训练

    # 训练


if __name__ == "__main__":

    main()