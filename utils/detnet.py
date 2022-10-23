import numpy as np
import networkx as nx
from utils.graph import Graph
class DetNet:
    """
    """
    def __init__(self):
        self.topo = Graph() # 拓扑图
        self.schedule_period = 9 # 调度周期 是所有流周期的最小公倍数
        self.slot = 125 # 单位是微秒
        self.min_pkg_len = 600 # 最大的数据包长度 单位 byte
        self.max_pkg_len = 1500
        self.que_len = 15000 # 循环队列的长度
        self.que_num = 2 # 循环队列的数量

        self.node_num = self.topo.node_num
        self.link_num = self.topo.link_num

        self.flow_periods = [2, 3, 4, 5, 6, 7, 8, 9] # 流的周期范围，几倍的slot
        self.max_delay = 200
        self.min_delay = 80

        self.active_links = [] # 当前还有资源能用的link编号集合
        
        self.edge_que = np.full((self.link_num, self.que_num, self.schedule_period), self.que_len, dtype=int)
        # for link in range(self.link_num):
        #     ques = []
        #     ques.append(np.full((self.que_num,self.schedule_period), 2, dtype=int))
        #     self.edge_que.append(ques)
    

    def get_flow(self):
        """
        生成确定想需求流flow[src, dst, period, pkg_len, delay, offset]

        TODO: 下一步要做的: 从没有done掉的节点中产生需求, 可做可不做
        """
        src = np.random.randint(0, self.node_num)
        dst = np.random.randint(0, self.node_num)
        while dst == src:
            dst = np.random.randint(0, self.node_num)
        
        length, path = nx.bidirectional_dijkstra(self.topo.nx_g, src, dst,weight="delay")
        period = np.random.randint(2, 9)
        pkg_len = np.random.randint(self.min_pkg_len, self.max_pkg_len)
        # delay = np.random.randint(self.min_delay, self.max_delay) 
        delay = np.random.randint(length, self.max_delay)
        offset = np.random.randint(0, self.schedule_period)
        return [src, dst, period, pkg_len, delay, offset]

    
    def get_obs(self, agent_id, pkg_len, offset, is_reset):
        """
        生成边id对应agent的obs 包长pkg_len 队列对应周期 可用的剩余资源 
        """
        obs = []
        if is_reset:
            self.edge_que = np.full((self.link_num, self.que_num, self.schedule_period), self.que_len, dtype=int)
        obs.append(pkg_len)
        for i in range(self.que_num):
            # obs.append(self.edge_que[agent_id][i][(offset + i + 1) % self.schedule_period])
            for j in range(self.schedule_period):
                obs.append(self.edge_que[agent_id][i][j])
            # obs.append(self.edge_que[agent_id][i])
        
        return obs

    def update_state(self, flow, actions):
        """
        更新确定性网络资源, 主要是更新每个节点周期队列资源占用情况
        """
        src, dst, period, pkg_len, delay, offset = flow
        # 从actions中获得每条边的action并将其从one-hot转化成数字, [0, 1, 2]
        shifts = [np.argmax(action) for action in actions]
        links = [x for x in shifts if x != 0]

        flag = False
        # 新建一个图来验证该方案是否满足需求
        f = nx.DiGraph(self.topo.nx_g)         
        to_remove = [(a,b) for a, b, attrs in self.topo.nx_g.edges(data=True) if attrs["id"] in links]
        f.remove_edges_from(to_remove)
        
        resource_copy  = self.edge_que.copy()
        
        # 判断src到dst的连通性
        if nx.has_path(f, src, dst):
            #双向搜索的迪杰斯特拉
            length, path = nx.bidirectional_dijkstra(f, src, dst,weight="delay")
            # 遍历路径，看是否满足条件
            path_delay = length + sum(shifts) * self.slot // 1000
            if path_delay <= delay: # 如果时延满足要求，则对资源可用性进行判断
                current_delay = 0
                for k in range(0, (len(path) - 1)):
                    flag = True
                    edge_id = self.topo.nx_g[path[k]][path[k + 1]]["id"]
                    shift = shifts[edge_id]
                    slot = (current_delay + shift) // self.schedule_period
                    current_delay += self.topo.link_delays[edge_id]
                    if self.edge_que[edge_id][shift - 1][slot] < pkg_len:
                        flag = False
                        break
                    else :
                        self.edge_que[edge_id][shift - 1][slot] -= pkg_len
                
        # flow = self.get_flow
        reward = 0
        if flag:
            reward = 1
        else:
            self.edge_que = resource_copy.copy()


        done = False
        if np.min(self.edge_que) < self.min_pkg_len:
            done = True
        # dones = []
        # for i in range(self.topo.link_num):
        #     is_done = False
        #     for j in range(self.que_num):
        #         if np.all(self.edge_que[i][j] < self.min_pkg_len):
        #             is_done = True
        #             break
        #     dones.append(is_done)
        return reward, done



    def is_work(self, flow, action):
        """
        判断针对flow所作出的联合动作是否可以下发
        """
        res = False
        return res

    def is_done(self, agent_id):
        """对agent i进行判断, 是否已经done了, done的条件是, 所有slot都被分配完了
        """