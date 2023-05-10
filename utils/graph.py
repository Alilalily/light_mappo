import numpy as np
import networkx as nx

from itertools import islice

class Graph(object):
    """
    环境中用到的网络结构以及数据
    """
    def __init__(self):
        self.node_num = 14
        self.link_num = 42
        self.topo = np.array([	#0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
                    [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0],#0
                    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0],#1
                    [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0],#2
                    [0 ,1, 0, 0, 1, 0, 0, 0, 0, 0, 1,  0,  0,  0],#3
                    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,  0,  0,  0],#4
                    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0,  0,  1,  0],#5
                    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,  0,  0,  0],#6
                    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,  0,  0,  0],#7
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,  1,  0,  1],#8
                    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,  0,  0,  0],#9 
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,  1,  0,  1],#10
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,  0,  0,  1],#11
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  1],#12
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,  1,  1,  0]])#13
        self.nx_g = nx.from_numpy_array(self.topo)
        self.nx_g = nx.DiGraph(self.nx_g)

        self.link_delays = np.array([ 8, 18, 10, 13, 22, 16, 19, 11, 18, 20, 10, 14, 15, 10, 15, 11, 15,20, 18, 10, 12, 14, 16, 14, 13, 18, 21, 13, 13, 18, 12, 13, 13, 15, 11, 19, 18, 12, 19, 12, 12, 19])
        for i, (u, v) in enumerate(self.nx_g.edges()):
            self.nx_g[u][v]['id'] = i
            self.nx_g[u][v]
            self.nx_g[u][v]['delay'] = self.link_delays[i]
        
        self.BN= nx.edge_betweenness(self.nx_g)
        nx.set_edge_attributes(self.nx_g, self.BN, "betweenness")

    def get_all_path(self):
        paths = []
        for i in range(self.node_num):
            # paths_id = []
            for j in range(i,self.node_num):
                if(i<j):
                    k_path =  self.k_shortest_paths(self.nx_g,i,j)
                    paths_id = []
                    for p in k_path:
                        id = []
                        for k in range(0,(len(p)-1)):
                            id.append(self.nx_g[p[k]][p[k+1]]["id"])
                        paths_id.append(id)
                    paths.append(paths_id)
        return paths

    def k_shortest_paths(G, source, target, k = 3, weight=None):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

    def has_path():
        pass

    def is_connectivity():
        pass
    




# bw_real = [8,32,64]  # 只有在计算剩余可用带宽和计算reward的数值时时这个，其他时候都是0，1，2
