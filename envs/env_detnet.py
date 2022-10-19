
from utils.detnet import DetNet
class EnvDetnet(object):
    def __init__(self):
        self.detnet = DetNet() # 确定性网络
        self.agent_num = self.detnet.link_num # agent 个数还需要在config文件中修改
        self.obs_dim = 1 + self.detnet.que_num # 此处的obs 设计为 flow.pkg  q1.reserv  q2.reserv
        self.action_dim = 1 + self.detnet.que_num # 此处的action 设计为 [0, 1, 2] 分别代表 不选, 选q1, 选q2
        self.flow = [] # 环境中产生的确定性网络流，在每次更新obs的时候出现
	
    
    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        # print("-----------is reset---------")
        self.flow = self.detnet.get_flow()
        src, dst, period, pkg_len, delay, offset = self.flow

        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = self.detnet.get_obs(agent_id=i, pkg_len=pkg_len, offset=offset, is_reset=True)
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs
    
    def step(self, actions):
        # print(actions)
        reward = self.detnet.update_state(flow=self.flow, actions=actions) # 判断动作，获得reward
        self.flow = self.detnet.get_flow() # 产生下一条流
        src, dst, period, pkg_len, delay, offset = self.flow
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        # 进行合法性判断，看是否满足需求，并把结果回送到detnet，更新资源值，获得新的obs
        for i in range(self.agent_num):
            sub_agent_obs.append(self.detnet.get_obs(agent_id=i, pkg_len=pkg_len, offset=offset, is_reset=False))
            sub_agent_reward.append([0])
            sub_agent_done.append(False)
            sub_agent_info.append({})
        
        
        
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]