from .radix_cache import RadixCache,TreeNode
import numpy as np
from collections import defaultdict
import threading
from .utils import RequestFuncOutput
import copy

class GlobalScheduler:
    def __init__(self,num_gpus):
        self.num_gpus = num_gpus
        self.tree_cache = RadixCache()
        self.node_to_GPU = defaultdict(set) # 每个node对应的GPU sets
        self.req_to_node = defaultdict(TreeNode)
        self.per_gpu_load = [0 for i in range(num_gpus)]
        self.client_to_server = defaultdict(set) # clientid:{node1,node2,....}
        self.node_to_GPU[self.tree_cache.root_node]={i for i in range(num_gpus)}
        model_ids = ["lora1","lora2","/hy-tmp/"]
        self.locks = {model_id:threading.Lock() for model_id in model_ids}
        self.total_locks = threading.Lock()
        self.lora_to_GPU = { model_ids[i]:{i%self.num_gpus} for i in range(len(model_ids))}
        # root model
        self.lora_to_GPU["/hy-tmp/"] = {i for i in range(num_gpus)}
        self.per_gpu_load_len = [0 for i in range(num_gpus)]
        self.history = []
    
    
    def waiting_time(self,runtime_id,prefill_len,decode_len):
        pass
    def load_kv_time(self,runtime_id,prefix_len):
        pass
    def load_lora_time(self,runtime_id,model_id):
        pass
    def prefill_time(self,runtime_id,prefix_len,prefill_len):
        pass
    # def decode_time(self,runtime_id,decode_len):
    #     pass
    
    def find_best_runtime(self,runtime_ids,req_id,token_ids,model_id,cliend_id,decode_len=None,prefix_len=None):
        run_times=[]
        for runtime_id in runtime_ids:
            run_time = self.waiting_time(runtime_id,len(token_ids),decode_len)
            + self.load_kv_time(runtime_id,prefix_len)
            + self.load_lora_time(runtime_id,model_id)
            + self.prefill_time(runtime_id,prefix_len,len(token_ids))
            run_times.append(run_time)
        return [int(np.argmin(run_times))]
    
    def get_gpu_along_path(self,last_node,prefix_len):
        path_len = prefix_len
        ret_gpus = []
        gpus_set = set()
        while last_node is not None:
            for gpu in self.node_to_GPU[last_node]:
                ret_gpus.append(gpu,path_len)
                gpus_set.add(gpu)
            path_len -= last_node.value
            last_node = last_node.parent
        return ret_gpus,gpus_set

    def runtime_selector(self,req_id,token_ids,model_id=None,cliend_id=None):
        
        runtime_ids = None
        # TODO: 一致性哈希，现在只是简单记录重定向 
        # FIXME 当服务器缓存没有这个客户端的时候删除
        if cliend_id in self.client_to_server:
            runtime_ids = [(server,0) for server in self.client_to_server[cliend_id]]
            
        with self.locks[model_id]:
            token_ids[0] =  (model_id, token_ids[0])
            new_prefix_len, new_node =  self.tree_cache.insert(token_ids) # 为什么要token_ids而不能是字符串，因为不好分词比如Hello 是否可以拆分是无法得知的
            self.req_to_node[req_id] = new_node
            self.tree_cache.inc_lock_ref(new_node)
            #if self.node_to_GPU[now] is None: # 最后是一个新的节点还没分配GPU
            # 向上找父亲
            # FIXME:总是会落到同一个实例上
            if runtime_ids is None:
                # 按理来说，没有GPU对应的node应该要被删除,所以找到的new_node 一定都是带着GPU的！
                # FIXME 没有GPU对应的node应该要被删除
                if len(self.node_to_GPU[now])==0 or now == self.tree_cache.root_node:
                    #print("not find:",now,self.node_to_GPU[now],self.per_gpu_load)
                    pass
                else:  
                    #print("find:",now,self.node_to_GPU[now],self.per_gpu_load)
                    runtime_ids,gpus_set = self.get_gpu_along_path(new_node,new_prefix_len)
                    for gpu in self.lora_to_GPU[model_id]:
                        if gpu not in gpus_set:
                            runtime_ids.append(gpu,0)
                            gpus_set.add(gpu)
                    # runtime_ids = [int(np.argmin([self.per_gpu_load[gpu] for gpu in self.node_to_GPU[now]]))]
                    # FIXME: 选择次优的，而不是最优的
                    # if self.per_gpu_load[runtime_ids[0]] > 5:
                    #     runtime_ids = [int(np.argmin([self.per_gpu_load[gpu] for gpu in self.lora_to_GPU[model_id]]))]
                    # # update_gpu_allocation_for_parent
        
        with self.total_locks:  
            # 前缀没有匹配，代表全都需要重计算,注意这里model_id 可能是lora也可能是主函数
            if runtime_ids is None:
                runtime_ids = [(gpu,0) for gpu in self.lora_to_GPU[model_id]]

            runtime_ids = self.find_best_runtime(runtime_ids,req_id,token_ids,model_id,cliend_id)
            
            now = new_node
            while now is not None:
                self.node_to_GPU[now].add(runtime_ids[0])
                now = now.parent
            self.per_gpu_load[runtime_ids[0]] += 1
            self.per_gpu_load_len[runtime_ids[0]] += len(token_ids) - new_prefix_len
            self.client_to_node[cliend_id].add(runtime_ids[0])
        return runtime_ids[0], new_prefix_len

    def finish_request(self,req_id,runtime_id,output:RequestFuncOutput = None,now_per_gpu_load_len = 0):
        max_total_num = 12606 # 这个是3090 profill的 FIXME：还要减去lora的
        x1 = now_per_gpu_load_len
        x2 = output.prompt_len
        y = output.waiting_latency
        self.history.append((x1,x2,y))
        with self.total_locks:
            self.tree_cache.dec_lock_ref(self.req_to_node[req_id])
            self.per_gpu_load[runtime_id] -= 1
            self.per_gpu_load_len[runtime_id] -= output.prompt_len
        if len(self.history) > 10:
            print(self.history)
        # 现在默认不驱逐
        