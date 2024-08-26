from .radix_cache import RadixCache,TreeNode
import numpy as np
from collections import defaultdict
import threading

class GlobalScheduler:
    def __init__(self,num_gpus):
        self.lock = threading.Lock()
        self.num_gpus = num_gpus
        self.tree_cache = RadixCache()
        self.node_to_GPU = defaultdict(set) # 每个node对应的GPU sets
        self.req_to_node = defaultdict(TreeNode)
        self.per_gpu_load = [0 for i in range(num_gpus)]
        self.node_to_GPU[self.tree_cache.root_node]={i for i in range(num_gpus)}
        model_ids = ["lora1","lora2","base"]
        self.lora_to_GPU = { model_ids[i]:{i%self.num_gpus} for i in range(len(model_ids))}
        # root model
        self.lora_to_GPU["/hy-tmp/"] = {i for i in range(num_gpus)}
        

    def runtime_selector(self,req_id,token_ids,model_id=None):
        # insert 返回的是匹配成果的prefix_len 和插入的node
        
        with self.lock:
            token_ids = [str(model_id)] + token_ids
            new_prefix_len, new_node =  self.tree_cache.insert(token_ids) # 为什么要token_ids而不能是字符串，因为不好分词比如Hello 是否可以拆分是无法得知的
            self.req_to_node[req_id] = new_node
            self.tree_cache.inc_lock_ref(new_node)
            now = new_node
            #if self.node_to_GPU[now] is None: # 最后是一个新的节点还没分配GPU
            # 向上找父亲
            while self.node_to_GPU[now] is None: # 有可能这个node被GPU驱逐了，所以node_toGPU是None
                if now.parent is not None:
                    now = now.parent
                else:
                    break
            if len(self.node_to_GPU[now])==0 or now == self.tree_cache.root_node:
                print("not find:",now,self.node_to_GPU[now],self.per_gpu_load)
                runtime_id = int(np.argmin([self.per_gpu_load[gpu] for gpu in self.lora_to_GPU[model_id]]))
                # runtime_id = int(np.argmin(self.per_gpu_load))
            else:  
                print("find:",now,self.node_to_GPU[now],self.per_gpu_load)
                runtime_id = int(np.argmin([self.per_gpu_load[gpu] for gpu in self.node_to_GPU[now]]))
            # update_gpu_allocation_for_parent
            now = new_node
            while now is not None:
                self.node_to_GPU[now].add(runtime_id)
                now = now.parent
            self.per_gpu_load[runtime_id] += 1
        return runtime_id

    def finish_request(self,req_id,runtime_id):
        with self.lock:
            self.tree_cache.dec_lock_ref(self.req_to_node[req_id])
            self.per_gpu_load[runtime_id] -= 1
        # 现在默认不驱逐
        