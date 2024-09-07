import heapq
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List
import logging

import torch

logger = logging.getLogger(__name__)
# logger.setLevel(logging.CRITICAL + 1)
@dataclass
class EvictionData():
    input_ids: list
    evicted_ids: list

class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.time()
        self.cpu_node = 0

    def __lt__(self, other: "TreeNode"):
        # delta = self.last_access_time-other.last_access_time
        # if delta == 0:
        #     return len(self.value) > len(other.value)
        # return delta < 0
        return self.last_access_time < other.last_access_time


def _key_match(key0, key1):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


class RadixCache:
    def __init__(self, max_cpu_tokens, req_to_token_pool, token_to_kv_pool, disable: bool = False, enable_partial_eviction=False):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.disable = disable
        self.reset()
        self.enable_partial_eviction = enable_partial_eviction
        logger.info("using RadixCache")

    ##### Public API #####

    def flush_evicted(self):
        self.evicted_iteration = []

    def add_node_to_evicted_iteration(self, node, num_evited_token):
        input_ids = []
        evicted_ids = []
        while node != self.root_node:
            for k, v in node.parent.children.items():
                k = [k]
                if v == node:
                    input_ids = list(k) + input_ids
                    if not evicted_ids:
                        evicted_ids = list(k[-num_evited_token:])
                    break
            node = node.parent
        self.evicted_iteration.append(EvictionData(input_ids, evicted_ids))
        

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.evicted_iteration = []

    def match_prefix(self, key):
        if self.disable:
            return [], self.root_node
        value_list = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value_list, last_node)
        if len(value_list) >0:
            value = torch.concat(value_list)
        else:
            value = torch.tensor([], dtype=torch.int64)
        return value, last_node[0]

    def match_prefix_return_str(self, key):
        return "".join(self.match_prefix(key)[0])

    def insert(self, key, value=None):
        if self.disable:
            return len(key)

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    def cache_req(
        self,
        token_ids,
        last_uncached_pos,
        req_pool_idx,
        del_in_memory_pool=True,
        old_last_node=None,
        lora_uid=None,
    ):
        # Insert the request into radix cache
        # 这时候indices已经在GPU上了
        indices = self.req_to_token_pool.req_to_token[req_pool_idx, : len(token_ids)]
        if lora_uid is not None:
            token_ids[0] = (lora_uid,token_ids[0])
        # new_prefix_len 是匹配原有的前缀的token数目
        new_prefix_len = self.insert(token_ids, indices.clone())

        # Radix Cache takes one ref in memory pool，就是说在radixcache保存了的，就不需要在token_to_kv_pool保存了？
        # 那对应的kv cache怎么办？
        # 请求第一次的时候是不会dec_refs，此时的last_uncached_pos和new_prefix_len都为0，之后kv cache就保存在这里了，即kv cache1
        # 当有重复前缀的请求到来的时候，last_uncached_pos:new_prefix_len是跟第一次请求没有重复前缀的，
        # prefill 阶段只多生成一个token，这时候dec_refs掉这个token。总之没有覆盖掉之前的，即last_uncached_pos之前的没有被覆盖掉
        free_space = self.token_to_kv_pool.dec_refs(indices[last_uncached_pos:new_prefix_len])
        # finish 请求的时候del_in_memory_pool为True
        if del_in_memory_pool:
            self.req_to_token_pool.free(req_pool_idx)
        else:
            # 因为之前插入了，这时候拿到的就是已经在GPU上的indices了
            cached_indices, new_last_node  = self.match_prefix(token_ids)
            assert len(cached_indices) == len(token_ids)
            self.dec_lock_ref(old_last_node)
            self.inc_lock_ref(new_last_node)
            self.req_to_token_pool.req_to_token[
                req_pool_idx, last_uncached_pos : len(cached_indices)
            ] = cached_indices[last_uncached_pos:]
            return cached_indices, new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")
    
    def _total_size_helper(self, node):
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x
    def total_size(self):
        return self._total_size_helper(self.root_node)
    # 正是这里和vllm有时间上的区别，vllm的链表只需要去除最旧的几个就行，这里还需要dfs 遍历+堆排序
    def evict(self, num_tokens, evict_callback, collect_evicted_node=False):
        # curr_evict = self.evictable_size()
        # start = time.perf_counter()
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0: # 只有没人引用的才会被驱逐，这是可以驱逐的块
                continue
            # 这些快可以不被完全删除，而是放到CPU上
            holding_slots = len(x.value)
            desired_eviction = min(holding_slots, num_tokens - num_evicted) if self.enable_partial_eviction else holding_slots
            free_space = evict_callback(x.value[-desired_eviction:]).item()
            num_evicted += free_space
            self._delete_leaf(x, desired_eviction)
            
            if collect_evicted_node:
                self.add_node_to_evicted_iteration(x, desired_eviction)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)
        # end = time.perf_counter()
        # print(f"Eviction: {end-start}, Amount evicted: {curr_evict - self.evictable_size()}")
                
    def total_unique_kv_tokens(self, reqs):
        seen = set()
        
        # length of token is not in tree, calculated independently regardless
        def _traverse(node, length):
            while node != self.root_node:
                if node.lock_ref > 0:
                    seen.add(node)
                    length -= len(node.value)
                node = node.parent
            return length
            
        total_tokens = 0
        for req in reqs:
            total_tokens += _traverse(req.last_node, len(req.input_ids) + len(req.output_ids))
        for node in seen:
            total_tokens += len(node.value)
        return total_tokens
    
    def total_shared_token_by_count(self):
        def _helper(node: TreeNode):
            if node.ref_counter <= 1:
                return 0
            x = len(node.value) * (node.ref_counter - 1)
            for child in node.children.values():
                x += _helper(child)
            return x
        
        sum_shared = 0
        for child in self.root_node.children.values():
            sum_shared += _helper(child)    
        return sum_shared

    def inc_lock_ref(self, node: TreeNode):
        """
        每增加一个请求引用，则增加node的lock_ref,这个node一般是GPU
        """
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0 and node.value.device.type != "cpu":
                #logger.info(f"inc_lock_ref delete {len(node.value)}, {node}")
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        """
        每减少一个请求引用，则减少node的lock_ref，这个node一般是GPU
        """
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1 and node.value.device.type != "cpu":
                # logger.info(f"dec_lock_ref add {len(node.value)}, {node}")
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta
    def _total_evictable_helper(self, node):
        x = 0
        if node.lock_ref == 0 and node.value.device.type != "cpu":
            x = len(node.value)
        for child in node.children.values():
            x += self._total_evictable_helper(child)
        return x
    def evictable_size2(self):
        return self._total_evictable_helper(self.root_node)    
    def evictable_size(self):
        """
        返回的是GPU的evictable_size，而且lock_ref要为0，不包含CPU
        """
        return self.evictable_size_

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node, key, value, last_node):
        node.last_access_time = time.time()
        if len(key) == 0:
            return

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                last_node[0] = new_node
            else:
                value.append(child.value)
                last_node[0] = child
                self._match_prefix_helper(child, key[prefix_len:], value, last_node)

    def _split_node(self, key, child: TreeNode, split_len):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:][0]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        if child.value.device.type == "cpu":
            new_node.cpu_node = 1
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[key[:split_len][0]] = new_node
        return new_node

    def _insert_helper(self, node, key, value):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)

            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value)

            new_node = self._split_node(child.key, child, prefix_len)
            return prefix_len + self._insert_helper(
                new_node, key[prefix_len:], value[prefix_len:]
            )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[key[0]] = new_node
            if value.device.type != "cpu":
                # logger.info(f"insert add {len(value)}")
                self.evictable_size_ += len(value)
        return 0

    def _print_helper(self, node: TreeNode, indent):
        for _, child in node.children.items():
            print(" " * indent, len(child.key), child.key[:10], f"r={child.lock_ref},device={child.value.device.type},{child}")
            self._print_helper(child, indent=indent + 2)

    #NOTE: tree node should not be deleted if partial eviction
    def _delete_leaf(self, node, num_evict_token):
        assert num_evict_token > 0, "num_evict_token should be greater than 0"
        del node.parent.children[node.key[0]]
        if num_evict_token < len(node.value):
            node.value = node.value[:-num_evict_token]
            node.key = node.key[:-num_evict_token]
            node.parent.children[node.key[0]] = node
        self.evictable_size_ -= num_evict_token

    
    def _collect_leaves(self):
        ret_list = []

        def dfs_(cur_node):
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list



class RadixCacheMix(RadixCache):
    def __init__(self,max_cpu_tokens,req_to_token_pool, token_to_kv_pool, disable: bool = False, enable_partial_eviction=False):
        self.max_cpu_tokens = max_cpu_tokens
        self.cur_cpu_tokens = 0
        logger.info("using RadixCacheMix")
        super().__init__(max_cpu_tokens, req_to_token_pool, token_to_kv_pool, disable, enable_partial_eviction)

    def reset(self):
        self.cur_cpu_tokens = 0
        super().reset()
        

    # def cache_req(
    #     self,
    #     token_ids,
    #     last_uncached_pos,
    #     req_pool_idx,
    #     del_in_memory_pool=True,
    #     old_last_node=None,
    #     lora_uid=None,
    # ):
    #     # Insert the request into radix cache
    #     # 这时候indices已经在GPU上了
    #     indices = self.req_to_token_pool.req_to_token[req_pool_idx, : len(token_ids)]
    #     if lora_uid is not None:
    #         token_ids[0] = (lora_uid,token_ids[0])
    #     # new_prefix_len 是匹配原有的前缀的token数目
    #     new_prefix_len = self.insert(token_ids, indices.clone())
    #     assert indices[last_uncached_pos:new_prefix_len].device.type == "cuda"
    #     self.token_to_kv_pool.dec_refs(indices[last_uncached_pos:new_prefix_len])
    #     # finish 请求的时候del_in_memory_pool为True
    #     if del_in_memory_pool:
    #         self.req_to_token_pool.free(req_pool_idx)
    #     else:
    #         cached_indices, new_last_node  = self.match_prefix(token_ids)
    #         assert len(cached_indices) == len(token_ids)
    #         self.dec_lock_ref(old_last_node)
    #         self.inc_lock_ref(new_last_node)
    #         self.req_to_token_pool.req_to_token[
    #             req_pool_idx, last_uncached_pos : len(cached_indices)
    #         ] = cached_indices[last_uncached_pos:]
            
    #         return cached_indices, new_last_node
    
    def inc_lock_ref_(self, node: TreeNode):
        """
        每增加一个请求引用，则增加node的lock_ref,这个node一般是GPU
        """
        delta = 0
        while node != self.root_node:
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref_(self, node: TreeNode):
        """
        每减少一个请求引用，则减少node的lock_ref，这个node一般是GPU
        """
        delta = 0
        while node != self.root_node:
            node.lock_ref -= 1
            node = node.parent
        return delta
    
    def match_prefix(self, key):
        if self.disable:
            return [], self.root_node
        last_node = [self.root_node]
        all_nodes = []
        self._match_prefix_helper(self.root_node, key, last_node,all_nodes)
        value_list = []
        change_node = []
        self.inc_lock_ref_(last_node[0])
        for node in all_nodes:
            if node.value.device.type == "cpu":
                # 这个节点是一个游离节点，本来就已经被删除了
                if self.move_node_to_cuda(node):
                    value_list.append(node.value)
                    change_node.append(node)
                else:
                    break
            else:
                value_list.append(node.value)
        self.dec_lock_ref_(last_node[0])
        for node in change_node:
            if node.lock_ref == 0 and node.value.device.type == "cuda":
                self.evictable_size_ += len(node.value)
        if len(value_list) >0:
            value = torch.concat(value_list)
        else:
            value = torch.tensor([], dtype=torch.int64)
        return value, last_node[0]
    
    def _match_prefix_helper(self, node, key,  last_node, all_nodes):
        node.last_access_time = time.time()
        if len(key) == 0:
            return
        if key[0] in node.children.keys():
            child = node.children[key[0]]
            child.last_access_time = time.time()
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                all_nodes.append(new_node)
                last_node[0] = new_node
            else:
                all_nodes.append(child)
                last_node[0] = child
                self._match_prefix_helper(child, key[prefix_len:],  last_node,all_nodes)
                
    
    def move_node_to_cuda(self,node):
        v = node.value
        new_index = self.token_to_kv_pool.alloc(len(v))
        if new_index is None:
            # 驱逐掉没人引用的，但有没有可能有些虽然没人引用，但是刚好和我match上了，只是我还来不及引用
            self.evict(len(v),self.token_to_kv_pool.dec_refs,check=False)
            new_index = self.token_to_kv_pool.alloc(len(v))  
            if new_index is None:
                # 没有位置给你放GPU了，那你只能适配这前面的部分了
                return False
        self.token_to_kv_pool.dec_refs(v)
        self.cur_cpu_tokens -= len(v)
        # if move_kv:
        for i in range(self.token_to_kv_pool.layer_num):
            self.token_to_kv_pool.kv_data["cuda"][i][new_index] = self.token_to_kv_pool.kv_data["cpu"][i][v].to('cuda', copy=True)
        # 我忽略了一个问题，就是这个index原本放在cuda已经被人用了，你这个时候转换过来的话v.to("cuda") 是错误的
        node.value = new_index
        return True
            
    #NOTE: tree node should not be deleted if partial eviction
    def _delete_leaf_cpu(self, node, num_evict_token):
        assert num_evict_token > 0, "num_evict_token should be greater than 0"
        del node.parent.children[node.key[0]]
        if num_evict_token < len(node.value):
            node.value = node.value[:-num_evict_token]
            node.key = node.key[:-num_evict_token]
            node.parent.children[node.key[0]] = node

    def move_node_to_cpu(self,x):
        need_cpu_space = len(x.value)
        new_cpu_indices = self.token_to_kv_pool.alloc(need_cpu_space,"cpu")
        if new_cpu_indices is None:
            return False
        # 释放GPU显存
        self.token_to_kv_pool.dec_refs(x.value).item()
        # 前面已经驱逐了need_to_evicted_cpu，按理来说应该能alloc的
        # 到这里的都是没人引用kv cache的x.lock_ref = 0，这里cuda 上的kv_data是否还有意义？token_to_kv_pool的ref>0 肯定有意义
        # 就在刚刚token_to_kv_pool.dec_refs减到0了，所以正好把他挪走
        # 走到这里来，都是free_num == len(x.value) 即x.value的ref全部为0的
        for i in range(self.token_to_kv_pool.layer_num):
            self.token_to_kv_pool.kv_data["cpu"][i][new_cpu_indices] = self.token_to_kv_pool.kv_data["cuda"][i][x.value].to('cpu', copy=True)
        x.value =  new_cpu_indices 
        self.cur_cpu_tokens += need_cpu_space
        x.parent.cpu_node += 1
        return True
        

    def _collect_leaves(self):
        cpu_list = []
        gpu_list = []
        def dfs_(cur_node:TreeNode):
            have_gpu_child = False
            if len(cur_node.children) == 0 and cur_node.lock_ref == 0 :
                if cur_node.value.device.type == "cpu":
                    cpu_list.append(cur_node)
                else:
                    have_gpu_child = True
                    gpu_list.append(cur_node)
            for x in cur_node.children.values():
                if dfs_(x):
                    have_gpu_child = True
            if len(cur_node.children) > 0 and not have_gpu_child and cur_node.lock_ref == 0 and cur_node.value.device.type != "cpu":
                gpu_list.append(cur_node)
            return have_gpu_child

        dfs_(self.root_node)
        return cpu_list,gpu_list

    def push_heap(self,x,gpu_list,cpu_list):
        if len(x.parent.children) == 0 or x.parent.cpu_node == len(x.parent.children):
            if x.parent.value.device.type == "cpu":
                heapq.heappush(cpu_list, x.parent)
            else:
                heapq.heappush(gpu_list, x.parent)

    def remove_cpu_node(self,x):
        if x.key[0] in x.parent.children.keys():
            evicted_cpu_token = len(x.value)
            self._delete_leaf_cpu(x, evicted_cpu_token)
            self.token_to_kv_pool.dec_refs(x.value)
            self.cur_cpu_tokens -= evicted_cpu_token
            return True
        return False

    def evict(self, num_tokens, evict_callback, collect_evicted_node=False):
        """
        只驱逐GPU到CPU
        """
        if self.disable:
            return
        begin = time.time()
        cpu_list, gpu_list = self._collect_leaves()
        num_gpu_evicted = 0
        heapq.heapify(gpu_list)
        free_cpu_space = self.token_to_kv_pool.available_size("cpu")
        if free_cpu_space < num_tokens:
            heapq.heapify(cpu_list)
        x: TreeNode
        while num_gpu_evicted < num_tokens:
            while len(gpu_list) == 0 and len(cpu_list) > 0:
                cpu_node = heapq.heappop(cpu_list)
                if self.remove_cpu_node(cpu_node):
                    self.push_heap(cpu_node,gpu_list,cpu_list)
            
            if len(gpu_list) == 0:
                break
            
            x = heapq.heappop(gpu_list)
            if x == self.root_node:
                break
            if x.lock_ref > 0 or x.value.device.type == "cpu": # 只有没人引用的才会被驱逐，这是可以驱逐的块
                continue
            if not self.move_node_to_cpu(x): # 没有充足的cpu位置放
                need_to_evicted_cpu_token = len(x.value)
                while need_to_evicted_cpu_token > 0:
                    if len(cpu_list) == 0:
                        logger.info(f"you need cpu space {need_to_evicted_cpu_token} but do not have enough to evicted")
                        exit()
                    cpu_node = heapq.heappop(cpu_list)
                    if self.remove_cpu_node(cpu_node):
                        self.push_heap(cpu_node,gpu_list,cpu_list)
                        need_to_evicted_cpu_token -= len(cpu_node.value)
                if not self.move_node_to_cpu(x):
                    logger.info(f"no cpu enough")
                    exit()
            self.push_heap(x,gpu_list,cpu_list)
            num_gpu_evicted += len(x.value)
            self.evictable_size_ -= len(x.value)

        end = time.time()

if __name__ == "__main__":
    tree = RadixCache(None, None, False)

    tree.insert("Hello")
    tree.insert("Hello There")
    tree.insert("Hello_L.A.!")
    # tree.insert("Hello_world! Happy")
    # tree.insert("I love you!")
    tree.pretty_print()

    # print(tree.match_prefix_return_str("Hello T"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()
