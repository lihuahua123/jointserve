import logging
from dataclasses import dataclass
from typing import List, Dict

import torch

from sglang.srt.memory_pool import TokenToKVPool
from sglang.srt.models.lora import get_mapped_params
import triton
import triton.language as tl
logger = logging.getLogger("infer_adapter")


@dataclass
class InferAdapter:
    adapter_uids: List[str] # list of active adapters
    lora_idx: Dict[str, int] # adapter uid -> index in adapter_uids
    token_to_kv_pool: TokenToKVPool
    a_loc: torch.Tensor  # a_loc[i] is a list of indices occupied by adapter i
    a_start: torch.Tensor  # a_start[i] is the start location of adapter i
    a_len: torch.Tensor  # a_len[i] is the number of cells occupied by adapter i
    a_scaling: torch.Tensor  # a_scaling[i] is the scaling factor of adapter i

    @classmethod
    def init(cls, token_to_kv_pool):
        return cls(
            adapter_uids=[],
            lora_idx={},
            token_to_kv_pool=token_to_kv_pool,
            a_loc=torch.empty(0, dtype=torch.long, device="cuda"),
            a_start=torch.empty(0, dtype=torch.long, device="cuda"),
            a_len=torch.empty(0, dtype=torch.long, device="cuda"),
            a_scaling=torch.empty(0, dtype=torch.float16, device="cuda"),
        )

    def add_zero_lora(self):
        pass


    def load_lora(self, adapter, loc):
        for i in range(adapter.base_config.num_hidden_layers):
            adapter.layers[i].load_to_gpu(mode="paged")
            w_combined = adapter.layers[i].w_combined
            self.token_to_kv_pool.kv_data[i][loc] = w_combined
            # 前面两句已经指向了有内存空间的w_combined，因此可以将原本的释放了
            adapter.layers[i].offload_from_gpu(mode="paged")


    def load(self, adapters):
        if len(adapters) == 0:
            logger.info(f"load 0 adapters, {len(self.adapter_uids)} in total")
            return

        new_adapters = []
        tot_size = 0
        for adapter in adapters:
            if adapter is not None and adapter.uid not in self.lora_idx:
                new_adapters.append(adapter)
                tot_size += adapter.r * len(adapter.paged_modules)
        logger.info(f"load {len(new_adapters)} adapters, {len(self.adapter_uids) + len(new_adapters)} in total")

        new_loc = self.token_to_kv_pool.alloc(tot_size)
        assert new_loc is not None, "no space for new adapters"
        start_offset = self.a_start.shape[0]
        self.a_start = torch.cat((self.a_start, torch.empty(len(new_adapters,), dtype=torch.long, device="cuda")))
        len_offset = self.a_len.shape[0]
        self.a_len = torch.cat((self.a_len, torch.empty(len(new_adapters,), dtype=torch.long, device="cuda")))
        loc_offset = self.a_loc.shape[0]
        self.a_loc = torch.cat((self.a_loc, torch.empty(tot_size, dtype=torch.long, device="cuda"))) 

        cum_loc = 0
        cum_loc_list = []
        for i, new_adapter in enumerate(new_adapters):
            cum_loc_list.append(cum_loc)
            self.lora_idx[new_adapter.uid] = len(self.adapter_uids)
            self.adapter_uids.append(new_adapter.uid)
            self.a_start[start_offset + i] = loc_offset + cum_loc
            num_loc = new_adapter.r * len(new_adapter.paged_modules)
            self.a_len[len_offset + i] = num_loc
            self.a_loc[loc_offset + cum_loc: loc_offset + cum_loc + num_loc] = (
                    new_loc[cum_loc: cum_loc + num_loc])
            cum_loc += num_loc
        self.a_scaling = torch.cat((self.a_scaling, torch.tensor([adapter.scaling for adapter in new_adapters], dtype=torch.float16, device="cuda")))

        for i, new_adapter in enumerate(new_adapters):
            cum_loc = cum_loc_list[i]
            self.load_lora(new_adapter, new_loc[cum_loc: cum_loc + self.a_len[len_offset + i]])

    def offload_adapters(self, reserve_adapter_dirs):
        if len(reserve_adapter_dirs) == len(self.adapter_dirs):
            print(f"offload 0 adapters, {len(self.adapter_dirs)} remains")
            return
        if len(reserve_adapter_dirs) == 0:
            print(f"offload {len(self.adapter_dirs)} adapters, 0 remains")
            self.mem_manager.free(self.a_loc)
            self.adapter_dirs=[]
            self.a_loc=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_start=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_len=torch.empty(0, dtype=torch.long, device="cuda")
            self.a_scaling=torch.empty(0, dtype=torch.float16, device="cuda")
            self.idx_map={}
            return

        # mark_start("offload scan")
        remove_ind = []
        left_ind = []
        new_adapter_dirs = []
        self.idx_map = {}
        for i, adapter_dir in enumerate(self.adapter_dirs):
            if adapter_dir not in reserve_adapter_dirs:
                remove_ind.append(self.a_loc[self.a_start[i]:self.a_start[i] + self.a_len[i]])
            else:
                left_ind.append(i)
                self.idx_map[adapter_dir] = len(new_adapter_dirs)
                new_adapter_dirs.append(adapter_dir)
        if len(remove_ind) == 0:
            return
        # mark_end("offload scan")
        self.adapter_dirs = new_adapter_dirs
        tot_size = torch.sum(self.a_len[left_ind]).item()
        print(f"offload {len(remove_ind)} adapters, {len(left_ind)} remains")

        # mark_start("offload cat")
        remove_ind = torch.cat(remove_ind)
        # mark_end("offload cat")
        # release memory
        # mark_start("offload free mem manager")
        self.token_to_kv_pool.dec_refs(remove_ind)
        # mark_end("offload free mem manager")
        
        # reset indexing
        # mark_start("offload torch.empty")
        new_a_len = torch.empty(len(left_ind), dtype=torch.long, device="cuda")
        new_a_start = torch.empty(len(left_ind), dtype=torch.long, device="cuda")
        new_a_scaling = torch.empty(len(left_ind), dtype=torch.float16, device="cuda")
        new_a_loc = torch.empty(tot_size, dtype=torch.long, device="cuda")
        # mark_end("offload torch.empty")

        new_a_len[:] = self.a_len[left_ind]
        new_a_start[0] = 0
        new_a_start[1:] = torch.cumsum(new_a_len, dim=0)[:-1]
        new_a_scaling[:] = self.a_scaling[left_ind]
        # mark_start("offload a_loc update")
        launch_var_len_copy_triton(self.a_start[left_ind], new_a_len,
                                   self.a_loc, new_a_start, new_a_loc)
        # mark_end("offload a_loc update")

        self.a_start = new_a_start
        self.a_len = new_a_len
        self.a_loc = new_a_loc
        self.a_scaling = new_a_scaling

@triton.jit
def var_len_copy_kernel_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                               BLOCK_SIZE: tl.constexpr):
    a_id = tl.program_id(0)
    length = tl.load(old_a_len + a_id)
    old_start = tl.load(old_a_start + a_id)
    new_start = tl.load(new_a_start + a_id)
    old_offset = tl.arange(0, BLOCK_SIZE)
    new_offset = tl.arange(0, BLOCK_SIZE)
    for i in range(0, length, BLOCK_SIZE):
        v = tl.load(old_a_location + old_start + i + old_offset, mask=old_offset < length)
        tl.store(new_a_location + new_start + i + new_offset, v, mask=new_offset < length)


def launch_var_len_copy_triton(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    BLOCK_SIZE = 256
    grid_size = (len(old_a_start),)

    var_len_copy_kernel_triton[grid_size](
        old_a_start, old_a_len, old_location, new_a_start, new_a_location, BLOCK_SIZE)