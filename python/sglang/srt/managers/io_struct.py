import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from sglang.srt.sampling_params import SamplingParams
from sglang.srt.managers.router.infer_batch import Req


@dataclass
class GenerateReqInput:
    # The input prompt
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can either specify text or input_ids
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The image input
    image_data: Optional[Union[List[str], str]] = None
    # The sampling_params
    sampling_params: Union[List[Dict], Dict] = None
    # The request id
    rid: Optional[Union[List[str], str]] = None
    # Whether to return logprobs
    return_logprob: Optional[Union[List[bool], bool]] = None
    # The start location of the prompt for return_logprob
    logprob_start_len: Optional[Union[List[int], int]] = None
    # The number of top logprobs to return
    top_logprobs_num: Optional[Union[List[int], int]] = None
    # Whether to detokenize tokens in logprobs
    return_text_in_logprobs: bool = False
    # Whether to stream output
    stream: bool = False
    lora_uid: str = None
    need_cache: bool = False
    # TODO: make all parameters a Union[List[T], T] to allow for batched requests

    def post_init(self):

        if self.text is None:
            assert (
                self.input_ids is not None
            ), "Either text or input_ids should be provided"
        else:
            assert self.input_ids is None, "Either text or input_ids should be provided"

        if self.text is not None:
            is_single = isinstance(self.text, str)
        else:
            is_single = isinstance(self.input_ids[0], int)
        self.is_single = is_single

        if is_single:
            if self.sampling_params is None:
                self.sampling_params = {}
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.return_logprob is None:
                self.return_logprob = False
            if self.logprob_start_len is None:
                self.logprob_start_len = 0
            if self.top_logprobs_num is None:
                self.top_logprobs_num = 0
        else:
            num = len(self.text) if self.text is not None else len(self.input_ids)

            if self.image_data is None:
                self.image_data = [None] * num
            elif not isinstance(self.image_data, list):
                self.image_data = [self.image_data] * num

            if self.sampling_params is None:
                self.sampling_params = [{}] * num
            elif not isinstance(self.sampling_params, list):
                self.sampling_params = [self.sampling_params] * num

            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(num)]
            else:
                assert isinstance(self.rid, list)

            if self.return_logprob is None:
                self.return_logprob = [False] * num
            elif not isinstance(self.return_logprob, list):
                self.return_logprob = [self.return_logprob] * num

            if self.logprob_start_len is None:
                self.logprob_start_len = [0] * num
            elif not isinstance(self.logprob_start_len, list):
                self.logprob_start_len = [self.logprob_start_len] * num

            if self.top_logprobs_num is None:
                self.top_logprobs_num = [0] * num
            elif not isinstance(self.top_logprobs_num, list):
                self.top_logprobs_num = [self.top_logprobs_num] * num


@dataclass
class TokenizedGenerateReqInput:
    rid: str
    input_text: str
    input_ids: List[int]
    pixel_values: List[float]
    image_hash: int
    image_size: List[int]
    sampling_params: SamplingParams
    return_logprob: bool
    logprob_start_len: int
    top_logprobs_num: int
    stream: bool
    arrival_time: float
    append_to_queue_time: float = 0.0
    lora_uid: str = None
    need_cache: bool = False

@dataclass
class SchedulingMetricsReqInput:
    rid: str
    input_ids: List[int]
    tokenizer_dispatch_time: float
    manager_recv_time: float

@dataclass
class SchedulingMetricsOut:
    rid: str
    waiting_queue_len: int
    running_req_len: int
    prefix_match_len: int
    token_kv_available_size: int
    evicatable_size: int
    tree_cache_metrics_hit: int
    tree_cache_metrics_total: int
    input_len: int
    total_radix_cache_processing_time: float
    queue_processing_time: float
    inner_router_time: float
    waiting_time_tokenizer_manager: float
    matching_overhead: float
    manager_dispatch_time: float
    manager_recv_time: float
@dataclass
class CacheMetricsReqInput:
    pass
@dataclass
class BatchTokenIDOut:
    rids: List[str]
    output_tokens: List[List[int]]
    output_and_jump_forward_strs: List[str]
    hit_stop_str: List[Optional[str]]
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    meta_info: List[Dict]
    finished: List[bool]
    
    def brief(self):
        return {
            "finished": self.finished,
            "output_tokens": [len(tokens) for tokens in self.output_tokens],
        }


@dataclass
class BatchStrOut:
    rids: List[str]
    output_str: List[str]
    meta_info: List[Dict]
    finished: List[bool]
    
@dataclass
class MigrationReq:
    requets: List[Req]
    radix_cache = None

@dataclass
class FlushCacheReq:
    pass


@dataclass
class DetokenizeReqInput:
    input_ids: List[int]
    
@dataclass
class DumpTrace:
    fpath: str

@dataclass
class PrefixHitInspect:
    random_id: str
    windowed: bool = True
    hit_ratio: float = 0.0