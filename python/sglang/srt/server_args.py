"""The arguments of the server."""

import argparse
import dataclasses
from typing import List, Optional, Union


@dataclasses.dataclass
class ServerArgs:
    # Model and tokenizer
    model_path: str
    tokenizer_path: Optional[str] = None
    load_format: str = "auto"
    tokenizer_mode: str = "auto"
    chat_template: Optional[str] = None
    trust_remote_code: bool = True
    context_length: Optional[int] = None

    # Port
    host: str = "127.0.0.1"
    port: int = 30000
    additional_ports: Optional[Union[List[int], int]] = None

    # Memory and scheduling
    mem_fraction_static: Optional[float] = None
    max_prefill_num_token: Optional[int] = None
    schedule_heuristic: str = "lpm"
    schedule_conservativeness: float = 1.0

    # Other runtime options
    tp_size: int = 1
    stream_interval: int = 8
    random_seed: int = 42

    # Logging
    log_level: str = "info"
    log_requests: bool = False
    disable_log_stats: bool = False
    log_stats_interval: int = 10
    show_time_cost: bool = False

    # Other
    api_key: str = ""

    # Optimization/debug options
    enable_flashinfer: bool = False
    attention_reduce_in_fp32: bool = False
    disable_radix_cache: bool = False
    disable_regex_jump_forward: bool = False
    disable_disk_cache: bool = False
    cuda_devices: Optional[List[int]] = None
    metrics_buffer_size: int = 5
    freeze: bool = False
    log_prefix_hit: bool = False
    api_key: str = ""
    chunk_prefill_budget: int = 0
    hit_trace_window_size: int = 30 # seconds
    report_hit_ratio: bool = True
    enable_iterative_eviction: bool = False
    enable_partial_eviction: bool = False
    lora_paths: List[str] = None

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        if self.mem_fraction_static is None:
            if self.tp_size >= 8:
                self.mem_fraction_static = 0.80
            elif self.tp_size >= 4:
                self.mem_fraction_static = 0.82
            elif self.tp_size >= 2:
                self.mem_fraction_static = 0.85
            else:
                self.mem_fraction_static = 0.90
        if isinstance(self.additional_ports, int):
            self.additional_ports = [self.additional_ports]
        elif self.additional_ports is None:
            self.additional_ports = []

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            required=True,
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            default=ServerArgs.tokenizer_path,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--host", type=str, default=ServerArgs.host, help="The host of the server."
        )
        parser.add_argument(
            "--port", type=int, default=ServerArgs.port, help="The port of the server."
        )
        parser.add_argument(
            "--additional-ports",
            type=int,
            nargs="*",
            default=[],
            help="Additional ports specified for the server.",
        )
        parser.add_argument(
            "--load-format",
            type=str,
            default=ServerArgs.load_format,
            choices=["auto", "pt", "safetensors", "npcache", "dummy"],
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the pytorch bin format if safetensors format "
            "is not available. "
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values, '
            "which is mainly for profiling.",
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=ServerArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer if available, and 'slow' will "
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--chat-template",
            type=str,
            default=ServerArgs.chat_template,
            help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
        )
        parser.add_argument(
            "--context-length",
            type=int,
            default=ServerArgs.context_length,
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
        )
        parser.add_argument(
            "--mem-fraction-static",
            type=float,
            default=ServerArgs.mem_fraction_static,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--max-prefill-num-token",
            type=int,
            default=ServerArgs.max_prefill_num_token,
            help="The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length.",
        )
        parser.add_argument(
            "--schedule-heuristic",
            type=str,
            default=ServerArgs.schedule_heuristic,
            help="Schudule mode: [lpm, weight, random, fcfs, fcfs-mpq]",
        )
        parser.add_argument(
            "--schedule-conservativeness",
            type=float,
            default=ServerArgs.schedule_conservativeness,
            help="How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
        )
        parser.add_argument(
            "--tp-size",
            type=int,
            default=ServerArgs.tp_size,
            help="Tensor parallelism size.",
        )
        parser.add_argument(
            "--stream-interval",
            type=int,
            default=ServerArgs.stream_interval,
            help="The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            default=ServerArgs.random_seed,
            help="Random seed.",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="Logging level",
        )
        parser.add_argument(
            "--log-requests",
            action="store_true",
            help="Log all requests",
        )
        parser.add_argument(
            "--disable-log-stats",
            action="store_true",
            help="Disable logging throughput stats.",
        )
        parser.add_argument(
            "--log-stats-interval",
            type=int,
            default=ServerArgs.log_stats_interval,
            help="Log stats interval in second.",
        )
        parser.add_argument(
            "--show-time-cost",
            action="store_true",
            help="Show time cost of custom marks",
        )
        parser.add_argument(
            "--api-key",
            type=str,
            default=ServerArgs.api_key,
            help="Set API key of the server",
        )

        # Optimization/debug options
        parser.add_argument(
            "--enable-flashinfer",
            action="store_true",
            help="Enable flashinfer inference kernels",
        )
        parser.add_argument(
            "--attention-reduce-in-fp32",
            action="store_true",
            help="Cast the intermidiate attention results to fp32 to avoid possible crashes related to fp16.",
        )
        parser.add_argument(
            "--disable-radix-cache",
            action="store_true",
            help="Disable RadixAttention",
        )
        parser.add_argument(
            "--disable-regex-jump-forward",
            action="store_true",
            help="Disable regex jump-forward",
        )
        parser.add_argument(
            "--disable-disk-cache",
            action="store_true",
            help="Disable disk cache to avoid possible crashes related to file system or high concurrency.",
        )
        parser.add_argument(
            "--cuda-devices",
            help="Cuda devices.",
        )
        parser.add_argument(
            "--metrics-buffer-size",
            type=int,
            default=ServerArgs.metrics_buffer_size,
            help="The buffer size for storing metrics",
        )
        parser.add_argument(
            "--freeze",
            action="store_true",
            help="Whether the server should loop forward",
        )
        parser.add_argument(
            "--log-prefix-hit",
            action="store_true",
            help="Whether the server should log prefix hit",
        )
        parser.add_argument(
            '--chunk-prefill-budget',
            type=int,
            default=ServerArgs.chunk_prefill_budget,
            help='The maximum number of tokens that can be scheduled in a single iteration '
                 'Set to 0 to disable chunking.',
        )
        parser.add_argument(
            '--hit-trace-window-size',
            type=int,
            default=ServerArgs.hit_trace_window_size,
            help='The size of the window for hit trace in seconds.',
        )
        parser.add_argument(
            '--report-hit-ratio',
            action='store_true',
            help='Whether to return hit ratio. If disabled will always return 0, which disables hot/cold mixed scheduling.',
        )
        parser.add_argument(
            '--enable-iterative-eviction',
            action='store_true',
            help='Enable iterative eviction feedback.',
        )
        parser.add_argument(
            '--enable-partial-eviction',
            action='store_true',
            help='Enable partial eviction feedback.',
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def url(self):
        return f"http://{self.host}:{self.port}"

    def print_mode_args(self):
        return (
            f"enable_flashinfer={self.enable_flashinfer}, "
            f"attention_reduce_in_fp32={self.attention_reduce_in_fp32}, "
            f"disable_radix_cache={self.disable_radix_cache}, "
            f"disable_regex_jump_forward={self.disable_regex_jump_forward}, "
            f"disable_disk_cache={self.disable_disk_cache}, "
        )


@dataclasses.dataclass
class PortArgs:
    tokenizer_port: int
    router_port: int
    detokenizer_port: int
    nccl_port: int
    migrate_port: int
    model_rpc_ports: List[int]
