from transformers import AutoTokenizer
import random
import sys, os


# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from multi_experiment_benchmark_utils import AllExperiments, ExperimentType, DefaultWorkload, ConfigurableMajorExperimentArgs

from benchmark_utils import RequestGroup
from benchmark_workload_gen import *
from sglang.srt.managers.router.model_runner import GPUConfig
from data_parallel_request_cache import DataParallelRuntimeSelectionPolicy, CustomPolicyType
import random
from multi_exp_configs.multi_exp_utils import *

model_name = "/hy-tmp/"

"""sgalng baseline server runtime config
"""
sglang_server_args = {
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 32768,
    "enable_flashinfer": True,
    'schedule_heuristic': 'lpm',
    # "chunk_prefill_budget": 512,
}
# GPU Configuration
baseline_gpu_configs = [
    # GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=0, url='http://0.0.0.0:2333', use_ssh=False, runtime_args=sglang_server_args),
    GPUConfig(gpu_id=1, url='http://0.0.0.0:2334', use_ssh=False, runtime_args=sglang_server_args),
    #GPUConfig(gpu_id=2, url=None, use_ssh=False, runtime_args=sglang_server_args),
    #GPUConfig(gpu_id=3, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=4, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=5, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=6, url=None, use_ssh=False, runtime_args=sglang_server_args),
    # GPUConfig(gpu_id=7, url=None, use_ssh=False, runtime_args=sglang_server_args),
]
add_simulation_to_gpu_config(baseline_gpu_configs)

"""ours server runtime config
"""
ours_server_args = {
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 32768,
    "enable_flashinfer": True,
    'schedule_heuristic': 'fcfs-mpq',
    "chunk_prefill_budget": 512,
    'report_hit_ratio': True ,
    'enable_iterative_eviction': True,
}

# GPU Configuration
ours_gpu_configs = [
    # GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=ours_server_args),
    # GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=0, url='http://0.0.0.0:2333', use_ssh=False, runtime_args=ours_server_args),
    GPUConfig(gpu_id=1, url='http://0.0.0.0:2334', use_ssh=False, runtime_args=ours_server_args),
]
add_simulation_to_gpu_config(ours_gpu_configs)

ours_server_args_lpm = {
    'log_prefix_hit': True,
    'mem_fraction_static': 0.8,
    'context_length': 32768,
    "enable_flashinfer": True,
    'schedule_heuristic': 'lpm',
    "chunk_prefill_budget": 512,
    'report_hit_ratio': True ,
    'enable_iterative_eviction': True,
}
# GPU Configuration
ours_gpu_configs_lpm = [
    GPUConfig(gpu_id=0, url=None, use_ssh=False, runtime_args=ours_server_args_lpm),
    GPUConfig(gpu_id=1, url=None, use_ssh=False, runtime_args=ours_server_args_lpm),
]
add_simulation_to_gpu_config(ours_gpu_configs_lpm)

exp_time = float('inf')
configuration_to_test = [
    # scale_to_gpu([100, 2000, 2], len(ours_gpu_configs) // 2),
    # scale_to_gpu([100, 2500, 0.5], len(ours_gpu_configs) // 2),
    # scale_to_gpu([100, 312, 0.5], len(ours_gpu_configs) // 2),
    # scale_to_gpu([100, 625, 1], len(ours_gpu_configs) // 2),
    # scale_to_gpu([100, 1250, 2], len(ours_gpu_configs) // 2),
    # scale_to_gpu([160, 1875, 3], len(ours_gpu_configs) // 2),

    # scale_to_gpu([160, 1562, 2.5], len(ours_gpu_configs) // 2),
    # scale_to_gpu([160, 2187, 3.5], len(ours_gpu_configs) // 2),
    # scale_to_gpu([160, 2500, 4], len(ours_gpu_configs) // 2),
    scale_to_gpu([160, 2812, 4.5], len(ours_gpu_configs) // 2),
    # scale_to_gpu([160, 3125, 5], len(ours_gpu_configs) // 2),
    scale_to_gpu([160, 3437, 5.5], len(ours_gpu_configs) // 2),
    # scale_to_gpu([160, 3755, 6], len(ours_gpu_configs) // 2),
    # scale_to_gpu([160, 3755, 6], len(ours_gpu_configs) // 2),
]

policies_to_test = [
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerWithoutMissRate, ours_gpu_configs, 'global_without_rebalancing'),
    (DataParallelRuntimeSelectionPolicy.ROUND_ROBIN, "", baseline_gpu_configs, 'baseline_with_lpm'),
    # (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.GlobalSchedulerTimeWithEviction, ours_gpu_configs, 'ours'),
    (DataParallelRuntimeSelectionPolicy.CUSTOM, CustomPolicyType.PROGRAMMING_ORACLE, baseline_gpu_configs, 'oracle'),
]

def gen_workloads_for_programming(configuration_to_test, policies_to_test):
    for configuration in configuration_to_test:
        num_prefix_patters, num_requests, request_rate = configuration
        dataloader, requests, send_out_times = create_programming_dataset_micro(
            configuration,
            model_name, 
            exp_time, 
        )
        for policy, custom_policy, server_configs, custom_policy_msg in policies_to_test: # assuming each policy has the exact same settings
            # print(server_configs)
            yield DefaultWorkload(
                    dataloader=dataloader,
                    policy=policy,
                    custom_policy=custom_policy,
                    custom_policy_msg = custom_policy_msg,
                    request_groups=[RequestGroup(requests=requests,
                                                 request_rate=request_rate,
                                                 send_out_times=send_out_times,
                                                 request_type=ExperimentType.default)],
                    # send_out_times=send_out_times,
                    num_prefix_patterns=num_prefix_patters,
                    random_ratio=0.0,
                    exp_time=exp_time,
                    request_rate=request_rate,
                    num_requests=num_requests,
                    server_configs=server_configs,
                )

workloads = gen_workloads_for_programming(configuration_to_test, policies_to_test)
loogle_experiment = ConfigurableMajorExperimentArgs(
    log_file_path="real_ckpt_all_in_one/2r_programming_H100/exp.log",
    csv_log_path="real_ckpt_all_in_one/2r_programming_H100/exp.csv",
    # log_file_path="logs/debug_loogle_cp_2048/exp.log",
    # csv_log_path="logs/debug_loogle_cp_2048/exp.csv",
    simulate=False,
    model_path=model_name,
    workload_configs=workloads,
    experiment_type=ExperimentType.default,
    experiment_name="programming"
)

exp_args = AllExperiments(
    [loogle_experiment]
)
