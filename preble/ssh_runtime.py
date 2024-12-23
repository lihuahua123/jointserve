import paramiko
import re, time
import requests
import threading
import logging
import select

logger = logging.getLogger('SSHRuntimeLogger')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def stream_logger(node_name, stream):
    """
    Reads from a stream line by line and logs each line.
    
    :param name: A name to identify the stream in the logs.
    :param stream: The stream to read from.
    """
    try:
        while True:
            line = stream.readline().strip()
            if not line:
                break
            pattern = r"(GPU:\s+(\d+))"
            if node_name:
                line = re.sub(pattern, rf"GPU: {node_name}_\2", line)
                line = line.replace(f"INFO:model_rpc:", "")
            logger.info(f"{line.strip()}")
    finally:
        stream.close()

class SSHRuntimeManager:
    def __init__(self, model_path, ssh_config, gpu, **kwargs):
        self.model_path = model_path
        self.ssh_config = ssh_config
        self.gpu = gpu
        assert self.ssh_config
        self.ssh_client = self.initialize_ssh_client()
        self.node_name = self.ssh_config.get("node_name")
        self.port = self.start_remote_runtime(**kwargs)
        # Initialize server with running these configs
        # Save url to url
        super().__init__()

    def initialize_ssh_client(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=self.ssh_config["hostname"],
            username=self.ssh_config["username"],
            port=self.ssh_config.get("port", 456),
            key_filename=self.ssh_config.get("key_filename"),
            password=self.ssh_config.get("password"),
        )
        return ssh

    def start_remote_runtime(self, **kwargs):
        cli_args = self.kwargs_to_cli_args(**kwargs)
        environment_variables = {
            'CUDA_VISIBLE_DEVICES': str(self.gpu),
            'LOGLEVEL': 'DEBUG'
        }
        python_process = self.ssh_config.get("python_process", "/mnt/ssd1/vikranth/sglang_experiments/sglang_env/bin/python")
        command = f'setsid env CUDA_VISIBLE_DEVICES={self.gpu} {python_process} -m sglang.launch_server --model-path {self.model_path} {cli_args} --host 0.0.0.0'
        logger.info(f"Running command {command} on gpu {self.gpu}")
        transport = self.ssh_client.get_transport()
        channel = transport.open_session(window_size=paramiko.common.MAX_WINDOW_SIZE)
        self.channel = channel
        self.transport = transport
        self.transport.set_keepalive(0) # Send keepalive packets every 30 seconds

        channel.update_environment(environment_variables)
        channel.exec_command(command)
        stdout = channel.makefile('r', -1)
        stderr = channel.makefile_stderr('r', -1)
        timeout = 120
        start_time = time.time()
        port = None
        while time.time()  - start_time < timeout:
            ready_channels, _, _ = select.select([channel], [], [], timeout)
            if channel in ready_channels:
                # Here, you need to check if there's data on stdout or stderr
                while channel.recv_ready():
                    line = stdout.readline()
                    print(line, end='')
                while channel.recv_stderr_ready():
                    line = stderr.readline()
                    print(line, end='')
            else:
                print("No data received")
                time.sleep(1)
                continue
            # Search for the port number in the line
            match = re.search(r"Server is on port (\d+) on host (.*) on pid (\d+)", line)
            if match:
                port = match.group(1)  # Capture the port number
                pid = match.group(3)
                break

            if not line:
                time.sleep(1)  # Wait before trying to read more output

        if not port:
            raise Exception("Failed to detect server startup within the timeout period.")
        self.url = f"http://{self.ssh_config['hostname']}:{port}"
        self.generate_url = f"{self.url}/generate"
        self.hit_ratio_url = f"{self.url}/windowed_prefix_hit_ratio"

        # Wait for the /model_info to return valid json for 5 attempts
        for _ in range(5):
            try:
                response = requests.get(f"{self.url}/get_model_info")
                if response.status_code == 200:
                    break
                time.sleep(2)
            except Exception as e:
                print(e)
            time.sleep(1)
        self.process_pid = pid
        self.port = port

        stderr_thread = threading.Thread(target=stream_logger, args=(self.node_name, stderr), daemon=True)
        stderr_thread.start()
        return port
    
    def shutdown(self):
        if self.ssh_client:
            self.ssh_client.exec_command(f"pkill -KILL -f {self.port}")
            stdin, stdout, stderr = self.ssh_client.exec_command("pkill -KILL -f sglang")
            self.channel.close()
            self.ssh_client.close()


    def kwargs_to_cli_args(self, **kwargs):
        args = []
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key.replace('_', '-')}")
            else:
                args.append(f"--{key.replace('_', '-')} {value}")
        return ' '.join(args)

if __name__ == "__main__":
    ssh_config = {
        "hostname": "192.168.1.16",
        "username": "wuklab",
        "port": 456,
        "python_process": "/mnt/data/ssd/sglang_env/bin/python",
        "node_name": "06",
    }
    ours_server_args = {
        'log_prefix_hit': True,
        'mem_fraction_static': 0.8,
        'context_length': 32768,
        "enable_flashinfer": True,
        'schedule_heuristic': 'fcfs-mpq',
        "chunk_prefill_budget": 512,
        'report_hit_ratio': True ,
        'enable_iterative_eviction': False,
    }
    runtime = SSHRuntimeManager("/hy-tmp/", ssh_config, 0, cuda_devices=0, **ours_server_args)
    print(f"Running on port {runtime.port}")
    time.sleep(30)
    runtime.shutdown()
    