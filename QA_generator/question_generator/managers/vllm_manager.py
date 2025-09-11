import subprocess
import json
import time

from .log_manager import LogManager


def check_vLLM(port: int = 8096) -> bool:
    """
    Check if vLLM server is running on the given port.
    """
    logger = LogManager().get_logger(__name__)

    try:
        response = subprocess.run(
            ["curl", "-s", f"http://localhost:{str(port)}/v1/models"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )

        # if response.returncode != 0:
        #     # logger.error(f"Curl command failed with return code: {response.returncode}")
        #     return False

        try:
            data = json.loads(response.stdout)
            logger.success("Successfully got API response from vLLM")
            return True
        except json.JSONDecodeError:
            logger.error("No valid JSON response from vLLM API, vLLM may be stopped.")
            logger.debug(f"Response stdout: {response.stdout}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Timeout while checking vLLM API")
        return False
    except Exception as e:
        logger.error(f"Error contacting vLLM API: {e}")
        return False


def start_vllm_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8096,
    gpu_memory_utilization: float = 0.3,
    max_wait_time: int = 120,
    wait_interval: int = 5,
) -> bool:
    """
    Start the vLLM server if not already running.
    """
    logger = LogManager().get_logger(__name__)

    if check_vLLM(port=port):
        logger.info("vLLM server is already running")
        return True

    logger.info("Starting vLLM server............")
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        elapsed_time = 0
        while elapsed_time < max_wait_time:
            time.sleep(wait_interval)
            elapsed_time += wait_interval

            if check_vLLM(port=port):
                logger.success("vLLM server started successfully")
                return True

            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(
                    f"vLLM server failed to start. Exit code: {process.returncode}"
                )
                logger.error(f"Stderr: {stderr.decode()}")
                return False

        logger.error("Timeout waiting for vLLM server to start")
        process.terminate()
        return False

    except Exception as e:
        logger.error(f"Error starting vLLM server: {e}")
        return False
