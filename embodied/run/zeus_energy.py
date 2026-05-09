import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager

try:
    from zeus.monitor import ZeusMonitor
except Exception:
    ZeusMonitor = None


def get_gpu_info():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,power.limit,memory.total",
            "--format=csv,noheader,nounits",
        ]).decode().strip()
        return out
    except Exception as e:
        return f"nvidia-smi unavailable: {e}"


def get_env_info():
    info = {
        "python": sys.version,
    }
    try:
        import jax
        info["jax"] = jax.__version__
        info["jax_devices"] = [str(x) for x in jax.devices()]
    except Exception as e:
        info["jax"] = f"unavailable: {e}"
    return info


def make_monitor(enabled=True, gpu_indices=(0,)):
    if not enabled or ZeusMonitor is None:
        return None
    return ZeusMonitor(gpu_indices=list(gpu_indices), approx_instant_energy=True)


def write_meta(logdir, model_name, env_name, task, script, args):
    path = os.path.join(str(logdir), "meta.json")
    record = {
        "model": model_name,
        "env": env_name,
        "task": task,
        "script": script,
        "steps": int(args.steps),
        "gpu": get_gpu_info(),
        "env_info": get_env_info(),
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def block_until_ready(x):
    """
    Important for JAX: timing is async unless we block.
    """
    try:
        import jax
        leaves = jax.tree_util.tree_leaves(x)
        for leaf in leaves:
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
    except Exception:
        pass


@contextmanager
def energy_window(monitor, label, logfile):
    if monitor is None:
        yield
        return

    monitor.begin_window(label)
    start_time = time.perf_counter()

    result = None
    try:
        yield
    finally:
        duration_sec = time.perf_counter() - start_time
        measurement = monitor.end_window(label)

        record = {
            "label": label,
            "joules": measurement.total_energy,
            "seconds": duration_sec,
        }

        with open(logfile, "a") as f:
            json.dump(record, f)
            f.write("\n")