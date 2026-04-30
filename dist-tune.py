#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import queue
import threading
import shlex
import time
import signal
import json
from pathlib import Path
from collections import deque

from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console, Group
from rich.text import Text
from rich.columns import Columns
from rich.ansi import AnsiDecoder
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

# Global state for graceful shutdown
shutdown_requested = False
ctrl_c_count = 0
active_procs = []
procs_lock = threading.Lock()
nodes_list = [] # Store nodes for cleanup

# UI State
node_states = {}
orchestrator_logs = deque(maxlen=10)
ui_lock = threading.Lock()
console = Console()
decoder = AnsiDecoder()

def log_orchestrator(msg):
    with ui_lock:
        orchestrator_logs.append(msg)

def remote_cleanup(nodes):
    """Surgically kill tuning processes on all nodes."""
    log_orchestrator("[bold red]🧹 Surgically cleaning up remote tuning processes...[/bold red]")
    # Kill the tuning scripts and the specific benchmark worker
    cleanup_cmd = (
        "pkill -9 -f 'vllm-tune.sh|tune-moe.sh|tune-fp8.sh' || true; "
        "docker exec vllm_node pkill -9 -f 'ray::BenchmarkWorker.tune|benchmark_moe.py' || true"
    )
    for node in nodes:
        is_remote = node not in ["127.0.0.1", "localhost"]
        # Also check if it's the local IP
        env_nodes, local_ip = get_nodes()
        if node == local_ip: is_remote = False
        
        if is_remote:
            subprocess.run(f"ssh -n -o StrictHostKeyChecking=no {node} {shlex.quote(cleanup_cmd)}", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            subprocess.run(cleanup_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def signal_handler(sig, frame):
    global shutdown_requested, ctrl_c_count
    ctrl_c_count += 1
    if ctrl_c_count == 1:
        shutdown_requested = True
        log_orchestrator("[bold yellow]⚠️  Ctrl-C detected. Waiting for current tasks to finish...[/bold yellow]")
        log_orchestrator("[bold yellow]⚠️  Press Ctrl-C again to FORCE KILL everything.[/bold yellow]")
    else:
        # We can't easily use log_orchestrator here if we are about to exit, 
        # but let's try to print at least.
        print("\n[Orchestrator] 💀 Forcefully terminating all tuning processes...")
        with procs_lock:
            for p in active_procs:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
        remote_cleanup(nodes_list)
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

def get_nodes():
    env_file = os.path.expanduser("~/spark-vllm-docker/.env")
    nodes = []
    local_ip = None
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith("CLUSTER_NODES="):
                    nodes = line.strip().split("=")[1].split(",")
                elif line.startswith("LOCAL_IP="):
                    local_ip = line.strip().split("=")[1]
    return nodes, local_ip

def generate_dashboard(total_tasks, completed_tasks):
    with ui_lock:
        node_panels = []
        for node, state in sorted(node_states.items()):
            logs = list(state['logs'])
            # Decode ANSI and handle progress bar (\r)
            decoded_logs = []
            for line in logs:
                decoded_logs.extend(list(decoder.decode(line)))
            
            # Current status/progress line
            status_text = Text(state['status'], style="bold blue")
            if state['current_line']:
                # The current line might be a progress bar from tqdm
                progress_line = list(decoder.decode(state['current_line']))
                if progress_line:
                    status_text = progress_line[0]

            content = Group(
                Text(f"Task: {state['task']}", style="cyan"),
                Text("-" * 20, style="dim"),
                *decoded_logs[-3:], # Show last 3 lines of history
                Text("-" * 20, style="dim"),
                status_text
            )
            
            border_style = "blue"
            if "Completed" in state['status']: border_style = "green"
            elif "Failed" in state['status']: border_style = "red"
            elif "Syncing" in state['status']: border_style = "yellow"

            node_panels.append(
                Panel(
                    content,
                    title=f"[bold]{node}[/bold]",
                    border_style=border_style,
                    expand=True,
                    padding=(0, 1)
                )
            )

        # Global progress
        progress_bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        task_id = progress_bar.add_task("Total Progress", total=total_tasks, completed=completed_tasks)
        
        # Orchestrator logs
        orch_logs_group = Group(*[Text.from_markup(l) if isinstance(l, str) else l for l in orchestrator_logs])

        return Group(
            Panel(Text(f"Distributed Tuning Orchestrator - {completed_tasks}/{total_tasks} Tasks Completed", justify="center", style="bold magenta")),
            Columns(node_panels, equal=True),
            Panel(orch_logs_group, title="[bold]Orchestrator Logs[/bold]", border_style="dim", padding=(0, 1)),
            Panel(progress_bar, border_style="dim")
        )

def detect_metadata(model, tp):
    """Call lib/detect.py inside the container to get arch and shapes."""
    detect_script_path = Path(__file__).parent / "lib" / "detect.py"
    with open(detect_script_path) as f:
        script_content = f.read()
    
    try:
        res = subprocess.check_output(
            ["docker", "exec", "vllm_node", "python3", "-c", script_content, model, "--tp", str(tp), "--mode", "all"],
            text=True, stderr=subprocess.DEVNULL
        )
        # Filter for the JSON line (vLLM/HF might print logs to stdout)
        for line in res.strip().split("\n"):
            if line.startswith("{"):
                return json.loads(line)
        return {"arch": "unknown", "shapes": []}
    except Exception:
        return {"arch": "unknown", "shapes": []}

def slugify(model):
    import re
    s = model.lower().replace('/', '--')
    return re.sub(r'[^a-z0-9._-]', '-', s)

def check_already_tuned(model, tp, mode, batch_sizes, shapes):
    slug = slugify(model)
    config_dir = Path(f"/home/llm/vllm-tune/configs/{slug}/tp{tp}")
    moe_dir = config_dir / "moe"
    fp8_dir = config_dir / "fp8"

    tuned_moe = set()
    pending_moe = []
    tuned_fp8 = set()
    pending_fp8 = []

    # Check MoE configs
    if mode in ["all", "moe"]:
        if moe_dir.exists():
            for f in moe_dir.glob("*.json"):
                try:
                    with open(f) as jf:
                        data = json.load(jf)
                        for key in data.keys():
                            if key.isdigit():
                                tuned_moe.add(int(key))
                except Exception:
                    pass
        for bs in batch_sizes:
            if bs in tuned_moe:
                pass
            else:
                pending_moe.append(bs)

    # Check FP8 configs
    if mode in ["all", "fp8"]:
        if fp8_dir.exists():
            for f in fp8_dir.glob("*.json"):
                # filename format: N=...,K=...,...
                name = f.name
                if name.startswith("N="):
                    parts = name.split(",")
                    n_val = parts[0].split("=")[1]
                    k_val = parts[1].split("=")[1]
                    tuned_fp8.add(f"{n_val},{k_val}")
        for shape in shapes:
            if shape in tuned_fp8:
                pass
            else:
                pending_fp8.append(shape)

    return tuned_moe, pending_moe, tuned_fp8, pending_fp8


def worker(node, local_ip, task_queue, extra_args, stats):
    is_remote = node not in [local_ip, "127.0.0.1", "localhost"]
    
    with ui_lock:
        node_states[node] = {
            'task': 'Idle',
            'status': 'Ready',
            'logs': deque(maxlen=20),
            'current_line': ''
        }

    if is_remote:
        log_orchestrator(f"🔄 Syncing repo to [bold]{node}[/bold]...")
        with ui_lock:
            node_states[node]['status'] = 'Syncing repository...'
        subprocess.run(
            f"rsync -avz --exclude 'configs/' --exclude 'mod/' -e 'ssh -o StrictHostKeyChecking=no' /home/llm/vllm-tune/ /home/llm/vllm-tune/dist-tune.py {node}:/home/llm/vllm-tune/ >/dev/null 2>&1", 
            shell=True
        )
        log_orchestrator(f"✅ Sync complete for [bold]{node}[/bold]")

    while not shutdown_requested:
        try:
            task_type, model, tp, arg = task_queue.get(timeout=1)
        except queue.Empty:
            break

        if shutdown_requested:
            task_queue.put((task_type, model, tp, arg)) # Put it back
            break

        log_orchestrator(f"🚀 [bold]{node}[/bold] started {task_type.upper()} {arg}")
        with ui_lock:
            node_states[node]['task'] = f"{task_type.upper()} {arg}"
            node_states[node]['status'] = 'Active'

        extra_args_str = " ".join([shlex.quote(a) for a in extra_args])
        
        if task_type == "moe":
            cmd = f"/home/llm/vllm-tune/vllm-tune.sh {shlex.quote(model)} --tp {tp} --mode moe --batch-size {arg} --foreground {extra_args_str}"
        else:
            cmd = f"/home/llm/vllm-tune/vllm-tune.sh {shlex.quote(model)} --tp {tp} --mode fp8 --shapes {arg} --foreground {extra_args_str}"
        
        if is_remote:
            full_cmd = f"ssh -n -o StrictHostKeyChecking=no {node} {shlex.quote(cmd)}"
        else:
            full_cmd = cmd
            
        proc = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True, bufsize=0)
        with procs_lock:
            active_procs.append(proc)
            
        current_buffer = []
        while True:
            char = proc.stdout.read(1)
            if not char:
                break
            
            if char == '\n':
                line = "".join(current_buffer)
                with ui_lock:
                    node_states[node]['logs'].append(line)
                    node_states[node]['current_line'] = ''
                current_buffer = []
            elif char == '\r':
                line = "".join(current_buffer)
                with ui_lock:
                    node_states[node]['current_line'] = line
                current_buffer = []
            else:
                current_buffer.append(char)
                
        proc.wait()
        
        with procs_lock:
            if proc in active_procs:
                active_procs.remove(proc)
                
        if shutdown_requested and ctrl_c_count > 1:
            break # Force killed
        
        if proc.returncode == 0:
            log_orchestrator(f"✅ [bold]{node}[/bold] completed {task_type.upper()} {arg}")
            with ui_lock:
                node_states[node]['status'] = f'Completed {task_type.upper()} {arg}'
            stats['completed'] += 1
        else:
            log_orchestrator(f"❌ [bold]{node}[/bold] failed {task_type.upper()} {arg}")
            with ui_lock:
                node_states[node]['status'] = f'Failed {task_type.upper()} {arg}'

        if is_remote and not shutdown_requested:
            os.makedirs("/home/llm/vllm-tune/configs/", exist_ok=True)
            subprocess.run(
                f"rsync -avz -e 'ssh -o StrictHostKeyChecking=no' {node}:/home/llm/vllm-tune/configs/ /home/llm/vllm-tune/configs/ >/dev/null 2>&1", 
                shell=True
            )

        task_queue.task_done()
    
    with ui_lock:
        if node_states[node]['status'] == 'Active':
            node_states[node]['status'] = 'Stopped'

def main():
    global nodes_list
    if "--tmux" in sys.argv:
        if "TMUX" not in os.environ and os.environ.get("_VLLM_TUNE_INSIDE_TMUX") != "1":
            print("  Starting distributed tuning in tmux session: dist-tune")
            args_without_tmux = [arg for arg in sys.argv if arg != "--tmux"]
            cmd = " ".join([shlex.quote(arg) for arg in args_without_tmux])
            os.environ["_VLLM_TUNE_INSIDE_TMUX"] = "1"
            subprocess.run(f"tmux new-session -d -s dist-tune \"{cmd}; echo ''; echo 'Tuning complete. Press Enter to close.'; read\"", shell=True)
            print("  Attach:  tmux attach -t dist-tune")
            print("  Detach:  Ctrl-b d")
            sys.exit(0)
            
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tp", default="2")
    parser.add_argument("--mode", default="all")
    parser.add_argument("--tmux", action="store_true")
    args, unknown = parser.parse_known_args()

    nodes, local_ip = get_nodes()
    if not nodes:
        print("No nodes found in .env")
        sys.exit(1)
    nodes_list = nodes

    print("======================================================================")
    print("  Distributed vLLM-Tune Orchestrator (Dynamic Worker Pool)")
    print(f"  Nodes ({len(nodes)}): {', '.join(nodes)}")
    print(f"  Model: {args.model}")
    print("======================================================================")

    print("\nDetecting model architecture and shapes...")
    metadata = detect_metadata(args.model, args.tp)
    is_moe = metadata.get("arch") == "moe"
    all_shapes = metadata.get("shapes", [])

    if is_moe:
        print(f"  ✅ Architecture: MoE")
    else:
        print(f"  ⏭  Architecture: Dense (skipping MoE tuning)")
        
    if all_shapes:
        print(f"  ✅ Detected {len(all_shapes)} FP8 shapes.")
    else:
        print("  ⚠ Warning: Could not detect FP8 shapes.")
            
    print("\nScanning local configs to resume progress...")
    all_batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
    tuned_moe, pending_moe, tuned_fp8, pending_fp8 = check_already_tuned(args.model, args.tp, args.mode, all_batch_sizes, all_shapes)
    
    task_queue = queue.Queue()
    if args.mode in ["all", "moe"] and is_moe:
        print(f"MoE Batch Sizes: {len(tuned_moe)} already tuned, {len(pending_moe)} pending.")
        for bs in pending_moe:
            task_queue.put(("moe", args.model, args.tp, bs))
    elif args.mode == "moe" and not is_moe:
        print(f"Error: {args.model} is a dense model. MoE tuning is not applicable.")
        sys.exit(1)
            
    if args.mode in ["all", "fp8"] and all_shapes:
        print(f"FP8 Shapes: {len(tuned_fp8)} already tuned, {len(pending_fp8)} pending.")
        for shape in pending_fp8:
            task_queue.put(("fp8", args.model, args.tp, shape))
            
    total_tasks = task_queue.qsize()
    if total_tasks == 0:
        print("\n🎉 All tasks are already completed! Nothing to tune.")
        print(f"You can now run: /home/llm/vllm-tune/vllm-tune.sh {args.model} --tp {args.tp} --sync-mod")
        sys.exit(0)
        
    print(f"\nEnqueued {total_tasks} tuning tasks. Dispatching to workers...\n")
    time.sleep(2) # Give user a moment to read the summary

    stats = {'completed': 0}
    threads = []
    
    with Live(generate_dashboard(total_tasks, 0), refresh_per_second=4, console=console) as live:
        log_orchestrator(f"🚀 Started distributed tuning for [bold cyan]{args.model}[/bold cyan]")
        for node in nodes:
            t = threading.Thread(target=worker, args=(node, local_ip, task_queue, unknown, stats))
            t.start()
            threads.append(t)
            
        try:
            while any(t.is_alive() for t in threads):
                live.update(generate_dashboard(total_tasks, stats['completed']))
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            for t in threads:
                t.join(timeout=0.1)
            
    if shutdown_requested:
        print("\n[Orchestrator] Exited early due to Ctrl-C.")
        remote_cleanup(nodes)
    else:
        print("\n" + "="*70)
        print("  🎉 All distributed tuning tasks completed!")
        print("="*70)
        print(f"\n  Configs have been rsync'd back to: {os.path.abspath('/home/llm/vllm-tune/configs/')}")
        print("\n  Next step: Sync these to your project's mod directory for persistence:")
        print(f"  [bold cyan]  ./vllm-tune.sh {args.model} --tp {args.tp} --sync-mod[/bold cyan]")
        print("="*70 + "\n")

if __name__ == '__main__':
    main()
