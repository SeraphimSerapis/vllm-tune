#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# dist-tune.py — Distributed tuning orchestrator for vLLM-Tune
# ─────────────────────────────────────────────────────────────────────
#
# Parallelizes kernel tuning across multiple nodes in a cluster.
# Each node runs vllm-tune.sh for individual batch sizes or shapes,
# then results are rsync'd back and merged on the head node.
#
# Usage (via vllm-tune.sh):
#   vllm-tune.sh MODEL --dist --tp 2
#   vllm-tune.sh MODEL --dist --tp 2 --mode moe --foreground
#
# Direct usage:
#   ./dist-tune.py MODEL --tp 2 --nodes 10.0.0.1,10.0.0.2
#
# Node discovery (in priority order):
#   1. --nodes flag (comma-separated IPs)
#   2. CLUSTER_NODES environment variable
#   3. ~/spark-vllm-docker/.env file (CLUSTER_NODES= line)
#
# Requires: pip install rich
# ─────────────────────────────────────────────────────────────────────

import os
import re
import sys
import json
import queue
import shlex
import shutil
import signal
import subprocess
import threading
import time
import argparse
from collections import deque
from pathlib import Path

try:
    from rich.live import Live
    from rich.panel import Panel
    from rich.console import Console, Group
    from rich.text import Text
    from rich.columns import Columns
    from rich.ansi import AnsiDecoder
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
except ImportError:
    print("Error: 'rich' is required for the distributed tuning dashboard.",
          file=sys.stderr)
    print("Install: pip install rich", file=sys.stderr)
    sys.exit(1)


# ── Constants ───────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()

# Must match tune-moe.sh and tune-fp8.sh defaults
DEFAULT_BATCH_SIZES = [
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128,
    256, 512, 1024, 1536, 2048, 3072, 4096,
]

# ── Global state ────────────────────────────────────────────────────

shutdown_requested = False
ctrl_c_count = 0
active_procs = []
procs_lock = threading.Lock()
resolved_nodes = []

# UI state
node_states = {}
orchestrator_logs = deque(maxlen=10)
ui_lock = threading.Lock()
console = Console()
decoder = AnsiDecoder()


# ── Helpers ─────────────────────────────────────────────────────────

def slugify(model):
    """Generate a filesystem-safe slug from a model ID."""
    s = model.lower().replace('/', '--')
    return re.sub(r'[^a-z0-9._-]', '-', s)


def log_orchestrator(msg):
    """Append a message to the orchestrator log panel."""
    with ui_lock:
        orchestrator_logs.append(msg)


def deep_merge(base, overlay):
    """Recursively merge overlay into base dict (matches jq '.[0] * .[1]')."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# ── Node discovery ──────────────────────────────────────────────────

def discover_nodes(explicit_nodes=None):
    """Find cluster nodes. Returns (nodes_list, local_ip)."""
    local_ip = None

    # 1. Explicit --nodes flag
    if explicit_nodes:
        nodes = [n.strip() for n in explicit_nodes.split(',') if n.strip()]
        return nodes, _detect_local_ip()

    # 2. CLUSTER_NODES env var
    env_nodes = os.environ.get('CLUSTER_NODES', '')
    if env_nodes:
        nodes = [n.strip() for n in env_nodes.split(',') if n.strip()]
        return nodes, _detect_local_ip()

    # 3. Parse ~/spark-vllm-docker/.env
    env_file = Path.home() / 'spark-vllm-docker' / '.env'
    nodes = []
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith('CLUSTER_NODES='):
                    nodes = [n.strip() for n in line.split('=', 1)[1].split(',')
                             if n.strip()]
                elif line.startswith('LOCAL_IP='):
                    local_ip = line.split('=', 1)[1].strip()

    return nodes, local_ip or _detect_local_ip()


def _detect_local_ip():
    """Best-effort detection of this machine's IP."""
    try:
        result = subprocess.run(
            ['hostname', '-I'], capture_output=True, text=True, timeout=5)
        return result.stdout.strip().split()[0] if result.stdout.strip() else None
    except Exception:
        return None


# ── Model detection ─────────────────────────────────────────────────

def detect_metadata(model, tp, container):
    """Call lib/detect.py inside the container to get arch and shapes."""
    detect_script = SCRIPT_DIR / 'lib' / 'detect.py'
    if not detect_script.exists():
        log_orchestrator(f"⚠️  lib/detect.py not found at {detect_script}")
        return {'arch': 'unknown', 'shapes': []}

    with open(detect_script) as f:
        script_content = f.read()

    try:
        res = subprocess.check_output(
            ['docker', 'exec', container, 'python3', '-c', script_content,
             model, '--tp', str(tp), '--mode', 'all'],
            text=True, stderr=subprocess.DEVNULL, timeout=120)
        # Filter for the JSON line (vLLM/HF might print logs to stdout)
        for line in res.strip().split('\n'):
            if line.startswith('{'):
                return json.loads(line)
        return {'arch': 'unknown', 'shapes': []}
    except Exception as e:
        log_orchestrator(f"⚠️  Detection failed: {e}")
        return {'arch': 'unknown', 'shapes': []}


# ── Config scanning ─────────────────────────────────────────────────

def check_already_tuned(model, tp, mode, batch_sizes, shapes, configs_root):
    """Scan local configs to find which items are already tuned.

    Returns (pending_moe, pending_fp8) — lists of items still needing tuning.
    """
    slug = slugify(model)
    config_dir = configs_root / slug / f'tp{tp}'
    moe_dir = config_dir / 'moe'
    fp8_dir = config_dir / 'fp8'

    pending_moe = []
    pending_fp8 = []
    tuned_moe_count = 0
    tuned_fp8_count = 0

    # Check MoE: look for batch size keys inside config JSON files
    if mode in ('all', 'moe'):
        tuned_bs = set()
        if moe_dir.exists():
            for f in moe_dir.glob('*.json'):
                try:
                    data = json.loads(f.read_text())
                    tuned_bs.update(int(k) for k in data if k.isdigit())
                except Exception:
                    pass
        for bs in batch_sizes:
            if bs in tuned_bs:
                tuned_moe_count += 1
            else:
                pending_moe.append(bs)

    # Check FP8: look for shape files AND verify batch size coverage
    if mode in ('all', 'fp8'):
        for shape in shapes:
            n_val, k_val = shape.split(',')
            pattern = f'N={n_val},K={k_val},device_name=*,dtype=fp8_w8a8,block_shape=*.json'
            found_complete = False
            for cfg in fp8_dir.glob(pattern) if fp8_dir.exists() else []:
                try:
                    data = json.loads(cfg.read_text())
                    if all(str(bs) in data for bs in batch_sizes):
                        found_complete = True
                        break
                except Exception:
                    pass
            if found_complete:
                tuned_fp8_count += 1
            else:
                pending_fp8.append(shape)

    return tuned_moe_count, pending_moe, tuned_fp8_count, pending_fp8


# ── Config merging ──────────────────────────────────────────────────

def merge_remote_configs(tmp_dir, local_configs_dir):
    """Deep-merge retrieved configs from a remote node into local configs."""
    tmp_path = Path(tmp_dir)
    local_path = Path(local_configs_dir)

    for remote_file in tmp_path.rglob('*.json'):
        rel_path = remote_file.relative_to(tmp_path)
        local_file = local_path / rel_path

        if local_file.exists():
            try:
                remote_data = json.loads(remote_file.read_text())
                local_data = json.loads(local_file.read_text())
                deep_merge(local_data, remote_data)
                local_file.write_text(json.dumps(local_data, indent=4) + '\n')
            except Exception as e:
                log_orchestrator(f"⚠️  Merge error for {rel_path}: {e}")
        else:
            local_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_file, local_file)


# ── Signal handling ─────────────────────────────────────────────────

def _cleanup_remote(nodes, container):
    """Kill tuning processes on all nodes concurrently."""
    print("\n[Orchestrator] 🧹 Cleaning up remote tuning processes...")
    # Bracket regex trick ([t]une) prevents pkill from matching itself
    cleanup_cmd = (
        f"docker exec {shlex.quote(container)} "
        f"pkill -15 -f '[b]enchmark_moe.py' 2>/dev/null || true; "
        f"pkill -15 -f '[t]une-moe.sh|[t]une-fp8.sh' 2>/dev/null || true"
    )

    procs = []
    for node in nodes:
        if _is_local(node):
            p = subprocess.Popen(
                ['bash', '-c', cleanup_cmd],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:
            p = subprocess.Popen(
                ['ssh', '-n', '-o', 'StrictHostKeyChecking=no', node,
                 cleanup_cmd],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append((node, p))

    for node, p in procs:
        p.communicate(timeout=15)

    print("[Orchestrator] ✅ Cleanup complete.")


def _is_local(node, local_ip=None):
    """Check if a node address refers to this machine."""
    return node in ('127.0.0.1', 'localhost', local_ip)


def install_signal_handler(nodes, container):
    """Set up Ctrl-C handler with two-press escalation."""
    def handler(sig, frame):
        global shutdown_requested, ctrl_c_count
        ctrl_c_count += 1
        if ctrl_c_count == 1:
            shutdown_requested = True
            print("\n[Orchestrator] ⚠️  Ctrl-C detected. Shutting down...")
            with procs_lock:
                for p in active_procs:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    except Exception:
                        pass
            _cleanup_remote(nodes, container)
            sys.exit(1)
        else:
            print("\n[Orchestrator] 💀 Force exit.")
            os._exit(1)

    signal.signal(signal.SIGINT, handler)


# ── Dashboard ───────────────────────────────────────────────────────

def generate_dashboard(total_tasks, completed_tasks):
    """Build the Rich renderable for the live dashboard."""
    with ui_lock:
        node_panels = []
        for node, state in sorted(node_states.items()):
            logs = list(state['logs'])
            decoded_logs = []
            for line in logs:
                decoded_logs.extend(list(decoder.decode(line)))

            status_text = Text(state['status'], style="bold blue")
            if state['current_line']:
                progress_line = list(decoder.decode(state['current_line']))
                if progress_line:
                    status_text = progress_line[0]

            content = Group(
                Text(f"Task: {state['task']}", style="cyan"),
                Text("─" * 20, style="dim"),
                *decoded_logs[-3:],
                Text("─" * 20, style="dim"),
                status_text,
            )

            border = "blue"
            if "Completed" in state['status']:
                border = "green"
            elif "Failed" in state['status']:
                border = "red"
            elif "Syncing" in state['status']:
                border = "yellow"

            node_panels.append(Panel(
                content,
                title=f"[bold]{node}[/bold]",
                border_style=border,
                expand=True,
                padding=(0, 1),
            ))

        progress_bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        progress_bar.add_task(
            "Total Progress", total=total_tasks, completed=completed_tasks)

        orch_log_items = [
            Text.from_markup(l) if isinstance(l, str) else l
            for l in orchestrator_logs
        ]

        return Group(
            Panel(Text(
                f"Distributed vLLM-Tune — "
                f"{completed_tasks}/{total_tasks} Tasks Completed",
                justify="center", style="bold magenta")),
            Columns(node_panels, equal=True),
            Panel(Group(*orch_log_items),
                  title="[bold]Orchestrator Logs[/bold]",
                  border_style="dim", padding=(0, 1)),
            Panel(progress_bar, border_style="dim"),
        )


# ── Worker ──────────────────────────────────────────────────────────

def worker(node, local_ip, task_queue, args, stats, configs_root):
    """Per-node worker thread: pulls tasks from the queue and executes them."""
    is_remote = not _is_local(node, local_ip)
    vllm_tune = str(SCRIPT_DIR / 'vllm-tune.sh')

    with ui_lock:
        node_states[node] = {
            'task': 'Idle', 'status': 'Ready',
            'logs': deque(maxlen=20), 'current_line': '',
        }

    # Sync repo to remote node (skip configs/ and mod/ — we merge separately)
    if is_remote:
        log_orchestrator(f"🔄 Syncing repo to [bold]{node}[/bold]...")
        with ui_lock:
            node_states[node]['status'] = 'Syncing repository...'
        subprocess.run(
            ['rsync', '-az', '--exclude', 'configs/', '--exclude', 'mod/',
             '-e', 'ssh -o StrictHostKeyChecking=no',
             str(SCRIPT_DIR) + '/', f'{node}:{SCRIPT_DIR}/'],
            capture_output=True)
        log_orchestrator(f"✅ Sync complete for [bold]{node}[/bold]")

    while not shutdown_requested:
        try:
            task_type, model, tp, item = task_queue.get(timeout=1)
        except queue.Empty:
            break

        if shutdown_requested:
            task_queue.put((task_type, model, tp, item))
            break

        label = f"{task_type.upper()} {item}"
        log_orchestrator(f"🚀 [bold]{node}[/bold] started {label}")
        with ui_lock:
            node_states[node]['task'] = label
            node_states[node]['status'] = 'Active'

        # Build the vllm-tune.sh command for this single task
        cmd_parts = [
            vllm_tune, shlex.quote(model),
            '--tp', str(tp), '--mode', task_type,
            '--dtype', args.dtype, '-t', args.container, '--foreground',
        ]
        if task_type == 'moe':
            cmd_parts += ['--batch-size', str(item)]
        else:
            cmd_parts += ['--shapes', str(item)]

        cmd = ' '.join(cmd_parts)

        if is_remote:
            full_cmd = ['ssh', '-n', '-o', 'StrictHostKeyChecking=no', node, cmd]
            use_shell = False
        else:
            full_cmd = cmd
            use_shell = True

        proc = subprocess.Popen(
            full_cmd, shell=use_shell,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True, bufsize=0)

        with procs_lock:
            active_procs.append(proc)

        # Stream output character-by-character to handle \r progress bars
        buf = []
        while True:
            char = proc.stdout.read(1)
            if not char:
                break
            if char == '\n':
                line = ''.join(buf)
                with ui_lock:
                    node_states[node]['logs'].append(line)
                    node_states[node]['current_line'] = ''
                buf = []
            elif char == '\r':
                line = ''.join(buf)
                with ui_lock:
                    node_states[node]['current_line'] = line
                buf = []
            else:
                buf.append(char)

        proc.wait()
        with procs_lock:
            if proc in active_procs:
                active_procs.remove(proc)

        if shutdown_requested:
            break

        if proc.returncode == 0:
            log_orchestrator(f"✅ [bold]{node}[/bold] completed {label}")
            with ui_lock:
                node_states[node]['status'] = f'Completed {label}'
                stats['completed'] += 1
        else:
            log_orchestrator(f"❌ [bold]{node}[/bold] failed {label}")
            with ui_lock:
                node_states[node]['status'] = f'Failed {label}'

        # Retrieve configs from remote node
        if is_remote and not shutdown_requested:
            tmp_dir = configs_root / f'.tmp_{node.replace(".", "_")}'
            tmp_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ['rsync', '-az',
                 '-e', 'ssh -o StrictHostKeyChecking=no',
                 f'{node}:{SCRIPT_DIR}/configs/', str(tmp_dir) + '/'],
                capture_output=True)
            merge_remote_configs(str(tmp_dir), str(configs_root))
            shutil.rmtree(tmp_dir, ignore_errors=True)

        task_queue.task_done()

    with ui_lock:
        if node_states[node]['status'] == 'Active':
            node_states[node]['status'] = 'Stopped'


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='vLLM-Tune distributed tuning orchestrator')
    parser.add_argument('model', help='HuggingFace model ID')
    parser.add_argument('--tp', default='2', help='Tensor parallelism (default: 2)')
    parser.add_argument('--mode', default='all', choices=['all', 'moe', 'fp8'],
                        help='Tuning mode (default: all)')
    parser.add_argument('--dtype', default='fp8_w8a8', help='MoE dtype')
    parser.add_argument('-t', '--container', default='vllm_node',
                        help='Container name (default: vllm_node)')
    parser.add_argument('--nodes', default=None,
                        help='Comma-separated node IPs (auto-detected if omitted)')
    parser.add_argument('--batch-size', nargs='+', type=int, default=None,
                        help='Custom batch sizes')
    parser.add_argument('--shapes', nargs='+', default=None,
                        help='Explicit FP8 shapes (N,K)')
    args = parser.parse_args()

    configs_root = SCRIPT_DIR / 'configs'

    # Discover nodes
    nodes, local_ip = discover_nodes(args.nodes)
    if not nodes:
        print("Error: No cluster nodes found.", file=sys.stderr)
        print("  Provide --nodes, set CLUSTER_NODES, or configure "
              "~/spark-vllm-docker/.env", file=sys.stderr)
        sys.exit(1)

    global resolved_nodes
    resolved_nodes = nodes

    install_signal_handler(nodes, args.container)

    print("=" * 70)
    print("  Distributed vLLM-Tune Orchestrator")
    print(f"  Nodes ({len(nodes)}): {', '.join(nodes)}")
    print(f"  Model: {args.model}")
    print(f"  TP: {args.tp}  Mode: {args.mode}  Container: {args.container}")
    print("=" * 70)

    # Detect model architecture and shapes
    print("\nDetecting model architecture and shapes...")
    metadata = detect_metadata(args.model, args.tp, args.container)
    is_moe = metadata.get('arch') == 'moe'
    detected_shapes = args.shapes or metadata.get('shapes', [])

    if is_moe:
        print("  ✅ Architecture: MoE")
    else:
        print("  ⏭  Architecture: Dense (skipping MoE tuning)")

    if detected_shapes:
        print(f"  ✅ Detected {len(detected_shapes)} FP8 shapes.")
    else:
        print("  ⚠  Warning: Could not detect FP8 shapes.")

    if args.mode == 'moe' and not is_moe:
        print(f"\n  Error: {args.model} is a dense model. "
              "MoE tuning is not applicable.", file=sys.stderr)
        sys.exit(1)

    # Determine batch sizes
    batch_sizes = args.batch_size or DEFAULT_BATCH_SIZES

    # Scan existing configs for resume
    print("\nScanning local configs to resume progress...")
    tuned_moe, pending_moe, tuned_fp8, pending_fp8 = check_already_tuned(
        args.model, args.tp, args.mode, batch_sizes, detected_shapes,
        configs_root)

    # Build task queue
    task_queue = queue.Queue()

    if args.mode in ('all', 'moe') and is_moe:
        print(f"  MoE: {tuned_moe} already tuned, {len(pending_moe)} pending")
        for bs in pending_moe:
            task_queue.put(('moe', args.model, args.tp, bs))

    if args.mode in ('all', 'fp8') and detected_shapes:
        print(f"  FP8: {tuned_fp8} already tuned, {len(pending_fp8)} pending")
        for shape in pending_fp8:
            task_queue.put(('fp8', args.model, args.tp, shape))

    total_tasks = task_queue.qsize()
    if total_tasks == 0:
        print("\n🎉 All tasks already completed! Nothing to tune.")
        print(f"\n  Sync to mod:  ./vllm-tune.sh {args.model} "
              f"--tp {args.tp} --sync-mod")
        sys.exit(0)

    print(f"\nEnqueued {total_tasks} tuning tasks across {len(nodes)} node(s).")
    time.sleep(2)

    # Launch workers
    stats = {'completed': 0}
    threads = []

    with Live(generate_dashboard(total_tasks, 0),
              refresh_per_second=4, console=console) as live:
        log_orchestrator(
            f"🚀 Started distributed tuning for "
            f"[bold cyan]{args.model}[/bold cyan]")

        for node in nodes:
            t = threading.Thread(
                target=worker,
                args=(node, local_ip, task_queue, args, stats, configs_root))
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
    else:
        slug = slugify(args.model)
        print("\n" + "=" * 70)
        print("  🎉 All distributed tuning tasks completed!")
        print("=" * 70)
        print(f"\n  Configs: {configs_root / slug / f'tp{args.tp}'}/")
        print(f"\n  Next step — sync to mod directory:")
        console.print(
            f"    ./vllm-tune.sh {args.model} --tp {args.tp} --sync-mod")
        print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
