#!/usr/bin/env bash
# vLLM-Tune Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/SeraphimSerapis/vllm-tune/main/install.sh | bash
#
# What it does:
#   1. Clones (or updates) the vllm-tune repo
#   2. Detects spark-vllm-docker installation
#   3. Symlinks the mod into spark-vllm-docker (optional)
#   4. Adds vllm-tune to PATH (optional)
#   5. Verifies the installation
#
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────

REPO_URL="${VLLM_TUNE_REPO:-https://github.com/SeraphimSerapis/vllm-tune.git}"
INSTALL_DIR="${VLLM_TUNE_DIR:-$HOME/vllm-tune}"
SHELL_RC=""

# ── Colors ──────────────────────────────────────────────────────────

bold()  { printf "\033[1m%s\033[0m"   "$*"; }
cyan()  { printf "\033[1;36m%s\033[0m" "$*"; }
green() { printf "\033[1;32m%s\033[0m" "$*"; }
yellow(){ printf "\033[1;33m%s\033[0m" "$*"; }
red()   { printf "\033[1;31m%s\033[0m" "$*"; }

info()  { printf "  %s %s\n" "$(cyan "→")" "$*"; }
ok()    { printf "  %s %s\n" "$(green "✓")" "$*"; }
warn()  { printf "  %s %s\n" "$(yellow "⚠")" "$*"; }
fail()  { printf "  %s %s\n" "$(red "✗")" "$*"; }

# ── Helpers ─────────────────────────────────────────────────────────

ask() {
    local prompt="$1" default="${2:-}"
    if [[ -n "$default" ]]; then
        printf "  %s [%s]: " "$prompt" "$default"
    else
        printf "  %s: " "$prompt"
    fi
    read -r answer </dev/tty
    echo "${answer:-$default}"
}

ask_yn() {
    local prompt="$1" default="${2:-Y}"
    local hint="Y/n"
    [[ "$default" =~ ^[Nn] ]] && hint="y/N"
    printf "  %s [%s] " "$prompt" "$hint"
    read -r answer </dev/tty
    answer="${answer:-$default}"
    [[ "$answer" =~ ^[Yy] ]]
}

detect_shell_rc() {
    local shell_name
    shell_name="$(basename "${SHELL:-/bin/bash}")"
    case "$shell_name" in
        zsh)  SHELL_RC="$HOME/.zshrc" ;;
        bash)
            if [[ -f "$HOME/.bash_aliases" ]]; then
                SHELL_RC="$HOME/.bash_aliases"
            elif [[ -f "$HOME/.bashrc" ]]; then
                SHELL_RC="$HOME/.bashrc"
            fi ;;
        *)    SHELL_RC="$HOME/.profile" ;;
    esac
}

find_spark_vllm() {
    for candidate in \
        "$HOME/spark-vllm-docker" \
        "$HOME/code/spark-vllm-docker" \
        "/opt/spark-vllm-docker"; do
        [[ -d "$candidate/mods" ]] && echo "$candidate" && return
    done
}

# ── Banner ──────────────────────────────────────────────────────────

echo
printf "\033[1;36m"
cat << 'EOF'
██    ██ ██      ██      ███    ███       ████████ ██    ██ ███    ██ ███████
██    ██ ██      ██      ████  ████          ██    ██    ██ ████   ██ ██
██    ██ ██      ██      ██ ████ ██ ███████  ██    ██    ██ ██ ██  ██ █████
 ██  ██  ██      ██      ██  ██  ██          ██    ██    ██ ██  ██ ██ ██
  ████   ███████ ███████ ██      ██          ██     ██████  ██   ████ ███████
EOF
printf "\033[0m"
echo
echo "  Installer — Triton kernel tuning for vLLM on NVIDIA GPUs"
echo

# ── Prerequisites ───────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Checking prerequisites..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

missing=()
# docker is not apt-installable in the usual sense; flag separately
missing_noinstall=()
for cmd in git jq docker rsync; do
    if command -v "$cmd" &>/dev/null; then
        ok "$cmd"
    else
        fail "$cmd — not found"
        if [[ "$cmd" == "docker" ]]; then
            missing_noinstall+=("$cmd")
        else
            missing+=("$cmd")
        fi
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo
    if ask_yn "Install missing packages (${missing[*]}) via apt?"; then
        info "Running: sudo apt install -y ${missing[*]}"
        if sudo apt install -y "${missing[@]}"; then
            ok "Installed ${missing[*]}"
            missing=()
        else
            fail "apt install failed."
        fi
    fi
fi

if [[ ${#missing[@]} -gt 0 || ${#missing_noinstall[@]} -gt 0 ]]; then
    echo
    [[ ${#missing[@]} -gt 0 ]] && warn "Still missing: ${missing[*]}"
    [[ ${#missing_noinstall[@]} -gt 0 ]] && warn "Missing (install manually): ${missing_noinstall[*]}"
    warn "Install them and re-run this script."
    echo
    exit 1
fi
echo

# ── Step 1: Clone / update ──────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 1: Install vLLM-Tune"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

INSTALL_DIR="$(ask "Install location" "$INSTALL_DIR")"

if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Updating existing installation..."
    git -C "$INSTALL_DIR" pull --ff-only 2>/dev/null && ok "Updated" || warn "Pull failed (you may have local changes)"
elif [[ -d "$INSTALL_DIR" ]]; then
    warn "$INSTALL_DIR exists but is not a git repo."
    if ask_yn "Use it anyway?"; then
        ok "Using existing directory"
    else
        fail "Aborting."
        exit 1
    fi
else
    info "Cloning $REPO_URL..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    ok "Cloned to $INSTALL_DIR"
fi
echo

# Source common.sh for config helpers (now that INSTALL_DIR is resolved)
SCRIPT_DIR="$INSTALL_DIR"
source "$INSTALL_DIR/lib/common.sh"

# ── Step 2: spark-vllm-docker integration ───────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 2: spark-vllm-docker integration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SPARK_DIR="$(find_spark_vllm)"

if [[ -n "$SPARK_DIR" ]]; then
    ok "Found spark-vllm-docker at $SPARK_DIR"
    SYMLINK_TARGET="$SPARK_DIR/mods/vllm-tune"

    if [[ -L "$SYMLINK_TARGET" ]]; then
        ok "Symlink already exists: $SYMLINK_TARGET"
    elif [[ -d "$SYMLINK_TARGET" ]]; then
        warn "$SYMLINK_TARGET already exists (not a symlink)."
        warn "Skipping — remove it manually if you want to use the symlink approach."
    elif ask_yn "Create symlink $SYMLINK_TARGET → $INSTALL_DIR/mod?"; then
        ln -s "$INSTALL_DIR/mod" "$SYMLINK_TARGET"
        ok "Created symlink"
        echo "  Now you can use: ./launch-cluster.sh --apply-mod mods/vllm-tune ..."
    else
        info "Skipped. You can use --apply-mod with an absolute path instead:"
        echo "    --apply-mod $INSTALL_DIR/mod"
    fi
else
    info "spark-vllm-docker not found (checked ~/spark-vllm-docker, ~/code/spark-vllm-docker)"
    info "No action needed — you can use --deploy / --deploy-only with docker cp."
fi

# ── Save config.json ────────────────────────────────────────────────

info "Writing config.json..."
cfg_set "spark_vllm_docker" "${SPARK_DIR:-}"
cfg_set "configs_dir" "$INSTALL_DIR/configs"
cfg_set "container" "vllm_node"
cfg_set_num "tp" 2
cfg_set "dtype" "fp8_w8a8"
# Initialize sync_targets with detected mod path(s)
SYNC_INIT=()
[[ -n "$SPARK_DIR" && -d "$SPARK_DIR/mods" ]] && SYNC_INIT+=("$SPARK_DIR/mods/vllm-tune")
cfg_set_array "sync_targets" "${SYNC_INIT[@]+${SYNC_INIT[@]}}"
ok "Saved $VLLM_TUNE_CONFIG"
echo

# ── Step 3: PATH setup ──────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 3: Add to PATH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if echo "$PATH" | tr ':' '\n' | grep -qF "$INSTALL_DIR"; then
    ok "Already in PATH"
else
    detect_shell_rc
    if [[ -n "$SHELL_RC" ]] && ask_yn "Add $INSTALL_DIR to PATH in $SHELL_RC?"; then
        echo "" >> "$SHELL_RC"
        echo "# vLLM-Tune" >> "$SHELL_RC"
        echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$SHELL_RC"
        ok "Added to $SHELL_RC"
        info "Run 'source $SHELL_RC' or open a new terminal."
    else
        info "Skipped. Run manually:"
        echo "    export PATH=\"$INSTALL_DIR:\$PATH\""
    fi
fi
echo

# ── Step 4: Verify ──────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

VERSION=$("$INSTALL_DIR/vllm-tune.sh" --version 2>/dev/null || echo "unknown")
ok "Installed: $VERSION"

# Count pre-shipped configs
CONFIG_COUNT=$(find "$INSTALL_DIR/configs" -name '*.json' ! -name 'metadata.json' 2>/dev/null | wc -l)
ok "Pre-shipped configs: $CONFIG_COUNT"
echo

# ── Done ────────────────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  $(green "Installation complete!")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "  Quick start:"
echo "    $(bold "# Full tuning + deploy:")"
echo "    $INSTALL_DIR/vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --deploy"
echo
echo "    $(bold "# Deploy pre-shipped configs only:")"
echo "    $INSTALL_DIR/vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --deploy-only"
echo
echo "    $(bold "# Dry-run (no changes):")"
echo "    $INSTALL_DIR/vllm-tune.sh Qwen/Qwen3.6-35B-A3B-FP8 --tp 2 --dry-run"
echo
echo "  Documentation: $INSTALL_DIR/README.md"
echo
