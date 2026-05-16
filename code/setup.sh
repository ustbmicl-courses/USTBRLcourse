#!/usr/bin/env bash
# DRL 教学代码 — 一键环境搭建 + Jupyter 启动
#
# ┌──────────────────────────────────────────────────────────────┐
# │ 用法                                                          │
# ├──────────────────────────────────────────────────────────────┤
# │   bash setup.sh                # 全流程（建 env + 装依赖 +    │
# │                                #          clone 仓库 + 启动） │
# │   bash setup.sh install        # 只装依赖                     │
# │   bash setup.sh clone          # 只 clone 外部仓库            │
# │   bash setup.sh launch         # 只启动 jupyter               │
# │   bash setup.sh check          # 验证环境                     │
# │   bash setup.sh reuse <env>    # 复用已有 conda env，跳过创建 │
# │                                                              │
# │ 环境变量（可选）：                                             │
# │   DRLBOOK_ENV   conda env 名，默认 drlbook                    │
# │   PY_VERSION    Python 版本，默认 3.11                        │
# │   SKIP_TORCH=1  跳过 PyTorch 安装（仅 L1-L7 表格 RL）        │
# │   SKIP_VMAS=1   跳过 VMAS 安装（仅 L1-L11）                  │
# └──────────────────────────────────────────────────────────────┘

set -e

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
ENV_NAME="${DRLBOOK_ENV:-drlbook}"
PY_VERSION="${PY_VERSION:-3.11}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTERNAL_DIR="$SCRIPT_DIR/external"

# macOS 系统代理常导致 pip 卡死 → 强制绕过
export no_proxy='*'
export NO_PROXY='*'
PIP_FLAGS=(--proxy "" --no-cache-dir)

# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
log()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m[ ok  ]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn ]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; }

ensure_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        err "未找到 conda。请先安装 miniforge/miniconda: https://github.com/conda-forge/miniforge"
        exit 1
    fi
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
}

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

# ----------------------------------------------------------------------------
# 步骤
# ----------------------------------------------------------------------------
create_env() {
    if env_exists "$ENV_NAME"; then
        ok "conda env '$ENV_NAME' 已存在，跳过创建"
    else
        log "创建 conda env '$ENV_NAME' (python $PY_VERSION)"
        conda create -y -n "$ENV_NAME" python="$PY_VERSION"
    fi
}

clone_repos() {
    log "clone 外部参考仓库（Gymnasium / VMAS 用作 L1, L12 主环境）"
    if [ ! -f "$EXTERNAL_DIR/clone_all.sh" ]; then
        warn "未找到 $EXTERNAL_DIR/clone_all.sh，跳过 clone 步骤"
        return
    fi
    bash "$EXTERNAL_DIR/clone_all.sh"
}

install_deps() {
    log "激活 env '$ENV_NAME'"
    conda activate "$ENV_NAME"

    log "升级 pip"
    pip install "${PIP_FLAGS[@]}" -U pip wheel setuptools >/dev/null

    log "安装核心：numpy / matplotlib / Jupyter"
    pip install "${PIP_FLAGS[@]}" \
        "numpy>=1.23" \
        "matplotlib>=3.6" \
        "jupyter>=1.0" \
        "jupyterlab>=4.0" \
        "ipykernel>=6.0" \
        "nbformat>=5.0" \
        "nbconvert>=7.0"

    log "安装 RL 环境：gymnasium[classic-control,toy-text]（L1, L6, L7）"
    pip install "${PIP_FLAGS[@]}" "gymnasium[classic-control,toy-text]>=0.29"

    if [ "${SKIP_TORCH:-0}" != "1" ]; then
        log "安装 PyTorch（L8+ 深度 RL）"
        pip install "${PIP_FLAGS[@]}" "torch>=2.0" "tensorboard>=2.10"
    else
        warn "SKIP_TORCH=1 → 跳过 PyTorch（仅可跑 L1-L7 表格 RL）"
    fi

    if [ "${SKIP_VMAS:-0}" != "1" ] && [ -d "$EXTERNAL_DIR/VectorizedMultiAgentSimulator" ]; then
        log "安装 VMAS（L12 MARL 主环境，editable from local clone）"
        pip install "${PIP_FLAGS[@]}" -e "$EXTERNAL_DIR/VectorizedMultiAgentSimulator" || \
            warn "VMAS 安装失败（可能 macOS 代理）。可手动：no_proxy='*' pip install --proxy '' -e $EXTERNAL_DIR/VectorizedMultiAgentSimulator"
    fi

    log "注册 Jupyter kernel '$ENV_NAME'"
    python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)" >/dev/null
    ok "依赖安装完成"
}

verify_env() {
    log "在 env '$ENV_NAME' 中验证导入"
    conda activate "$ENV_NAME"
    DRLBOOK_REPO="$REPO_ROOT" python - <<'PYEOF'
import importlib, os, sys
core = ["numpy", "matplotlib", "gymnasium", "jupyter", "ipykernel", "nbformat"]
optional = ["torch", "vmas"]
print("Python:", sys.version.split()[0])
fail = False
for m in core:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  ✅ {m:<12} {v}")
    except ImportError as e:
        print(f"  ❌ {m:<12} ({e})")
        fail = True
for m in optional:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  ✅ {m:<12} {v}  (optional)")
    except ImportError:
        print(f"  ⚠️  {m:<12} 未安装  (optional, 仅 L8+ / L12 需要)")

# GridWorld 自检
sys.path.insert(0, os.path.join(os.environ.get("DRLBOOK_REPO", ""), "code"))
try:
    from shared.grid_world import GridWorld
    env = GridWorld()
    V, _, res = env.value_iteration()
    print(f"\n✅ shared.GridWorld 自检通过: |S|={env.n_states}, VI 收敛 in {len(res)} 轮, V*[s1]={V[0]:.4f}")
except Exception as e:
    print(f"\n❌ shared.GridWorld 自检失败: {e}")
    fail = True

# Gymnasium 自检
try:
    import gymnasium as gym
    e = gym.make("CartPole-v1"); obs, _ = e.reset(seed=0); e.close()
    print(f"✅ Gymnasium CartPole-v1 reset 成功, obs shape={obs.shape}")
except Exception as e:
    print(f"⚠️  Gymnasium CartPole 测试失败: {e}")

if fail:
    sys.exit(1)
PYEOF
    ok "环境验证通过"
}

launch_jupyter() {
    log "激活 env '$ENV_NAME' 并启动 Jupyter"
    conda activate "$ENV_NAME"
    cd "$SCRIPT_DIR"
    log "工作目录: $SCRIPT_DIR"
    log "推荐打开顺序: L01_MDP_foundations/01_gridworld_basics.ipynb → 02 → 03 → L02_... → L03_..."
    log "（按 Ctrl+C 停止 Jupyter）"
    jupyter notebook --notebook-dir="$SCRIPT_DIR"
}

show_summary() {
    cat <<'EOF'

═══════════════════════════════════════════════════════════════
DRL 教学代码 — 一键启动脚本
═══════════════════════════════════════════════════════════════
EOF
    cat <<EOF
  conda env:  $ENV_NAME  (python $PY_VERSION)
  代码目录:    $SCRIPT_DIR
  外部仓库:    $EXTERNAL_DIR

可用 Notebook:
  L01_MDP_foundations/   (3 notebooks)  ✅ 已建并跑通
  L02_Bellman_equation/  (3 notebooks)  ✅ 已建
  L03_Bellman_optimality/(3 notebooks)  ✅ 已建
  L4-dynamic-programming/(VI/PI .py)    ✅ 已建并跑通

EOF
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
ACTION="${1:-all}"
ensure_conda
show_summary

case "$ACTION" in
    all)
        create_env
        clone_repos
        install_deps
        verify_env
        launch_jupyter
        ;;
    install)
        create_env
        install_deps
        verify_env
        ok "依赖安装完成。运行 'bash setup.sh launch' 启动 Jupyter"
        ;;
    clone)
        clone_repos
        ;;
    launch)
        if ! env_exists "$ENV_NAME"; then
            err "conda env '$ENV_NAME' 不存在。先运行 'bash setup.sh install'"
            exit 1
        fi
        launch_jupyter
        ;;
    check)
        if ! env_exists "$ENV_NAME"; then
            err "conda env '$ENV_NAME' 不存在。先运行 'bash setup.sh install'"
            exit 1
        fi
        verify_env
        ;;
    reuse)
        # 复用已有 env：bash setup.sh reuse dc-mcts
        REUSE_ENV="${2:-}"
        if [ -z "$REUSE_ENV" ]; then
            err "用法: bash setup.sh reuse <existing-env-name>"
            exit 1
        fi
        if ! env_exists "$REUSE_ENV"; then
            err "conda env '$REUSE_ENV' 不存在"
            exit 1
        fi
        ENV_NAME="$REUSE_ENV"
        log "复用现有 env '$ENV_NAME'，跳过创建步骤"
        install_deps
        verify_env
        launch_jupyter
        ;;
    *)
        err "未知动作: $ACTION  (可用: all | install | clone | launch | check | reuse <env>)"
        exit 1
        ;;
esac
