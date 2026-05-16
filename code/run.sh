#!/usr/bin/env bash
# DRLbook 教学代码统一入口
# 位置：code/run.sh
# 用法：bash run.sh <command>
#
# 命令：
#   setup        创建独立 venv（code/.venv/）+ 安装 requirements.txt
#   check        检查环境与依赖完整性（不修改任何东西）
#   status       一览：venv / 依赖 / 外部仓库 / 各讲代码完成情况
#   run <id>     运行指定讲的演示，例如  bash run.sh run l1
#                可用 id：l1 / l1-cartpole / l2 / l3 / l4 / l4-pi / l4-vi / l4-conv / l4-qvi / l5 / l6 / l7
#   jupyter      在 venv 内启动 JupyterLab 并自动打开浏览器
#                选项：--lab (默认) / --notebook / --no-browser / --port <N>
#   compile <id>    仅编译该讲的 PPT-cn（xelatex 两遍），不触发 notebook 执行
#                可用 id：l1 / l2 / l3 / l4 / l5 / l6 / l7。选项：--open 完成后弹 PDF
#   notebooks <id>  仅执行该讲所有 .ipynb（重生成 figures/），不触发 PPT 编译
#                选项：--check 只显示将要执行的列表，不实际执行
#   test         运行 pytest（code/tests/） —— 测 shared.grid_world / shared.uav_grid
#                额外参数透传 pytest，例如 bash run.sh test -k bellman
#   clone        执行 external/clone_all.sh，拉取教学参考仓库
#   clean        清理 __pycache__ 与可选的 .venv
#   help         显示帮助
#
# 注：PPT 编译与代码执行**完全解耦**。改 PPT 用 compile，跑代码 / 重生成图用
#     notebooks 或 run / jupyter。不再提供合并流水。
#
# 安全约定：除 setup / clean / clone 外，本脚本不修改文件系统。

set -euo pipefail

# ----------- 路径与颜色 -----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQ_FILE="$SCRIPT_DIR/requirements.txt"
EXTERNAL_DIR="$SCRIPT_DIR/external"
PY_MIN="3.9"

if [ -t 1 ]; then
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
    BLUE=$'\033[34m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    GREEN=""; YELLOW=""; RED=""; BLUE=""; BOLD=""; RESET=""
fi

ok()    { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
warn()  { printf "  ${YELLOW}!${RESET} %s\n" "$*"; }
fail()  { printf "  ${RED}✗${RESET} %s\n" "$*"; }
info()  { printf "${BLUE}==>${RESET} ${BOLD}%s${RESET}\n" "$*"; }
die()   { printf "${RED}error: %s${RESET}\n" "$*" >&2; exit 1; }

# ----------- python 检测 -----------
# 搜索顺序：直接 PATH → conda envs → brew Python → /usr/local/bin → /opt/homebrew/bin
collect_python_candidates() {
    local cands=()
    for p in python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
        if command -v "$p" >/dev/null 2>&1; then
            cands+=("$(command -v "$p")")
        fi
    done
    # conda envs
    if command -v conda >/dev/null 2>&1; then
        local base
        base="$(conda info --base 2>/dev/null || true)"
        if [ -n "$base" ] && [ -d "$base/envs" ]; then
            for env in "$base/envs"/*/bin/python*; do
                [ -x "$env" ] && cands+=("$env")
            done
            [ -x "$base/bin/python3" ] && cands+=("$base/bin/python3")
        fi
    fi
    # brew / typical locations
    for d in /opt/homebrew/bin /usr/local/bin /opt/homebrew/opt/python@3.11/bin /opt/homebrew/opt/python@3.12/bin; do
        for p in "$d"/python3.{9,10,11,12,13} "$d"/python3; do
            [ -x "$p" ] && cands+=("$p")
        done
    done
    # 去重保持顺序
    printf "%s\n" "${cands[@]}" | awk '!x[$0]++'
}

find_python() {
    while IFS= read -r p; do
        [ -x "$p" ] || continue
        local v
        v="$("$p" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || continue
        if [ "$(printf "%s\n%s\n" "$PY_MIN" "$v" | sort -V | head -1)" = "$PY_MIN" ]; then
            echo "$p"
            return 0
        fi
    done < <(collect_python_candidates)
    return 1
}

list_python_candidates() {
    while IFS= read -r p; do
        [ -x "$p" ] || continue
        local v
        v="$("$p" --version 2>&1 | awk '{print $2}')"
        printf "    %s  (%s)\n" "$p" "$v"
    done < <(collect_python_candidates)
}

venv_python() {
    echo "$VENV_DIR/bin/python"
}

ensure_venv() {
    [ -x "$(venv_python)" ] || die "venv 未就绪，请先运行：bash run.sh setup"
}

# ----------- 子命令实现 -----------
cmd_setup() {
    info "创建独立 Python 环境（venv）"
    local PY
    if ! PY="$(find_python)"; then
        die "未找到 Python ${PY_MIN}+，请先安装 (例如：brew install python@3.11)"
    fi
    ok "使用 $PY ($($PY --version 2>&1 | awk '{print $2}'))"

    if [ -d "$VENV_DIR" ]; then
        warn "venv 已存在：$VENV_DIR （将复用，如需重建请先 clean --venv）"
    else
        "$PY" -m venv "$VENV_DIR"
        ok "创建 venv: $VENV_DIR"
    fi

    # PyPI 镜像（国内默认走清华源；用 PIP_INDEX_URL 可覆盖）
    local PIP_MIRROR="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    info "PyPI 镜像: ${PIP_MIRROR}"
    info "升级 pip / wheel"
    "$(venv_python)" -m pip install --upgrade pip wheel -i "${PIP_MIRROR}" >/dev/null
    ok "pip / wheel 已升级"

    info "安装依赖（${REQ_FILE}）"
    [ -f "${REQ_FILE}" ] || die "requirements.txt 不存在：${REQ_FILE}"
    "$(venv_python)" -m pip install -r "${REQ_FILE}" -i "${PIP_MIRROR}"
    ok "依赖安装完成"

    cmd_check
    info "完成。激活 venv："
    echo "    source $VENV_DIR/bin/activate"
}

cmd_check() {
    info "环境检查（只读，不修改文件）"
    local exit_code=0

    # ---- Python ----
    local PY
    if PY="$(find_python)"; then
        ok "宿主 Python: $PY ($($PY --version 2>&1 | awk '{print $2}'))"
    else
        fail "未找到 Python ${PY_MIN}+。已扫描的候选："
        list_python_candidates
        warn "请安装 Python (推荐：brew install python@3.11) 或激活合适的 conda env 后重试"
        exit_code=1
    fi

    # ---- venv ----
    if [ -x "$(venv_python)" ]; then
        ok "venv 存在: $VENV_DIR ($("$(venv_python)" --version 2>&1 | awk '{print $2}'))"
    else
        warn "venv 未创建。运行 'bash run.sh setup' 即可"
        exit_code=1
    fi

    # ---- 依赖 ----
    if [ -x "$(venv_python)" ]; then
        info "venv 内依赖检查"
        check_pkg numpy       "L1/L4 必需"
        check_pkg matplotlib  "L1/L4 绘图"
        check_pkg gymnasium   "L1 CartPole 可选"
        check_pkg torch       "L8+ 深度 RL 可选"
    fi

    # ---- 外部仓库 ----
    info "外部参考仓库（code/external/）"
    if [ -d "$EXTERNAL_DIR/Gymnasium/.git" ]; then
        ok "Gymnasium  ($(cd "$EXTERNAL_DIR/Gymnasium" && git log -1 --format=%h\ %s 2>/dev/null | head -c 70))"
    else
        warn "Gymnasium 未克隆。运行 'bash run.sh clone' 即可"
    fi
    if [ -d "$EXTERNAL_DIR/VectorizedMultiAgentSimulator/.git" ]; then
        ok "VectorizedMultiAgentSimulator"
    else
        warn "VectorizedMultiAgentSimulator 未克隆"
    fi

    # ---- 教学代码 ----
    info "各讲代码完成情况"
    for d in "$SCRIPT_DIR"/L*/; do
        [ -d "$d" ] || continue
        local name py_count
        name="$(basename "$d")"
        py_count=$(find "$d" -maxdepth 1 -name '*.py' | wc -l | tr -d ' ')
        if [ "$py_count" -gt 0 ]; then
            ok "$name: $py_count 个 .py"
        else
            warn "$name: 暂无 .py 文件"
        fi
    done

    return $exit_code
}

check_pkg() {
    local pkg="$1" desc="$2"
    # torch 在部分 macOS/Python 组合下执行 "import torch" 可能触发 Abort trap，
    # 这里改为 pip 元数据探测，避免 check 阶段出现噪声错误。
    if [ "$pkg" = "torch" ]; then
        local ver
        ver=$("$(venv_python)" -m pip show torch 2>/dev/null | awk -F': ' '/^Version:/{print $2; exit}')
        if [ -n "$ver" ]; then
            ok "$pkg ($ver)  — $desc"
        else
            warn "$pkg 未安装  — $desc"
        fi
        return 0
    fi

    if "$(venv_python)" -c "import $pkg" 2>/dev/null; then
        local ver
        ver=$("$(venv_python)" -c "import $pkg; print(getattr($pkg, '__version__', '?'))" 2>/dev/null)
        ok "$pkg ($ver)  — $desc"
    else
        warn "$pkg 未安装  — $desc"
    fi
}

cmd_status() {
    info "DRLbook code/ 状态"
    cmd_check || true
}

cmd_run() {
    [ "${1:-}" ] || die "用法: bash run.sh run <id>。可选：l1 l1-cartpole l2 l3 l4 l4-pi l4-vi l4-conv l5 l6 l7"
    ensure_venv
    local id="$1"; shift || true
    local PY
    PY="$(venv_python)"

    case "$id" in
        l1)            run_one "$SCRIPT_DIR/L1-mdp-foundations/grid_world.py" "$@" ;;
        l1-cartpole)   run_one "$SCRIPT_DIR/L1-mdp-foundations/cartpole_random.py" "$@" ;;
        l1-vis)        run_one "$SCRIPT_DIR/L1-mdp-foundations/visualize_grid.py" "$@" ;;
        l2)            run_one "$SCRIPT_DIR/L2-bellman-equation/policy_evaluation.py" "$@" ;;
        l3)            run_one "$SCRIPT_DIR/L3-bellman-optimality/bellman_optimality.py" "$@" ;;
        l4|l4-vi)      run_one "$SCRIPT_DIR/L4-dynamic-programming/value_iteration.py" "$@" ;;
        l4-pi)         run_one "$SCRIPT_DIR/L4-dynamic-programming/policy_iteration.py" "$@" ;;
        l4-conv)       run_one "$SCRIPT_DIR/L4-dynamic-programming/convergence_plot.py" "$@" ;;
        l4-qvi)        run_one "$SCRIPT_DIR/L4-dynamic-programming/q_value_iteration.py" "$@" ;;
        l5)            run_one "$SCRIPT_DIR/L5-stochastic-approximation/rm_step_size_demo.py" "$@" ;;
        l6)            run_one "$SCRIPT_DIR/L6-monte-carlo/mc_convergence_demo.py" "$@" ;;
        l7)            run_one "$SCRIPT_DIR/L7-td-learning/td_vs_mc_demo.py" "$@" ;;
        *)             die "未知 id: $id" ;;
    esac
}

run_one() {
    local script="$1"; shift || true
    [ -f "$script" ] || die "脚本不存在: $script"
    info "运行 $script"
    "$(venv_python)" "$script" "$@"
}

cmd_jupyter() {
    ensure_venv
    local mode="lab"
    local port=8888
    local browser_arg=""
    local extra_args=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --lab)        mode="lab"; shift ;;
            --notebook)   mode="notebook"; shift ;;
            --no-browser) browser_arg="--no-browser"; shift ;;
            --port)       port="$2"; shift 2 ;;
            *)            extra_args+=("$1"); shift ;;
        esac
    done

    local PIP_MIRROR="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    local PY
    PY="$(venv_python)"

    # 检查 jupyter / jupyterlab；缺则按需安装
    info "检查 Jupyter 组件"
    local need_install=()
    if [ "$mode" = "lab" ]; then
        "$PY" -c 'import jupyterlab' 2>/dev/null || need_install+=(jupyterlab)
    fi
    "$PY" -c 'import notebook' 2>/dev/null || need_install+=(notebook)
    "$PY" -c 'import ipykernel' 2>/dev/null || need_install+=(ipykernel)

    if [ ${#need_install[@]} -gt 0 ]; then
        info "安装：${need_install[*]}（PyPI: ${PIP_MIRROR}）"
        "$PY" -m pip install -i "${PIP_MIRROR}" "${need_install[@]}"
        ok "Jupyter 组件安装完成"
    else
        ok "Jupyter 组件已就绪"
    fi

    # 注册当前 venv 为 Jupyter kernel（如未注册）
    if ! "$PY" -m jupyter kernelspec list 2>/dev/null | grep -q "drlbook"; then
        info "把 venv 注册为 Jupyter kernel: drlbook"
        "$PY" -m ipykernel install --user --name drlbook --display-name "Python (DRLbook venv)" >/dev/null
        ok "Kernel 已注册：drlbook"
    else
        ok "Kernel 已注册：drlbook"
    fi

    info "启动 Jupyter${mode:+ $mode} 于 http://localhost:${port}"
    info "工作目录: $SCRIPT_DIR"
    info "按 Ctrl+C 停止"

    cd "$SCRIPT_DIR"
    # bash 3.x 下空数组在 set -u 下报 unbound，用 +"${...[@]}" 形式安全展开
    if [ "$mode" = "lab" ]; then
        exec "$PY" -m jupyter lab --port="${port}" ${browser_arg} ${extra_args[@]+"${extra_args[@]}"}
    else
        exec "$PY" -m jupyter notebook --port="${port}" ${browser_arg} ${extra_args[@]+"${extra_args[@]}"}
    fi
}

# 配对查找表：id → (code_dir, ppt_dir, tex_file)
# 由 cmd_compile / cmd_notebooks 共享
resolve_pair() {
    local id="$1"
    case "$id" in
        l1) echo "$SCRIPT_DIR/L01_MDP_foundations" \
                 "$SCRIPT_DIR/../ustbmicl_lectureNotes/L1-mdp-foundations-cn" \
                 "L1-basic.tex" ;;
        l2) echo "$SCRIPT_DIR/L02_Bellman_equation" \
                 "$SCRIPT_DIR/../ustbmicl_lectureNotes/L2-bellman-equation-cn" \
                 "L2-Bellman.tex" ;;
        l3) echo "$SCRIPT_DIR/L03_Bellman_optimality" \
                 "$SCRIPT_DIR/../ustbmicl_lectureNotes/L3-bellman-optimality-cn" \
                 "L3-BOE.tex" ;;
        l4) echo "$SCRIPT_DIR/L04_Dynamic_Programming" \
                 "$SCRIPT_DIR/../ustbmicl_lectureNotes/L4-dynamic-programming-cn" \
                 "L4-DP.tex" ;;
        l5) echo "$SCRIPT_DIR/L05_Stochastic_approximation" \
                 "$SCRIPT_DIR/../ustbmicl_lectureNotes/L5-stochastic-approximation-cn" \
                 "L5-SA.tex" ;;
        l6) echo "$SCRIPT_DIR/L6-monte-carlo" \
                 "$SCRIPT_DIR/../ustbmicl_lectureNotes/L6-monte-carlo-cn" \
                 "L6-MC.tex" ;;
        l7) echo "$SCRIPT_DIR/L7-td-learning" \
                 "$SCRIPT_DIR/../ustbmicl_lectureNotes/L7-td-learning-cn" \
                 "L7-TD.tex" ;;
        l8) echo "$SCRIPT_DIR/L08_Value_function_approx" \
                 "$SCRIPT_DIR/../ustbmicl_lectureNotes/L8-value-function-approx-cn" \
                 "L8-VFA.tex" ;;
        *)  return 1 ;;
    esac
}

# ---- compile <id>：仅编译 PPT，与 notebook 执行无关 ----
cmd_compile() {
    [ "${1:-}" ] || die "用法: bash run.sh compile <id>。可选：l1 / l2 / l3 / l4 / l5 / l6 / l7 / l8"
    local id="$1"; shift || true
    local do_open=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --open) do_open=1; shift ;;
            *)      die "compile: 未知选项 $1" ;;
        esac
    done

    local pair
    pair=$(resolve_pair "$id") || die "未知 id: $id"
    # shellcheck disable=SC2206
    local arr=($pair)
    local ppt_dir="${arr[1]}" tex_file="${arr[2]}"

    [ -f "${ppt_dir}/${tex_file}" ] || die "PPT .tex 不存在: ${ppt_dir}/${tex_file}"

    local xelatex_bin
    xelatex_bin="$(command -v xelatex 2>/dev/null || echo /Library/TeX/texbin/xelatex)"
    [ -x "$xelatex_bin" ] || die "找不到 xelatex；请安装 MacTeX"

    info "编译 PPT-cn  id=${id}  (${tex_file})"
    info "  目录: ${ppt_dir}"
    (
        cd "$ppt_dir"
        "$xelatex_bin" -interaction=nonstopmode "$tex_file" >/dev/null 2>&1 || \
            { fail "  第一遍 xelatex 失败，看 ${tex_file%.tex}.log"; exit 1; }
        "$xelatex_bin" -interaction=nonstopmode "$tex_file" >/dev/null 2>&1 || \
            { fail "  第二遍 xelatex 失败，看 ${tex_file%.tex}.log"; exit 1; }
    )

    local pdf_file="${tex_file%.tex}.pdf"
    local pdf_path="${ppt_dir}/${pdf_file}"
    if [ -f "$pdf_path" ]; then
        local pages=""
        if command -v pdfinfo >/dev/null 2>&1; then
            pages=$(pdfinfo "$pdf_path" | awk '/^Pages:/{print $2}')
        fi
        ok "${pdf_file}${pages:+ ($pages 页)}"
        if [ "$do_open" -eq 1 ] && command -v open >/dev/null 2>&1; then
            open "$pdf_path"
        fi
    else
        fail "PDF 未生成"
    fi
}

# ---- notebooks <id>：仅执行该讲所有 .ipynb，与 PPT 编译无关 ----
cmd_notebooks() {
    [ "${1:-}" ] || die "用法: bash run.sh notebooks <id>。可选：l1 / l2 / l3 / l4 / l5 / l6 / l7 / l8"
    ensure_venv

    local id="$1"; shift || true
    local check_only=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --check) check_only=1; shift ;;
            *)       die "notebooks: 未知选项 $1" ;;
        esac
    done

    local pair
    pair=$(resolve_pair "$id") || die "未知 id: $id"
    # shellcheck disable=SC2206
    local arr=($pair)
    local code_dir="${arr[0]}"

    [ -d "$code_dir" ] || die "代码目录不存在: $code_dir"

    # 收集 .ipynb
    local nbs=()
    for nb in "$code_dir"/*.ipynb; do
        [ -f "$nb" ] && nbs+=("$nb")
    done
    if [ ${#nbs[@]} -eq 0 ]; then
        warn "未找到 *.ipynb 于 ${code_dir}"
        return 0
    fi

    if [ "$check_only" -eq 1 ]; then
        info "将要执行的 notebook（id=${id}, ${#nbs[@]} 个）："
        for nb in "${nbs[@]}"; do
            printf "    %s\n" "$(basename "$nb")"
        done
        return 0
    fi

    info "执行 notebook  id=${id}  (${#nbs[@]} 个)"
    info "  目录: ${code_dir}"
    local PY
    PY="$(venv_python)"
    "$PY" -c 'import nbconvert' 2>/dev/null || {
        warn "nbconvert 未装，安装中"
        "$PY" -m pip install -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" nbconvert
    }
    local ok_count=0 fail_count=0
    for nb in "${nbs[@]}"; do
        printf "    → %s ... " "$(basename "$nb")"
        if "$PY" -m jupyter nbconvert --to notebook --execute --inplace "$nb" >/dev/null 2>&1; then
            printf "${GREEN}ok${RESET}\n"
            ok_count=$((ok_count+1))
        else
            printf "${RED}fail${RESET}\n"
            fail_count=$((fail_count+1))
        fi
    done
    if [ $fail_count -eq 0 ]; then
        ok "全部 ${ok_count} 个 notebook 执行成功（figures/ 已重生成）"
    else
        fail "${fail_count} 个失败，${ok_count} 成功；运行 'jupyter nbconvert --execute <nb>' 看详细错误"
        return 1
    fi
}

cmd_clone() {
    info "clone 外部参考仓库"
    [ -f "$EXTERNAL_DIR/clone_all.sh" ] || die "clone_all.sh 不存在"
    bash "$EXTERNAL_DIR/clone_all.sh"
}

cmd_test() {
    ensure_venv
    local PY
    PY="$(venv_python)"
    # 按需自动装 pytest（首次使用 test 子命令时）
    if ! "$PY" -c 'import pytest' 2>/dev/null; then
        warn "pytest 未装，安装中"
        "$PY" -m pip install pytest -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" >/dev/null
        ok "pytest 已装"
    fi
    [ -d "$SCRIPT_DIR/tests" ] || die "tests 目录不存在：$SCRIPT_DIR/tests"
    info "运行 pytest（code/tests/）"
    exec "$PY" -m pytest "$SCRIPT_DIR/tests/" "$@"
}

cmd_clean() {
    local venv_too=false
    if [ "${1:-}" = "--venv" ] || [ "${1:-}" = "--all" ]; then venv_too=true; fi

    info "清理 __pycache__"
    find "$SCRIPT_DIR" -type d -name '__pycache__' -not -path '*/external/*' -prune -exec rm -rf {} +
    ok "__pycache__ 已清理"

    if $venv_too; then
        if [ -d "$VENV_DIR" ]; then
            rm -rf "$VENV_DIR"
            ok "venv 已删除：$VENV_DIR"
        fi
    else
        info "如需删除 venv：bash run.sh clean --venv"
    fi
}

cmd_help() {
    sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

# ----------- 入口 -----------
case "${1:-help}" in
    setup)   shift; cmd_setup    "$@" ;;
    check)   shift; cmd_check    "$@" ;;
    status)  shift; cmd_status   "$@" ;;
    run)     shift; cmd_run      "$@" ;;
    jupyter|nb|lab) shift; cmd_jupyter "$@" ;;
    compile)        shift; cmd_compile   "$@" ;;
    notebooks|nbrun) shift; cmd_notebooks "$@" ;;
    test|pytest) shift; cmd_test "$@" ;;
    clone)          shift; cmd_clone     "$@" ;;
    clean)   shift; cmd_clean    "$@" ;;
    help|-h|--help) cmd_help ;;
    *)       cmd_help; die "未知命令: $1" ;;
esac
