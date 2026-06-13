#!/usr/bin/env bash
# evolve.sh — autonomous INT8 kernel optimization loop.
#
# Each iteration: profile (ncu) -> let claude make ONE rule-compliant change
# -> compile-check -> 5-gate accuracy validation -> perf sweep vs baseline
# -> commit + append the OPTIMIZATION_LOG.md iteration record. Any failed check
# restores the tree and moves on.
#
# Usage:
#   ./evolve.sh                 # 5 iterations
#   MAX_ITER=10 ./evolve.sh     # override iteration count
#
# Requirements: clean git tree, passwordless sudo for ncu (falls back to
# torch.profiler), `claude` CLI on PATH.

set -uo pipefail
cd "$(dirname "$0")"

MAX_ITER="${MAX_ITER:-5}"
LOG_DIR="${LOG_DIR:-/tmp/evolve_logs}"
CLAUDE_TIMEOUT="${CLAUDE_TIMEOUT:-1800}"
# Sweeps are noisy (~2% run-to-run); take the per-config median of this many
# runs so the accept/reject gate doesn't fire on noise. Raise for less noise.
SWEEP_REPS="${SWEEP_REPS:-3}"
# Latency thresholds (percent) for the 3-way perf gate, see compare_sweep.
PERF_IMPROVE="${PERF_IMPROVE:--2.0}"   # <= this  -> genuine improvement (commit)
PERF_REGRESS="${PERF_REGRESS:-5.0}"    # >  this  -> regression (revert)
mkdir -p "$LOG_DIR"

# CLAUDE.md §3 metrics. occupancy_limit_{registers,shared_mem} +
# maximum_warps_per_active_cycle are here so the diagnosis can see WHICH
# resource caps blocks/SM: on the graded head_dim=64 shape both registers AND
# shared mem cap at 3 blocks/SM and warps_active (~18%) already equals the
# ceiling, so a registers-only cut is a no-op — only a JOINT reg+smem cut
# reaches 4 blocks/SM.
NCU_METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
launch__registers_per_thread,launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,sm__maximum_warps_per_active_cycle_pct"

PASS_COUNT=0
declare -a ITER_RESULTS

log() { echo "[evolve $(date +%H:%M:%S)] $*"; }

run_claude() {  # run_claude <logfile>  (prompt on stdin)
    timeout "$CLAUDE_TIMEOUT" claude --print --dangerously-skip-permissions \
        < /dev/stdin 2>&1 | tee "$1"
}

restore_tree() {
    git reset --hard HEAD >/dev/null
    log "tree restored to HEAD"
}

compile_check() {  # syntax + ptxas check of both INT8 kernels
    local out
    for f in kernels/int8_attention.cu kernels/int8_mlp.cu; do
        out=$(nvcc -arch=sm_80 -O3 --std=c++17 -I kernels \
                   --ptxas-options=-v -c "$f" -o /dev/null 2>&1) || {
            echo "$out"; return 1; }
        echo "$out" | grep -E "spill|registers" | head -4
    done
    return 0
}

# Profile the attention kernel on the SWEEP-matched shape (batch=8, head_dim=64)
# into $1. Pre-warms the JIT cache first so ncu profiles the kernel, not the
# build. Returns 0 only if ncu metric data was written (1 if sudo/ncu absent).
run_ncu() {  # run_ncu <outfile>
    local out="$1"
    python3 profile_kernel.py both > /dev/null 2>&1     # pre-warm JIT cache
    sudo -n true 2>/dev/null && command -v ncu >/dev/null || return 1
    # env PATH/HOME: keep the user's JIT cache + ninja visible under sudo,
    # otherwise root rebuilds the extension inside the profiler (or fails).
    sudo env "PATH=$PATH" HOME="$HOME" \
        ncu --kernel-name regex:int8_wmma --launch-count 2 \
        --metrics "$NCU_METRICS" \
        python3 profile_kernel.py attn 2>&1 \
        | grep -E "int8_|Metric Name|----|pct|registers|wavefronts|occupancy|maximum_warps" \
        > "$out"
    grep -q "registers" "$out" 2>/dev/null
}

# Did a bottleneck metric move beyond run-to-run noise between two ncu reports?
# Prints a human description ("mem_shared -24.3%, tensor_pipe +2.1pp") and exits
# 0 if any moved; prints nothing and exits 1 otherwise. Used to catch a
# latency-neutral change that nonetheless SHIFTED the bottleneck — a negative
# result worth recording (per CLAUDE.md §4) instead of silently reverting.
metric_moved() {  # metric_moved <baseline_ncu.txt> <after_ncu.txt>
    python3 - "$1" "$2" <<'EOF'
import re, sys
KEYS = ("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")
def grab(path):
    m = {}
    for line in open(path):
        for k in KEYS:
            if k in line:
                nums = re.findall(r"[-+]?\d+\.?\d*", line.split(k, 1)[1])
                if nums: m.setdefault(k, []).append(float(nums[-1]))
    return {k: sum(v)/len(v) for k, v in m.items()}   # avg the 2 launches
try:
    b, a = grab(sys.argv[1]), grab(sys.argv[2])
except Exception:
    sys.exit(1)
moved = []
ms = KEYS[0]   # wavefront sum: relative %
if b.get(ms, 0) > 0 and ms in a:
    d = (a[ms] - b[ms]) / b[ms] * 100
    if abs(d) >= 5.0: moved.append(f"mem_shared {d:+.1f}%")
for k, label, thr in ((KEYS[1], "warps_active", 1.0),
                      (KEYS[2], "tensor_pipe", 2.0)):   # percentages: pp
    if k in b and k in a and abs(a[k] - b[k]) >= thr:
        moved.append(f"{label} {a[k]-b[k]:+.1f}pp")
if moved:
    print(", ".join(moved)); sys.exit(0)
sys.exit(1)
EOF
}

# Run sweep.py SWEEP_REPS times and write the per-config MEDIAN kernel_ms back
# into results/<kernel>_sweep.csv (so the median is what gets compared and, on a
# committed iteration, what gets committed). Cuts the ~2% run-to-run noise that
# otherwise makes the perf gate fire randomly.
median_sweep() {  # median_sweep <int8_attn|int8_mlp> <logprefix>
    local kernel="$1" logpfx="$2" r
    local -a csvs=()
    for r in $(seq 1 "$SWEEP_REPS"); do
        python3 sweep.py --kernel "$kernel" > "${logpfx}_$r.log" 2>&1 || return 1
        cp "results/${kernel}_sweep.csv" "${logpfx}_$r.csv"
        csvs+=("${logpfx}_$r.csv")
    done
    python3 - "results/${kernel}_sweep.csv" "${csvs[@]}" <<'EOF'
import csv, sys, statistics
out, paths = sys.argv[1], sys.argv[2:]
runs = []
for p in paths:
    with open(p) as f:
        runs.append({(r["seq_len"], r["d_model"]): r for r in csv.DictReader(f)})
base = runs[0]
fields = list(next(iter(base.values())).keys())
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for k, r0 in base.items():
        vals = [float(run[k]["kernel_ms"]) for run in runs if k in run]
        row = dict(r0); row["kernel_ms"] = f"{statistics.median(vals):.6f}"
        w.writerow(row)
EOF
}

# Median kernel_ms ratio: new vs baseline CSV. Prints "+X.X%". 3-way exit code:
#   2 = regression  (> PERF_REGRESS%)        -> caller reverts
#   0 = improvement (<= PERF_IMPROVE%)       -> counts as real progress
#   1 = neutral     (within noise, in between) -> caller reverts as a no-op
compare_sweep() {  # compare_sweep <baseline.csv> <new.csv>
    PERF_IMPROVE="$PERF_IMPROVE" PERF_REGRESS="$PERF_REGRESS" python3 - "$1" "$2" <<'EOF'
import csv, os, sys
def ms(p):
    with open(p) as f:
        return {(r["seq_len"], r["d_model"]): float(r["kernel_ms"])
                for r in csv.DictReader(f)}
old, new = ms(sys.argv[1]), ms(sys.argv[2])
ratios = [new[k] / old[k] for k in old if k in new]
d = (sum(ratios) / len(ratios) - 1) * 100
print(f"{d:+.1f}%")
imp = float(os.environ["PERF_IMPROVE"]); reg = float(os.environ["PERF_REGRESS"])
sys.exit(2 if d > reg else (0 if d <= imp else 1))
EOF
}

# ════════════════════════════════════════════════════════════════════════
# 1. Initialization
# ════════════════════════════════════════════════════════════════════════
if [ -n "$(git status --porcelain)" ]; then
    log "ERROR: git tree not clean — commit or stash first."; exit 1
fi

if [ ! -d testdata/validate ]; then
    log "generating validation datasets ..."
    python3 generate_test_data.py || exit 1
fi

log "establishing accuracy baseline -> /tmp/baseline.txt"
python3 validate_int8.py > /tmp/baseline.txt 2>&1 || {
    log "ERROR: baseline validation FAILED — fix before evolving."
    tail -20 /tmp/baseline.txt; exit 1; }

log "establishing perf baseline (median of $SWEEP_REPS sweeps, int8_attn + int8_mlp) ..."
median_sweep int8_attn "$LOG_DIR/sweep_attn_base" || exit 1
median_sweep int8_mlp  "$LOG_DIR/sweep_mlp_base"  || exit 1
cp results/int8_attn_sweep.csv /tmp/baseline_attn.csv
cp results/int8_mlp_sweep.csv  /tmp/baseline_mlp.csv
git checkout -- results/ 2>/dev/null   # keep committed CSVs canonical
log "baselines ready."

# ════════════════════════════════════════════════════════════════════════
# 2. Main loop
# ════════════════════════════════════════════════════════════════════════
for i in $(seq 1 "$MAX_ITER"); do
    echo; log "════════ ITERATION $i / $MAX_ITER ════════"
    NCU_OUT="$LOG_DIR/ncu_$i.txt"

    # a. Profile on the SAME shape sweep.py grades (batch=8, head_dim=64) — NOT
    #    test_int8.py --quick (batch=2, grid-starved toy). Profiling the toy is
    #    why prior iterations chased a phantom occupancy/register bottleneck the
    #    graded workload does not have.
    log "profiling (sweep-matched shape: b=8 h=8 s=2048 d=64) ..."
    if ! run_ncu "$NCU_OUT"; then
        log "ncu unavailable/empty — falling back to torch.profiler"
        python3 - > "$NCU_OUT" 2>&1 <<'EOF'
import torch, sys; sys.path.insert(0, ".")
from torch.utils.cpp_extension import load
ext = load(name="int8_ext", sources=["kernels/int8_attention.cu",
    "kernels/int8_mlp.cu", "kernels/quant_utils.cu", "kernels/int8_ext.cu"],
    extra_cuda_cflags=["-arch=sm_80", "--std=c++17", "-O3"], verbose=False)
def q(t): s=t.float().abs().max()/127; return (t.float()/s).round().clamp(-128,127).to(torch.int8), float(s)
def qt(t):
    s=t.float().abs().amax(-1,keepdim=True).clamp(min=1e-8)/127
    return (t.float()/s).round().clamp(-128,127).to(torch.int8), s.squeeze(-1).contiguous()
Q=torch.randn(8,8,2048,64,device="cuda").half(); qi,sq=qt(Q)
x=torch.randn(8,512,1024,device="cuda").half(); W1=torch.randn(1024,4096,device="cuda").half()*0.02
W2=torch.randn(4096,1024,device="cuda").half()*0.02
xi,sx=q(x); w1,s1=q(W1); w2,s2=q(W2)
for _ in range(3):
    ext.int8_attention_forward(qi,qi,qi,sq,sq,sq)
    ext.int8_mlp_forward(xi,w1,w2,sx,s1,s2)
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(5):
        ext.int8_attention_forward(qi,qi,qi,sq,sq,sq)
        ext.int8_mlp_forward(xi,w1,w2,sx,s1,s2)
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=12))
EOF
    fi

    # b. One rule-compliant change.
    log "asking claude for ONE optimization ..."
    run_claude "$LOG_DIR/claude_change_$i.log" <<EOF
You are optimizing the INT8 CUDA kernels in this repo. Read CLAUDE.md fully
and follow §4 strictly. Profiling output for this iteration:

$(cat "$NCU_OUT")

The profiling above is from the SAME shape sweep.py grades (batch=8,
head_dim=64, seq=2048). Read the actual numbers above — do NOT assume the old
"warps_active ~7.4%" figure; that came from a grid-starved toy shape that is
no longer profiled. On the graded shape the real picture is: warps_active ~18%
which already EQUALS sm__maximum_warps_per_active_cycle_pct (~18.75%), i.e.
occupancy is MAXED at 3 blocks/SM, co-limited by BOTH
launch__occupancy_limit_registers=3 AND launch__occupancy_limit_shared_mem=3.
tensor pipe is ~28% (cuBLAS reaches 60%+); the smem round-trip
(l1tex__...mem_shared) is large.

Make ONE coherent structural change (one idea) to kernels/int8_attention.cu or
kernels/int8_mlp.cu (you may touch kernels/int8_common.cuh if shared). "One
change" is NOT a size limit: a full rewrite of a kernel's main loop counts as
one change. BE BOLD. A bold attempt that gets reverted by the gates is better
than a latency-neutral no-op. Do NOT downgrade to a one-line reorder to play
safe.

Where the real headroom is (pick ONE, justify from the numbers above):
- The kernel is occupancy-MAXED at 3 blocks/SM AND already runs P in registers
  via the INT8_ATTN_REGPV=1 default path (mma.sync m16n8k32 QK^T + P@V is
  ALREADY built — do NOT "add" it). With only ~12 warps/SM there are too few
  warps to hide the smem-round-trip + MMA latency, so tensor pipe stalls at
  ~28%. The two real levers are: (a) raise occupancy to 4 blocks/SM, which
  requires cutting BOTH registers AND shared-memory-per-block below their
  4-block thresholds TOGETHER (a registers-only cut is a NO-OP — smem still
  caps at 3); or (b) cut the smem round-trip / raise per-warp ILP so the
  existing 12 warps hide more latency.
Hard rules:
- Do NOT touch LayerNorm/residual handling, the public interface signatures,
  or anything outside kernels/.
- Do NOT retry the known negative results: cp.async double-buffering
  (INT8_ATTN_DB) and V register prefetch (INT8_ATTN_VPREFETCH).
- Registers: the target is OCCUPANCY (blocks/SM), not a fixed register cap, and
  on this shape occupancy is already maxed — a registers-only change that does
  not also reduce shared memory will NOT add a block. No spills, and do not
  drop blocks/SM below the current baseline. Check with ptxas AND re-profile.
After editing, output these lines:
CHANGE: <files + what you changed>
TARGET: <which metric you expect to improve and why>

OPTIONAL — only if this change is a NEGATIVE RESULT you want to PRESERVE rather
than have reverted: an idea you tried that did NOT improve latency, kept solely
as an OFF-by-default ablation flag (new code fully behind an OFF #define so the
default path is unchanged) plus a documented note. Per CLAUDE.md §4 "record
negative results too (keep them behind ablation flags)". If and ONLY if that is
the case, ALSO append a one-line note to README.md explaining the negative
result, and output a third line:
NEGATIVE_RESULT: <one-line why it is worth keeping behind the OFF flag>
Do NOT output this line for a normal optimization attempt — a neutral
default-path change with no NEGATIVE_RESULT line is reverted as a no-op.
EOF
    CHANGE_DESC=$(grep -E "^CHANGE:" "$LOG_DIR/claude_change_$i.log" | tail -1)
    TARGET_DESC=$(grep -E "^TARGET:" "$LOG_DIR/claude_change_$i.log" | tail -1)
    NEG_DESC=$(grep -E "^NEGATIVE_RESULT:" "$LOG_DIR/claude_change_$i.log" | tail -1)
    if [ -z "$(git status --porcelain kernels/)" ]; then
        log "claude made no kernel change — skipping iteration"
        ITER_RESULTS[$i]="SKIP (no change)"; continue
    fi

    # c. Compile check, up to 3 fix attempts.
    BUILD_OK=0
    for attempt in 1 2 3; do
        log "compile check (attempt $attempt) ..."
        if ERR=$(compile_check 2>&1); then BUILD_OK=1; echo "$ERR"; break; fi
        echo "$ERR" | tail -20
        run_claude "$LOG_DIR/claude_fixbuild_${i}_$attempt.log" <<EOF
The kernel change you just made fails to compile. Fix the compile error in
kernels/ without changing the optimization's intent. Error output:

$(echo "$ERR" | tail -40)
EOF
    done
    if [ "$BUILD_OK" -ne 1 ]; then
        log "compile still failing after 3 attempts — restoring"
        restore_tree; ITER_RESULTS[$i]="FAIL (compile)"; continue
    fi
    rm -rf ~/.cache/torch_extensions/py312_cu128/int8_ext 2>/dev/null

    # d. Five-gate accuracy validation, up to 2 repair attempts.
    VAL_OK=0
    for attempt in 0 1 2; do
        log "validate_int8.py (attempt $attempt) ..."
        if python3 validate_int8.py > "$LOG_DIR/validate_${i}_$attempt.log" 2>&1
        then VAL_OK=1; break; fi
        [ "$attempt" -eq 2 ] && break
        run_claude "$LOG_DIR/claude_fixval_${i}_$attempt.log" <<EOF
Your kernel change broke the 5-gate INT8 validation. Follow the repair order
in CLAUDE.md §5 (per-channel quantization -> QK^T back to fp16 -> adjust
scale computation -> worst stage back to fp16) or revert the precision-
affecting part of your change while keeping the perf part. Only touch
kernels/. Validation output (failures + repair suggestions):

$(grep -E "FAIL|XFAIL|Gate|RECOVERS|DATASET" "$LOG_DIR/validate_${i}_$attempt.log" | head -60)
EOF
        compile_check >/dev/null 2>&1 || { VAL_OK=0; break; }
        rm -rf ~/.cache/torch_extensions/py312_cu128/int8_ext 2>/dev/null
    done
    if [ "$VAL_OK" -ne 1 ]; then
        log "validation failing — restoring"
        restore_tree
        ITER_RESULTS[$i]="FAIL (accuracy gates)"; continue
    fi

    # e. Perf gate (median of SWEEP_REPS sweeps). 3-way per kernel:
    #    regression (>PERF_REGRESS%) -> revert; latency-neutral no-op -> revert;
    #    commit ONLY if at least one kernel genuinely improves (<=PERF_IMPROVE%)
    #    and neither regresses. This is what stops the loop committing noise.
    log "perf sweep (median of $SWEEP_REPS) ..."
    median_sweep int8_attn "$LOG_DIR/sweep_attn_$i"
    median_sweep int8_mlp  "$LOG_DIR/sweep_mlp_$i"
    ATTN_DELTA=$(compare_sweep /tmp/baseline_attn.csv results/int8_attn_sweep.csv); ATTN_CODE=$?
    MLP_DELTA=$(compare_sweep /tmp/baseline_mlp.csv results/int8_mlp_sweep.csv);   MLP_CODE=$?
    log "latency vs baseline: attn $ATTN_DELTA, mlp $MLP_DELTA"
    if [ "$ATTN_CODE" -eq 2 ] || [ "$MLP_CODE" -eq 2 ]; then
        log "latency regressed >${PERF_REGRESS}% — restoring"
        restore_tree
        ITER_RESULTS[$i]="FAIL (perf regression: attn $ATTN_DELTA mlp $MLP_DELTA)"
        continue
    fi
    if [ "$ATTN_CODE" -ne 0 ] && [ "$MLP_CODE" -ne 0 ]; then
        # Latency-neutral. Keep ONLY as a documented negative result, via either:
        #  (1) claude declared NEGATIVE_RESULT upfront (change already behind an
        #      OFF flag, default path neutral BY DESIGN), or
        #  (2) the change MOVED a bottleneck metric beyond noise despite neutral
        #      latency — re-profile to confirm, then give claude ONE chance to
        #      gate it behind an OFF flag + document it. This is exactly the
        #      "-24% smem but no speedup proves the warp-count ceiling, not smem
        #      traffic, is the bottleneck" kind of insight that must not be lost
        #      to a silent git-reset.
        if [ -z "$NEG_DESC" ]; then
            log "latency-neutral — re-profiling to check whether a bottleneck metric moved ..."
            NCU_AFTER="$LOG_DIR/ncu_${i}_after.txt"
            if run_ncu "$NCU_AFTER" && MOVE=$(metric_moved "$NCU_OUT" "$NCU_AFTER"); then
                log "bottleneck moved despite neutral latency ($MOVE) — offering negative-result keep"
                run_claude "$LOG_DIR/claude_negkeep_$i.log" <<EOF
Your change is latency-neutral (median sweep: attn $ATTN_DELTA, mlp $MLP_DELTA)
but it MOVED a bottleneck metric beyond run-to-run noise: $MOVE. Per CLAUDE.md
§4 that is a valuable NEGATIVE RESULT — a metric moved without translating to a
speedup, which RULES OUT a direction and must be recorded, not silently
reverted.

If (and only if) it is worth keeping: put your ENTIRE change behind an
OFF-by-default \`#define\` so the default compiled path is byte-identical to
before — the ablation flag preserves the experiment without changing the
default — append a one-line note to README.md explaining the negative result
(what moved, what it rules out), and output exactly:
NEGATIVE_RESULT: <which metric moved, and what direction it rules out>
Only touch kernels/ and README.md. If you judge it not worth keeping, change
nothing and output nothing — it will be reverted as a no-op.
EOF
                NEG_DESC=$(grep -E "^NEGATIVE_RESULT:" "$LOG_DIR/claude_negkeep_$i.log" | tail -1)
                # The now-flag-gated default path must still build.
                if [ -n "$NEG_DESC" ] && ! compile_check >/dev/null 2>&1; then
                    log "negative-result refactor broke the build — discarding it"; NEG_DESC=""
                fi
            fi
        fi
        if [ -n "$NEG_DESC" ]; then
            # Preserve as a documented negative (behind an OFF flag). Do NOT update
            # the perf baseline (default unchanged) and do NOT count it as a PASS.
            log "committing as NEGATIVE RESULT (preserved behind OFF flag; not a speedup)"
            git add kernels/ README.md OPTIMIZATION_LOG.md CLAUDE.md 2>/dev/null
            git commit -m "evolve iter $i (negative result): ${CHANGE_DESC#CHANGE: }

${NEG_DESC}
Kept behind an OFF-by-default ablation flag; default-path latency neutral
(attn ${ATTN_DELTA}, mlp ${MLP_DELTA}). Recorded per CLAUDE.md §4.

Co-Authored-By: Claude (evolve.sh) <noreply@anthropic.com>" >/dev/null
            log "iteration $i NEGATIVE RESULT preserved ($(git rev-parse --short HEAD))"
            ITER_RESULTS[$i]="KEPT (negative result: ${NEG_DESC#NEGATIVE_RESULT: })"
            continue
        fi
        log "latency-neutral (within ~2% noise) and no bottleneck moved — reverting as no-op"
        restore_tree
        ITER_RESULTS[$i]="SKIP (no-op: attn $ATTN_DELTA mlp $MLP_DELTA — within noise)"
        continue
    fi

    # f. Record + commit. New baseline = this iteration's numbers.
    run_claude "$LOG_DIR/claude_record_$i.log" <<EOF
Append an iteration record to the bottom of OPTIMIZATION_LOG.md following the
CLAUDE.md §7 format exactly (next iteration number after the existing ones,
today's date). Facts:
- $CHANGE_DESC
- $TARGET_DESC
- Latency vs previous baseline: int8_attn $ATTN_DELTA, int8_mlp $MLP_DELTA
- All five validation gates passed (see below for gate-1 numbers)
$(grep -E "Gate1|SUMMARY|PASS" "$LOG_DIR/validate_${i}_0.log" | head -25)
Mark all five gate checkboxes as checked. Conclusion: pass. Only edit OPTIMIZATION_LOG.md.
EOF
    cp results/int8_attn_sweep.csv /tmp/baseline_attn.csv
    cp results/int8_mlp_sweep.csv  /tmp/baseline_mlp.csv
    git add kernels/ results/ OPTIMIZATION_LOG.md
    git commit -m "evolve iter $i: ${CHANGE_DESC#CHANGE: }

${TARGET_DESC}
Latency vs previous baseline: int8_attn ${ATTN_DELTA}, int8_mlp ${MLP_DELTA}
All 5 validation gates passed.

Co-Authored-By: Claude (evolve.sh) <noreply@anthropic.com>" >/dev/null
    log "iteration $i COMMITTED ($(git rev-parse --short HEAD))"
    PASS_COUNT=$((PASS_COUNT + 1))
    ITER_RESULTS[$i]="PASS (attn $ATTN_DELTA, mlp $MLP_DELTA)"
done

# ════════════════════════════════════════════════════════════════════════
# 3. Summary
# ════════════════════════════════════════════════════════════════════════
echo; log "════════ SUMMARY ════════"
for i in $(seq 1 "$MAX_ITER"); do
    echo "  iter $i: ${ITER_RESULTS[$i]:-not run}"
done
echo "  passed: $PASS_COUNT / $MAX_ITER"

median_sweep int8_attn "$LOG_DIR/sweep_attn_final" || true
median_sweep int8_mlp  "$LOG_DIR/sweep_mlp_final"  || true
TOTAL_ATTN=$(compare_sweep /tmp/baseline_attn.csv results/int8_attn_sweep.csv || true)
TOTAL_MLP=$(compare_sweep /tmp/baseline_mlp.csv results/int8_mlp_sweep.csv || true)
git checkout -- results/ 2>/dev/null
echo "  final latency vs last-accepted baseline: attn $TOTAL_ATTN, mlp $TOTAL_MLP"

log "asking claude for the next recommended direction ..."
run_claude "$LOG_DIR/claude_summary.log" <<EOF
Read the Iteration Log in OPTIMIZATION_LOG.md and the latest ncu output:
$(tail -30 "$LOG_DIR/ncu_$MAX_ITER.txt" 2>/dev/null)
In 5 lines or fewer, state the single most promising next optimization
direction and why, consistent with CLAUDE.md §4. Do not edit any files.
EOF
