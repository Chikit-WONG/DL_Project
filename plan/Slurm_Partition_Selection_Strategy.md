# Slurm Partition Selection Strategy

## Goal

Choose partitions by balancing three factors:

1. resource fit
2. queue speed
3. cost / priority level

The default rule is:

- use `debug` first when the job fits its free limits and the queue is not materially slower than other suitable partitions
- otherwise use the cheapest and lowest-priority partition that can finish the job
- within the paid partitions, search in this order: shared low -> exclusive medium -> emergency high
- only move to a higher-priority partition when the lower-priority queue is clearly slower
- do not use `long_*` by default unless the job truly needs the longer runtime or `long_*` is currently faster

Official queue reference:

- https://docs.hpc.hkust-gz.edu.cn/docs/hpc12/slurm/queue/

## Official HPC2 Partition Facts

The official queue page for HPC2 groups partitions by priority and price:

- shared = low priority, low price
- exclusive = medium priority, medium price
- emergency = high priority, high price
- debug = free, short debug / short formal jobs only

### CPU pool

| Partition | Priority / Price | User resource limit | Default time | Node spec |
|---|---|---:|---|---|
| `i64m512u` | shared / low | 1024 CPU cores | 7 days | 2x Intel 8358P, 512GB |
| `i64m512ue` | exclusive / medium | 1024 CPU cores | 7 days | 2x Intel 8358P, 512GB |
| `emergency_cpu` | emergency / high | 512 CPU cores | 7 days | same CPU pool |
| `long_cpu` | shared / low | 1024 CPU cores | 14 days | same CPU pool |
| `i64m512r` | shared / low | 128 CPU cores | 7 days | 2x Intel 8358P, 512GB, 6x1.92TB |
| `i64m512re` | exclusive / medium | 128 CPU cores | 7 days | same node class |
| `a128m512u` | shared / low | 256 CPU cores | 7 days | 2x AMD EPYC 7763, 512GB |
| `a128m512ue` | exclusive / medium | 128 CPU cores | 7 days | same node class |

### GPU pool

| Partition | Priority / Price | User resource limit | Default time | Node spec |
|---|---|---:|---|---|
| `i64m1tga800u` | shared / low | 128 CPU cores, 16 GPUs | 7 days | 2x Intel 8358P, 8x A800 80GB, 1024GB |
| `i64m1tga800ue` | exclusive / medium | 64 CPU cores, 8 GPUs | 7 days | same node class |
| `emergency_gpu` | emergency / high | 64 CPU cores, 8 GPUs | 7 days | same node class |
| `long_gpu` | shared / low | 128 CPU cores, 16 GPUs | 14 days | same node class |
| `i64m1tga40u` | shared / low | 128 CPU cores, 16 GPUs | 7 days | 2x Intel 8358P, 8x A40 48GB, 1024GB |
| `i64m1tga40ue` | exclusive / medium | 64 CPU cores, 8 GPUs | 7 days | same node class |
| `emergency_gpua40` | emergency / high | 64 CPU cores, 8 GPUs | 7 days | same node class |

### Large-memory pool

| Partition | Priority / Price | User resource limit | Default time | Node spec |
|---|---|---:|---|---|
| `i96m3tu` | shared / low | 192 CPU cores | 7 days | 4x Intel 6348H, 3TB |
| `i96m3tue` | exclusive / medium | 192 CPU cores | 7 days | same node class |

### Debug

| Partition | Priority / Price | User resource limit | Default time | Node spec |
|---|---|---:|---|---|
| `debug` | free | 16 CPU cores, 2 GPUs | 0.5 hour | mixed CPU/A40 debug pool |

## Selection Order

### 1. Match the resource first

- Before choosing any non-`debug` partition, check whether the job can finish inside the `debug` limits:
  - free
  - up to `2` A40 GPUs
  - up to `30` minutes
  - valid for both short GPU jobs and short CPU jobs
- CPU-only post-processing, metrics merge, summary generation:
  use CPU partitions, not GPU partitions
- GPU jobs that fit on A40:
  prefer A40 partitions over A800 partitions
- Use A800 only when:
  - the job actually needs A800 memory / throughput
  - or A800 partitions are materially faster to start than A40 partitions

### 2. Within the same resource class, search from lower to higher priority

#### CPU

- first check: `debug` if the CPU job fits the debug limits
- then check shared low: `i64m512u`, `a128m512u`, `i64m512r`
- then check exclusive medium: `i64m512ue`, `a128m512ue`, `i64m512re`
- then check emergency high: `emergency_cpu`
- only consider `long_cpu` when:
  - the job may exceed 7 days
  - or `long_cpu` is currently less congested than the normal CPU queues

#### GPU A40

- first check: `debug` if the GPU job fits the debug limits
- then check shared low: `i64m1tga40u`
- then check exclusive medium: `i64m1tga40ue`
- then check emergency high: `emergency_gpua40`

#### GPU A800

- first check shared low: `i64m1tga800u`, `i64m1tga800u8ka`
- then check exclusive medium: `i64m1tga800ue`
- then check emergency high: `emergency_gpu`
- only consider `long_gpu` when:
  - the job may exceed 7 days
  - or `long_gpu` is currently less congested than the normal A800 queues

#### Large-memory jobs

- first check shared low: `i96m3tu`
- then check exclusive medium: `i96m3tue`
- only move away from the shared tier when the queue gap is clearly worth the higher price

## Practical Decision Rules

Before submitting or resubmitting a job:

1. check official partition capability and max runtime from the queue documentation
2. check real-time backlog with `squeue`, `sinfo`, and partition-specific pending/running counts
3. if the job fits `debug`, prefer `debug` unless it is clearly more congested than another suitable partition
4. if the lower-priority partition is only slightly slower, stay on the lower-priority partition
5. if the higher-priority partition is clearly less congested, upgrade
6. if a job is already close to starting or already running, do not resubmit just to chase a marginally better queue

## QOS Notes

QOS means the scheduler policy layer that can impose limits such as:

- max wall time
- max jobs per user
- max submitted jobs per user
- max CPUs / GPUs per user

These limits are not unique to `debug`. Other partitions also have QOS-related limits, usually expressed through entries like `MaxTRESPU`, `MaxJobsPU`, or `MaxSubmitPU`.

Observed examples from the live cluster configuration:

- `debug`: `MaxTRESPU=cpu=16,gres/gpu=2`, `MaxJobsPU=10`, `MaxSubmitPU=8`
- `i64m1tga40u`: `MaxTRESPU=cpu=128,gres/gpu:a40=16,gres/gpu=16`
- `emergency_gpua40`: `MaxTRESPU=cpu=64,gres/gpu:a40=8,gres/gpu=8`
- `i64m1tga800u`: `MaxTRESPU=cpu=128,gres/gpu:a800=16,gres/gpu=16`
- `emergency_gpu`: `MaxTRESPU=cpu=64,gres/gpu:a800=8,gres/gpu=8`
- `i64m512u`: `MaxTRESPU=cpu=1024`
- `emergency_cpu`: `MaxTRESPU=cpu=512`

### `debug` in particular

`debug` is free and should be checked first for short jobs, but it has practical constraints:

- partition limit: `30` minutes
- resource limit: at most `2` A40 GPUs
- observed QOS limit for one user:
  - `MaxTRESPU=cpu=16,gres/gpu=2`
  - `MaxJobsPU=10`
  - `MaxSubmitPU=8`
  - `Flags=DenyOnLimit`

This means:

- a short job can fit inside `debug` on paper but still queue because your user already occupies the `debug` CPU/GPU quota
- CPU-only `debug` jobs can also queue behind your own other `debug` jobs if your per-user CPU quota is full
- `debug` should therefore be treated as:
  - first choice when it fits
  - but not assumed to support unlimited parallelism

## Job-Type Recommendations

### Smoke test / env check

- use `debug`
- keep the script short and bounded
- `debug` is also the first choice for short formal jobs when the same limits are satisfied and the live queue is acceptable

### Short GPU jobs on A40

- if the formal job is expected to fit within the `debug` limits, try `debug` first
- if `debug` is more congested than another suitable partition, switch based on the live queue state
- otherwise, for larger or slower A40 jobs:
  - try `i64m1tga40u` first
  - if backlog is clearly better, move to `emergency_gpua40`

Examples:

- cache rebuild
- light training stages
- small eval jobs
- short formal reruns that really can finish inside the `debug` limits

### Short CPU jobs

- if the job fits inside the `debug` limits, try `debug` first because it is free
- if `debug` is materially slower or blocked by user QOS limits, try shared low first: `i64m512u`, `a128m512u`, `i64m512r`
- consider exclusive medium only when shared low is noticeably slower and the added cost is justified
- move to `emergency_cpu` only when lower tiers are slower enough to justify it

Examples:

- summarizing results
- merging JSON / CSV
- making tables
- lightweight file validation or output packaging

### Long A800 jobs

- examples: multi-stage retrieval training, SDXL-based generation pipelines
- try `i64m1tga800u` or `i64m1tga800u8ka` first if they are not badly congested
- if the medium exclusive tier is meaningfully better, consider `i64m1tga800ue`
- if low and medium are much more congested, move to `emergency_gpu`
- avoid `long_gpu` unless runtime or queue speed justifies it

## Current Example From This Project

Using the current queue snapshot:

- short formal jobs that truly fit the `debug` limits should check `debug` first, but still need a live check because `debug` can queue behind per-user QOS limits
- `version1` and `version3_ATM` fit on A40, so `emergency_gpua40` is preferable when it is less congested than `i64m1tga40u`
- `version4_CCP` retrieval / alignment / generation is better treated as A800 work, and the current choice between `long_gpu` and `emergency_gpu` must weigh:
  - `long_gpu` = lower price, lower priority
  - `emergency_gpu` = higher price, higher priority
  - if `long_gpu` is only slightly less busy, the higher priority of `emergency_gpu` can still make it the safer choice
- `version4_CCP` summary is CPU-only, so a lower-priority CPU queue such as `a128m512u` is preferable when it is at least as empty as `emergency_cpu`

## Commands To Check Before Deciding

```bash
sinfo -o '%20P %8a %12l %6D %10t %20G'
```

```bash
squeue -h -t PENDING -o '%P' | sort | uniq -c | sort -k2,2
```

```bash
for p in i64m1tga40u emergency_gpua40 i64m1tga800u i64m1tga800u8ka emergency_gpu long_gpu i64m512u a128m512u emergency_cpu long_cpu; do
  printf '%-18s pending=%-3s running=%-3s\n' "$p" "$(squeue -h -p "$p" -t PENDING | wc -l)" "$(squeue -h -p "$p" -t RUNNING | wc -l)"
done
```
