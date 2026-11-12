"""Run a small, reproducible physics-based coordination experiment on CPU."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import ogbench
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle


def emit(kind, **data):
    print(json.dumps({"kind": kind, **data}, allow_nan=False), flush=True)


def cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-15 else None


def state_embedding(info, cube, target):
    """An explicitly engineered state vector, not a neural embedding."""
    return np.concatenate([
        info["proprio/joint_pos"], info["proprio/effector_pos"],
        info["proprio/gripper_opening"], info["proprio/gripper_contact"],
        cube, target,
    ]).astype(np.float64)


def target_for_oracle(env, info):
    base = env.unwrapped
    goals = base.cur_task_info["goal_xyzs"]
    starts = base.cur_task_info["init_xyzs"]
    idx = int(np.argmax(np.linalg.norm(goals - starts, axis=1)))
    out = dict(info)
    out["privileged/target_block"] = idx
    out["privileged/target_block_pos"] = goals[idx].copy()
    out["privileged/target_block_yaw"] = np.array([0.0])
    return out, idx, goals[idx].copy()


def rollout(env, cfg, seed, fault, method, trace=False):
    np.random.seed(seed)
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed, options={"render_goal": False})
    info, idx, target = target_for_oracle(env, info)
    initial_hash = hashlib.sha256(np.asarray(info["qpos"], dtype='<f8').tobytes()).hexdigest()
    controllers = [CubeMarkovOracle(env=env), CubeMarkovOracle(env=env)]
    for controller in controllers:
        np.random.seed(seed + 100000)
        controller.reset(obs, info)

    cache = copy.deepcopy(info)
    stamp = 0
    active = 0
    switches = 0
    missing = 0
    refresh_attempts = refresh_successes = 0
    best_distance = float('inf')
    last_progress = 0
    evidence = []
    hidden_bad_steps = 0
    inaccurate_steps = 0
    max_pose_error = 0.0
    success = False
    start = time.monotonic()
    key = f"privileged/block_{idx}_pos"
    yaw_key = f"privileged/block_{idx}_yaw"

    for step in range(cfg["max_steps"]):
        fault_active = step >= cfg["fault_start"]
        publisher_failed = fault_active and fault in {"shared_stale", "shared_blind", "forged_fresh"}
        if not publisher_failed:
            cache = copy.deepcopy(info)
            stamp = step
        elif fault == "forged_fresh":
            # Cached content remains unchanged but its source incorrectly claims a new timestamp.
            stamp = step
        age = step - stamp

        wants_refresh = (
            method == "freshness" and age >= cfg["freshness_ttl"]
        ) or (
            method == "periodic" and step > 0 and step % cfg["freshness_ttl"] == 0
        )
        if wants_refresh and refresh_attempts < cfg["refresh_budget"]:
            refresh_attempts += 1
            if not (fault_active and fault == "shared_blind"):
                cache = copy.deepcopy(info)
                stamp = step
                refresh_successes += 1

        seen = dict(info)
        # Controllers see fresh proprioception but only the shared object's cached pose.
        seen[key] = cache[key].copy()
        seen[yaw_key] = cache[yaw_key].copy()
        distance = float(np.linalg.norm(seen[key] - target))
        if distance < best_distance - 0.002:
            best_distance = distance
            last_progress = step
        primary_missing = fault_active and fault == "primary_dropout" and active == 0
        missing = missing + 1 if primary_missing else 0
        if method != "single" and active == 0 and (
            missing >= cfg["heartbeat_timeout"] or step - last_progress >= cfg["stagnation_timeout"]
        ):
            active = 1
            switches += 1
            primary_missing = False

        # Diagnostic-only truth: never passed to the selection logic above.
        error = float(np.linalg.norm(cache[key] - info[key]))
        full_cos = cosine(state_embedding(info, cache[key], target), state_embedding(info, info[key], target))
        pose_cos = cosine(cache[key], info[key])
        max_pose_error = max(max_pose_error, error)
        inaccurate_steps += int(error > 0.04)
        hidden_bad_steps += int(error > 0.04 and full_cos is not None and full_cos > 0.99)

        if primary_missing:
            action = np.zeros(env.action_space.shape, dtype=np.float64)
        else:
            action = controllers[active].select_action(obs, seen)
        assert np.isfinite(action).all(), "non-finite action"
        assert np.max(np.abs(action)) <= 1.00001, "action outside normalized bounds"

        if trace and step % 3 == 0:
            evidence.append({
                "step": step, "time_s": step * float(env.unwrapped._control_timestep),
                "effector": info["proprio/effector_pos"].tolist(),
                "cube": info[key].tolist(), "cached_cube": cache[key].tolist(),
                "target": target.tolist(), "active": active, "age": step-stamp,
                "cosine": full_cos, "pose_cosine": pose_cos, "pose_error_m": error,
                "gripper": float(info["proprio/gripper_opening"][0]),
            })

        obs, reward, terminated, truncated, next_info = env.step(action)
        info, _, _ = target_for_oracle(env, next_info)
        success = bool(info.get("success", False))
        if success or terminated or truncated:
            break

    row = {
        "seed": seed, "fault": fault, "method": method, "success": success,
        "steps": step+1, "sim_time_s": (step+1)*float(env.unwrapped._control_timestep),
        "wall_time_s": time.monotonic()-start, "switches": switches,
        "refresh_attempts": refresh_attempts, "refresh_successes": refresh_successes,
        "final_goal_error_m": float(np.linalg.norm(info[key]-target)),
        "max_cache_error_m": max_pose_error, "inaccurate_steps": inaccurate_steps,
        "hidden_bad_steps": hidden_bad_steps, "initial_hash": initial_hash,
    }
    return row, evidence


def wilson(successes, n):
    z = 1.959963984540054
    p = successes/n
    d = 1+z*z/n
    mid = (p+z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,mid-half), min(1.,mid+half)]


def summarize(rows, cfg):
    groups = []
    for fault in cfg["faults"]:
        for method in cfg["methods"]:
            r = [x for x in rows if x["fault"]==fault and x["method"]==method]
            n = len(r)
            k = sum(x["success"] for x in r)
            groups.append({
                "fault": fault, "method": method, "n": n, "successes": k,
                "success_rate": k/n, "wilson95": wilson(k,n),
                "mean_steps": float(np.mean([x["steps"] for x in r])),
                "mean_switches": float(np.mean([x["switches"] for x in r])),
                "mean_refresh_attempts": float(np.mean([x["refresh_attempts"] for x in r])),
                "episodes_with_hidden_pose_error": sum(x["hidden_bad_steps"]>0 for x in r),
            })
    pairs = []
    rng = np.random.default_rng(80221)
    for fault in cfg["faults"]:
        base = {x["seed"]:int(x["success"]) for x in rows if x["fault"]==fault and x["method"]=="single"}
        for method in cfg["methods"]:
            if method == "single":
                continue
            vals = np.array([int(x["success"])-base[x["seed"]] for x in rows if x["fault"]==fault and x["method"]==method])
            if len(vals):
                samples = rng.choice(vals, (10000,len(vals)), replace=True).mean(1)
                pairs.append({"fault":fault,"method":method,"reference":"single","difference":float(vals.mean()),
                              "paired_bootstrap95":np.quantile(samples,[.025,.975]).tolist(),"n_pairs":len(vals)})
    return {"groups": groups, "paired_differences": pairs}


def main():
    cfg = json.loads(Path("config.json").read_text())
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    versions = {p:importlib.metadata.version(p) for p in ["ogbench","mujoco","numpy","gymnasium"]}
    metadata = {"config":cfg,"versions":versions,"python":platform.python_version(),
                "run_id":os.environ.get("ORX_RUN_ID"),"commit":os.environ.get("ORX_COMMIT"),
                "scope":"Scripted oracle, privileged state, single simulated arm; not a learned MoE or neural representation experiment."}
    emit("configuration", **metadata)
    env = ogbench.make_env_and_datasets(cfg["environment"], env_only=True, permute_blocks=False, visualize_info=False)
    rows = []
    traces = []
    hashes = {}
    began = time.monotonic()
    try:
        for fault in cfg["faults"]:
            for seed in cfg["seeds"]:
                for method in cfg["methods"]:
                    row, trace = rollout(env,cfg,seed,fault,method,seed in cfg["trace_seeds"])
                    if seed in hashes:
                        assert hashes[seed] == row["initial_hash"], "paired reset differs"
                    hashes[seed] = row["initial_hash"]
                    rows.append(row)
                    if trace:
                        traces.append({"seed":seed,"fault":fault,"method":method,"success":row["success"],"frames":trace})
                    emit("episode", **row)
                    (out/"episodes.json").write_text(json.dumps(rows,indent=2))

        checks = {"paired_initial_states": True,"finite_bounded_actions": True}
        if cfg["verify_replay"]:
            again, _ = rollout(env,cfg,cfg["seeds"][0],cfg["faults"][0],cfg["methods"][0])
            original = rows[0]
            keys = ["success","steps","initial_hash","final_goal_error_m","max_cache_error_m"]
            checks["deterministic_replay"] = all(again[k] == original[k] for k in keys)
            assert checks["deterministic_replay"], "repeat baseline was not deterministic"
        result = {**metadata, **summarize(rows,cfg),"integrity_checks":checks,"episodes":len(rows),
                  "total_wall_time_s":time.monotonic()-began}
        (out/"summary.json").write_text(json.dumps(result,indent=2))
        (out/"traces.json").write_text(json.dumps(traces,separators=(',',':')))
        emit("summary", **result)
    finally:
        env.close()


if __name__ == "__main__":
    main()
