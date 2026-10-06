"""Portable stdlib publication tool for the trusted complete v2 analysis."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_SCHEMA = ROOT / "docs/reproduction/evidence/optimization-v2-report-schema.json"
BASELINE, PRIMARY = "confidence_nooracle", "defect_guarded"
METRICS = {"accuracy_cm": "Accuracy ↓ / cm", "completion_cm": "Completion ↓ / cm",
           "coverage_percent": "2 cm 覆盖率 ↑ / %", "chamfer_mm": "Chamfer ↓ / mm"}
COSTS = {"mission_seconds": "任务 / s", "wall_seconds": "重建墙钟 / s", "planning_seconds": "规划 / s",
         "mapping_seconds": "建图 / s", "sensor_seconds": "传感器 / s", "diagnostic_seconds": "诊断 IO / s",
         "simulated_flight_seconds": "估算移动 / s", "path_length_m": "路径 / m", "observations": "观测数",
         "events": "事件数", "optimizer_steps": "优化步数", "additional_optimizer_steps": "前缀后优化步数",
         "peak_torch_allocated_mb": "Torch 峰值 / MiB", "meshing_evaluation_wall_seconds": "网格与评估 / s"}
POST = ("planning_seconds", "mapping_seconds", "sensor_seconds", "diagnostic_seconds", "events", "observations")
LABELS = {BASELINE: "Confidence（无候选真值）", "defect": "Defect v1（本轮开发对照）",
          PRIMARY: "Guarded v2（固定主方法）", "defect_guarded_no_gate": "Guarded v2 no gate（消融）"}
TITLES = {"development_observations": "开发种子 · 固定 60 次观测/更新", "development_time": "开发种子 · 180 秒任务预算",
          "heldout_observations": "留出种子 · 固定 60 次观测/更新", "heldout_time": "留出种子 · 180 秒任务预算"}
FALLBACKS = {"weight_zero": "β=0", "base_all_zero": "基础效用全零", "single_reachable": "仅一个可达候选",
             "weak_geometry": "几何信号不足", "constant_geometry": "几何项恒定"}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "必须是有限实数")
    return value


def count(value):
    require(type(value) is int and value >= 0, "必须是非负整数计数")
    return value


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "缺失或非法 SHA256")
    return value


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "JSON 含重复键")
            value[key] = item
        return value
    raw = Path(path).read_bytes()
    data = json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("JSON 含非有限常量")))
    return data, hashlib.sha256(raw).hexdigest()


def stats(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def check_stat(record, values):
    expected = stats(values)
    require(record.get("n") == expected["n"], "统计样本数不匹配")
    for key in ("mean", "sample_std"):
        if expected[key] is None:
            require(record.get(key) is None, "单样本 SD 应为 null")
        else:
            require(math.isclose(number(record.get(key)), expected[key], rel_tol=1e-10, abs_tol=1e-9), "统计未对应逐种子值")


def check_paired(record, values, reference, seeds):
    delta = [a - b for a, b in zip(values, reference)]
    check_stat(record, delta)
    require(record.get("method_minus_baseline_by_seed") == dict(zip(map(str, seeds), delta)), "配对差不是同 seed 方法减基线")
    require(record.get("negative_difference_seeds") == sum(d < 0 for d in delta)
            and record.get("positive_difference_seeds") == sum(d > 0 for d in delta), "配对差符号计数不匹配")
    expected = {str(seed): 100 * delta[i] / reference[i] if reference[i] else None for i, seed in enumerate(seeds)}
    require(record.get("relative_percent_by_seed") == expected, "相对百分比不匹配（覆盖率差仍须用 pp）")


def check_guarded(record, events, prefix):
    rows = record.get("events", [])
    require([row.get("event") for row in rows] == list(range(prefix + 1, events + 1)), "缺失完整 guarded 后缀诊断")
    require(count(record.get("validated_events")) == len(rows), "guarded 诊断计数不匹配")
    for row in rows:
        for key in ("geometry_signal_active", "geometry_discriminative", "geometry_active", "selection_changed"):
            require(type(row.get(key)) is bool, "诊断布尔标志缺失")
        require(row.get("fallback_reason") in (*FALLBACKS, "none"), "未知回退原因")
        require(row["geometry_active"] == (row["fallback_reason"] == "none"), "奖励激活和回退记录不一致")
        for key in ("baseline_regret", "bonus_cap", "geometry_peak", "utility_seconds"):
            require(number(row.get(key)) >= 0, "诊断值不得为负")
        require(row["baseline_regret"] <= row["bonus_cap"] + 3e-6, "regret 超过记录的奖励上限与浮点容差")
    pairs = {"signal_active_events": "geometry_signal_active", "discriminative_events": "geometry_discriminative",
             "reward_applied_events": "geometry_active", "selection_changed_events": "selection_changed"}
    for key, field in pairs.items():
        require(count(record.get(key)) == sum(row[field] for row in rows), "诊断激活/改选计数不匹配")
    expected = {reason: sum(row["fallback_reason"] == reason for row in rows) for reason in FALLBACKS}
    require(record.get("fallback_counts") == expected, "回退原因分布不匹配")
    for key, value in {"maximum_baseline_regret": max((r["baseline_regret"] for r in rows), default=0),
                       "baseline_regret_sum": math.fsum(r["baseline_regret"] for r in rows),
                       "utility_seconds": math.fsum(r["utility_seconds"] for r in rows)}.items():
        require(math.isclose(number(record.get(key)), value, rel_tol=1e-10, abs_tol=1e-9), "诊断汇总与逐事件值不符")


def validate(data, schema):
    require(data.get("version") == "viewmend-campaign-analysis-v2" and data.get("status") == "complete", "整链未完成：拒绝生成正式结果")
    require(data.get("campaign") == "optimization-v2" and data.get("scene") == "replica/office0", "不是本轮 v2 office0")
    require([data.get(k) for k in ("completed_campaign_stages", "validated_campaign_experiments", "quality_runs")] == [5, 27, 24], "必须完整验收五阶段、27分支，24正式分支")
    require(data.get("primary_method") == PRIMARY and data.get("development_seeds") == [0, 1]
            and data.get("heldout_seeds") == [3, 4, 5], "主方法或开发/留出种子被替换")
    require(data.get("process_observation_platform") == "linux", "需要实际 Linux 进程退出门控后的分析")
    require(data.get("campaign_spec_sha256") == schema["campaign_spec_sha256"], "不是冻结的 v2 campaign")
    frozen_source = schema.get("source_identity")
    source_fields = {"project_commit", "project_source_sha256", "upstream_commit",
                     "upstream_source_sha256", "upstream_diff_sha256"}
    require(isinstance(frozen_source, dict) and set(frozen_source) == source_fields
            and frozen_source["project_commit"] == schema["source_commit"], "发布 schema 缺少完整五项冻结源码指纹")
    require(data.get("campaign_project_source_identity") == frozen_source, "实验的五项源码指纹不是本轮冻结值")
    expected_units = {"accuracy_cm": "cm", "completion_cm": "cm", "coverage_percent": "% / paired pp", "chamfer_mm": "mm"}
    require(data.get("units") == expected_units, "指标单位不匹配，禁止二次换算")
    stages = data.get("stages", {})
    require(set(stages) == {s["name"] for s in schema["profile"]["stages"]}, "阶段缺失或混入其他实验")
    inputs = data.get("source_input_sha256", {})
    require(isinstance(inputs, dict) and inputs, "缺少 fresh 输入指纹")
    for name, identity in inputs.items():
        require(isinstance(name, str) and name, "非法输入路径")
        digest(identity)
    identities, total = [], 0
    for spec in schema["profile"]["stages"]:
        name, seeds, methods = spec["name"], spec["seeds"], spec["methods"]
        stage = stages[name]
        require(stage.get("status") == "completed_and_freshly_audited" and stage.get("fresh_artifact_audit") is True
                and stage.get("saved_paired_report_verified_against_current_files") is True, "存在未 fresh 审计阶段")
        require(stage.get("seeds") == seeds and stage.get("protocol") == schema["protocols"][name], "阶段种子/预算/完整 recipe 改变")
        require(stage.get("partition") == ("smoke" if name == "smoke" else name.split("_")[0]), "开发与留出边界改变")
        run_id = stage.get("run_id", "")
        require(re.fullmatch(re.escape(name) + r"-[0-9a-f]{32}", run_id), "未知阶段标识")
        for tail in ("benchmark-summary.json", "paired-results.json"):
            require(any(path.replace("\\", "/").endswith(f"/{run_id}/{tail}") for path in inputs), "缺少阶段来源指纹")
        require(stage.get("scene") == "replica/office0" and stage.get("recipe") == "optimization-v2", "场景/recipe 不一致")
        require(stage.get("recipe_sha256") == schema["protocols"][name]["recipe_sha256"]
                and stage.get("campaign_spec_sha256") == schema["campaign_spec_sha256"], "recipe 摘要不一致")
        for key in ("scene_mesh_sha256", "scene_assets_sha256", "context_hash"):
            digest(stage.get(key))
        identities.append({k: stage[k] for k in ("source_versions", "evaluation", "scene_mesh_sha256", "scene_assets_sha256")})
        rows = stage.get("per_run", [])
        expected = {(method, seed) for method in methods for seed in seeds}
        require(len(rows) == len(expected) and {(r.get("method"), r.get("seed")) for r in rows} == expected, "不完整或重复 method×seed")
        index = {(r["method"], r["seed"]): r for r in rows}
        require(set(stage.get("method_statistics", {})) == set(methods), "方法统计不是完整预注册对照")
        require(set(stage.get("paired_difference_vs_confidence", {})) == set(methods) - {BASELINE}, "配对方法缺失或被替换")
        families = {"metrics": tuple(METRICS), "cost": tuple(COSTS), "post_prefix_cost": POST}
        for row in rows:
            for family, keys in families.items():
                require(set(row.get(family, {})) == set(keys), "质量/成本字段不完整")
                for key in keys:
                    require(number(row[family][key]) >= 0, "质量/成本值无效")
            require(row["metrics"]["coverage_percent"] <= 100, "覆盖率超出百分比范围")
            digest(row.get("prefix_sha256")); digest(row.get("prefix_camera_sha256"))
            cost, branch = row["cost"], row["post_prefix_cost"]
            for key in ("events", "observations", "optimizer_steps", "additional_optimizer_steps"):
                count(cost[key])
            require(cost["events"] == cost["observations"] and cost["events"] >= spec["prefix"], "采集/事件数量不匹配")
            require(cost["optimizer_steps"] == 10 * cost["events"]
                    and cost["additional_optimizer_steps"] == 10 * (cost["events"] - spec["prefix"]),
                    "优化步数必须为每事件10步，前缀后优化步数不得包含公共前缀")
            require(branch["events"] == cost["events"] - spec["prefix"] and branch["observations"] == cost["observations"] - spec["prefix"], "后缀成本混入公共前缀")
            if spec["mode"] == "observations":
                require(cost["events"] == spec["frames"], "固定观测分支预算不完整")
            else:
                require(cost["mission_seconds"] >= spec["seconds"], "时间分支未到任务预算")
            suffix = f"/{run_id}/experiments/benchmark/replica/office0/{row['method']}/{row['seed']}/"
            for tail in ("protocol.json", "final_result.json", "run-summary.json", "steps.json", "exp_config.json"):
                require(any(path.replace("\\", "/").endswith(suffix + tail) for path in inputs), "缺少分支 fresh 输入指纹")
        for seed in seeds:
            require(len({(index[m, seed]["prefix_sha256"], index[m, seed]["prefix_camera_sha256"]) for m in methods}) == 1, "同 seed 比较未共享真实前缀/相机")
        for method in methods:
            for family, keys in families.items():
                for key in keys:
                    values = [index[method, seed][family][key] for seed in seeds]
                    check_stat(stage["method_statistics"][method][family][key], values)
                    if method != BASELINE:
                        reference = [index[BASELINE, seed][family][key] for seed in seeds]
                        check_paired(stage["paired_difference_vs_confidence"][method][family][key], values, reference, seeds)
        diagnostics = stage.get("diagnostics_per_run", [])
        require(len(diagnostics) == len(expected) and {(d.get("method"), d.get("seed")) for d in diagnostics} == expected, "诊断分支不完整")
        for item in diagnostics:
            if item["method"].startswith("defect_guarded"):
                check_guarded(item, index[item["method"], item["seed"]]["cost"]["events"], spec["prefix"])
        exports = [f"campaign-optimization-v2-{run_id}-{method}-s{seed}" for method in methods for seed in seeds]
        require(sorted(stage.get("verified_exports", [])) == sorted(exports), "27个真实导出未全部验证")
        total += len(rows)
    require(total == 27 and all(identity == identities[0] for identity in identities), "整链跨源/资产/评估混合")
    return data


def guarded_summary(stage, method):
    rows = [r for d in stage["diagnostics_per_run"] if d["method"] == method for r in d["events"]]
    n = len(rows)
    return {"validated_events": n, "reward_applied_events": sum(r["geometry_active"] for r in rows),
            "signal_active_events": sum(r["geometry_signal_active"] for r in rows),
            "selection_changed_events": sum(r["selection_changed"] for r in rows),
            "fallback_counts": {reason: sum(r["fallback_reason"] == reason for r in rows) for reason in FALLBACKS},
            "event_weighted_mean_baseline_regret": math.fsum(r["baseline_regret"] for r in rows) / n if n else None,
            "maximum_baseline_regret": max((r["baseline_regret"] for r in rows), default=None),
            "bonus_cap_range": [min(r["bonus_cap"] for r in rows), max(r["bonus_cap"] for r in rows)] if n else [],
            "maximum_regret_over_cap": max((r["baseline_regret"] / r["bonus_cap"] for r in rows if r["bonus_cap"]), default=None),
            "utility_seconds_total": math.fsum(r["utility_seconds"] for r in rows)}


def snapshot(data, analysis_sha, schema_sha):
    public = copy.deepcopy(data)
    public.update(version="viewmend-v2-summary-v1", evidence={"analysis_sha256": analysis_sha, "schema_sha256": schema_sha,
                  "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "source_input_count": len(data["source_input_sha256"]),
                  "source_input_manifest_sha256": hashlib.sha256(json.dumps(data["source_input_sha256"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    public.pop("source_input_sha256")
    for stage in public["stages"].values():
        stage["guarded_signal_statistics"] = {m: guarded_summary(stage, m) for m in stage["method_statistics"] if m.startswith("defect_guarded")}
        for diagnostic in stage["diagnostics_per_run"]:
            diagnostic.pop("events", None)
    return public


def fmt(value, signed=False, digits=5):
    return "未定义" if value is None else format(value, ("+" if signed else "") + f".{digits}f")


def formatted_stat(value, signed=False, digits=3):
    sd = fmt(value["sample_std"], digits=digits)
    return f"{fmt(value['mean'], signed, digits)} ± {sd}"


def table(headers, rows):
    return ["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows], ""]


def render(data):
    lines = ["# optimization-v2 完整结果（待发布稿）", "", f"最终分析时间：{data['generated_at']}。五阶段27分支完成并通过 fresh 审计；其中24条用于正式质量统计。",
             "主方法按预注册固定为 **Guarded v2（defect_guarded）**。不按开发消融结果改换主方法；以下只报告实际差异，不作显著性或跨场景泛化结论。", "",
             "## 协议与解释", "", "四个正式阶段独立展示。开发种子0/1（n=2）与留出种子3/4/5（n=3）不合并；观测预算与时间预算也不合并。均值±样本标准差采用 n−1，既不是置信区间，也不是显著性检验。",
             "Accuracy/Completion 已是 cm，Chamfer 已是 mm，覆盖率已是 %，不再次换算。配对差一律同 seed 方法减 Confidence；距离负值、覆盖率正值方向更好，覆盖率差用百分点 pp。",
             "所有质量值来自相同评估配置。公共前缀真实地图、采集与 RNG 按 seed 配对；候选未来观测与掩码关闭，参考网格只用于评估，公开场景包围盒为共同先验。", "",
             "## 固定主方法的描述性结果", ""]
    for name in TITLES:
        stage = data["stages"][name]
        pair = stage["paired_difference_vs_confidence"][PRIMARY]["metrics"]
        desc = "；".join(f"{METRICS[k].split(' / ')[0]} Δ {fmt(pair[k]['mean'], True)} {'pp' if k == 'coverage_percent' else data['units'][k]}" for k in METRICS)
        lines.append(f"- {TITLES[name]}（n={len(stage['seeds'])}）：{desc}。Completion 更低 {pair['completion_cm']['negative_difference_seeds']}/{len(stage['seeds'])} 个 seed，覆盖率更高 {pair['coverage_percent']['positive_difference_seeds']}/{len(stage['seeds'])} 个 seed。")
    lines += ["", "即使部分指标均值方向有利，也需同时查看每个 seed、其他质量指标与采集/耗时权衡；低基线代理 regret 不是重建质量保证。", ""]
    for name, title in TITLES.items():
        stage = data["stages"][name]
        methods, seeds = list(stage["method_statistics"]), stage["seeds"]
        ordered = sorted(stage["per_run"], key=lambda r: (methods.index(r["method"]), r["seed"]))
        lines += [f"## {title}", "", f"实际批次：`{stage['run_id']}`；种子 {seeds}，n={len(seeds)}。前20次采集为同 seed 的公共前缀。", "",
                  "时间协议累积同步规划＋建图＋估算移动，完整事件跨过180秒后才停止，最终观测数可不同。" if stage["protocol"]["mode"] == "time" else "所有方法最终60次观测与60次更新；消融只在开发观测阶段执行。", "", "### 质量均值与样本 SD", ""]
        lines += table(["方法", *METRICS.values()], [[LABELS[m], *[formatted_stat(stage["method_statistics"][m]["metrics"][k]) for k in METRICS]] for m in methods])
        lines += ["### 逐种子质量", ""]
        lines += table(["方法", "seed", "观测", "事件", *METRICS.values()], [[LABELS[r["method"]], r["seed"], r["cost"]["observations"], r["cost"]["events"], *[fmt(r["metrics"][k], digits=6) for k in METRICS]] for r in ordered])
        lines += ["### 同 seed 配对差（方法 − Confidence）", "", "距离差为 cm/mm；覆盖率差为 pp。配对差的样本 SD 单独计算。", ""]
        for method, comparison in stage["paired_difference_vs_confidence"].items():
            lines += [f"**{LABELS[method]}**", ""]
            lines += table(["指标", *[f"seed {s} Δ" for s in seeds], "配对差均值 ± SD"],
                           [[k + (" / pp" if k == "coverage_percent" else ""), *[fmt(comparison["metrics"][k]["method_minus_baseline_by_seed"][str(s)], True) for s in seeds], formatted_stat(comparison["metrics"][k], True, 5)] for k in METRICS])
        lines += ["### 成本均值与样本 SD", ""]
        for keys in (tuple(COSTS)[:7], tuple(COSTS)[7:]):
            lines += table(["方法", *[COSTS[k] for k in keys]], [[LABELS[m], *[formatted_stat(stage["method_statistics"][m]["cost"][k], digits=2) for k in keys]] for m in methods])
        lines += ["### 逐种子成本", ""]
        for keys in (tuple(COSTS)[:7], tuple(COSTS)[7:]):
            lines += table(["方法", "seed", *[COSTS[k] for k in keys]], [[LABELS[r["method"]], r["seed"], *[fmt(r["cost"][k], digits=3) for k in keys]] for r in ordered])
        lines += ["### 前缀后成本及配对差", "", "以下不包含公共前缀；逐事件诊断 IO 的完整计时使用 run-summary 字段。", ""]
        lines += table(["方法", *POST], [[LABELS[m], *[formatted_stat(stage["method_statistics"][m]["post_prefix_cost"][k], digits=3) for k in POST]] for m in methods])
        for method, comparison in stage["paired_difference_vs_confidence"].items():
            lines += [f"**{LABELS[method]}的成本配对差**（方法 − Confidence）", ""]
            lines += table(["成本指标", *[f"seed {s} Δ" for s in seeds], "配对差均值 ± SD"],
                           [[COSTS[k], *[fmt(comparison["cost"][k]["method_minus_baseline_by_seed"][str(s)], True, 3) for s in seeds], formatted_stat(comparison["cost"][k], True, 3)] for k in COSTS])
        lines += ["### 当前地图信号、奖励与改选", "", "诊断只统计各分支前缀后的实际候选组。selection_changed 比较该方法自身当前地图上同一次候选集的 S0 首选与 S2 首选，不是与独立 Confidence 分支的实际轨迹逐事件比较。各方法改选后地图与候选可能不同。事件加权均值不等于 seed 均值；它只描述评分行为，不是质量样本数。", ""]
        for method, signal in stage["guarded_signal_statistics"].items():
            lines += [f"**{LABELS[method]}**：测得 {signal['validated_events']} 个事件，信号有效 {signal['signal_active_events']}，实际奖励应用 {signal['reward_applied_events']}，相对同次候选集、同一当前地图上的 S0 首选改选 {signal['selection_changed_events']}。",
                      f"事件加权平均 baseline-score regret={fmt(signal['event_weighted_mean_baseline_regret'])}，最大值={fmt(signal['maximum_baseline_regret'])}；奖励上限范围={signal['bonus_cap_range']}，最大 regret/cap={fmt(signal['maximum_regret_over_cap'])}（实现允许记录的浮点容差）。", ""]
            lines += table(["回退原因", "事件数"], [[FALLBACKS[k], v] for k, v in signal["fallback_counts"].items()])
        lines += ["逐 seed 的奖励应用、同次候选集 S0→S2 改选与 regret：", ""]
        lines += table(["方法", "seed", "诊断事件", "奖励应用", "同次候选集 S0→S2 改选", "平均 regret", "最大 regret", "utility s"],
                       [[LABELS[d["method"]], d["seed"], d["validated_events"], d["reward_applied_events"], d["selection_changed_events"],
                         fmt(d["baseline_regret_sum"] / d["validated_events"] if d["validated_events"] else None), fmt(d["maximum_baseline_regret"]), fmt(d["utility_seconds"], digits=3)]
                        for d in stage["diagnostics_per_run"] if d["method"].startswith("defect_guarded")])
    lines += ["## 成本、消融与局限", "", "任务时间、重建墙钟和完整流水线时间不是同一计量；sensor/诊断 IO 单列，网格评估另计，导出不包含在重建墙钟中。Torch 峰值只统计 PyTorch 分配，未包含 Habitat/OpenGL，也不是最低设备显存要求。更少观测伴随更低时间或峰值，不能说明等质量加速。",
              "Guarded no gate 和旧 Defect 只在开发观测阶段提供机制对照，不能按消融赢家替换已固定主方法，也不补造时间或留出消融。β=0日志反事实不是一次完整重建运行；有界 regret 只约束代理分数。",
              "种子留出仍是同一个 office0；开发 n=2、留出 n=3 属于小样本描述统计。未测试多场景、跨硬件泛化或局部残差与真值缺陷的因果关联。自定义 CUDA kernel 可能有数值非确定性。", "",
              "## 来源与验收", "", f"实验冻结源码：`{data['campaign_project_source_identity']['project_commit']}`；最终分析 SHA256：`{data['evidence']['analysis_sha256']}`。",
              f"campaign spec SHA256：`{data['campaign_spec_sha256']}`；输入清单 {data['evidence']['source_input_count']} 项，清单 SHA256：`{data['evidence']['source_input_manifest_sha256']}`。", ""]
    lines += table(["阶段", "分支数", "真实导出数", "pipeline墙钟 s"], [[name, len(s["per_run"]), len(s["verified_exports"]), fmt(s["pipeline_wall_seconds"], digits=2)] for name, s in data["stages"].items()])
    first = data["stages"]["smoke"]
    lines += [f"完整场景资产 SHA256：`{first['scene_assets_sha256']}`；网格 SHA256：`{first['scene_mesh_sha256']}`。",
              "完整 fresh 输入指纹另存 inputs.json；公开小型快照从最终分析直接复制质量、成本及配对值，并另存前缀后诊断汇总。生成器不重新评估网格；真实二进制/相机/预览审计由冻结 analyze_v2_campaign.py 完成。", ""]
    return "\n".join(lines)


def safe_output_directory(value):
    target = Path(value).resolve()
    if target.is_relative_to(ROOT):
        require(target.is_relative_to(ROOT / "setup") or target.is_relative_to(ROOT / "runs"),
                "仓库内仅允许输出 setup/runs；禁止写入源码、docs 或 web")
    require(not target.exists(), "输出目录必须从未存在；不会覆盖已有目录或文件")
    return target


def load_verified_analysis(path, expected_sha256, schema_path=DEFAULT_SCHEMA):
    data, sha = read(path)
    require(sha == digest(expected_sha256), "最终分析字节与独立验收 SHA256 不匹配")
    schema, schema_sha = read(schema_path)
    require(schema.get("schema") == "v2-publication-prep-schema-v1", "不是注册的 v2 发布 schema")
    validate(data, schema)
    return data, snapshot(data, sha, schema_sha)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path, help="可信服务器 fresh 全27分支最终分析 JSON")
    parser.add_argument("--expected-sha256", required=True, help="独立验收后记录的最终分析 SHA256")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA, help="默认 docs/reproduction/evidence/optimization-v2-report-schema.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="新的 setup/runs 子目录或仓库外私人目录；不覆盖")
    parser.add_argument("--plots", action="store_true", help="可选 CPU Matplotlib 图；不导入 Torch")
    args = parser.parse_args(argv)
    try:
        data, result = load_verified_analysis(args.analysis, args.expected_sha256, args.schema)
        markdown = render(result)
        target = safe_output_directory(args.output_dir)
        if args.plots:
            from plot_v2_results import draw
            import matplotlib
            matplotlib.use("Agg")
        target.mkdir(parents=True)
        with (target / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(markdown)
        for name, content in (("summary.json", result), ("inputs.json", data["source_input_sha256"])):
            with (target / name).open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        if args.plots:
            draw(result, target)
        print(json.dumps({"status": "complete", "publication_draft_generated": True, "output_dir": str(target),
                          "analysis_sha256": result["evidence"]["analysis_sha256"], "validated_branches": 27, "quality_branches": 24}, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
        print(json.dumps({"status": "rejected", "publication_draft_generated": False, "reason": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
