import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

METRICS = ["lddt", "bb_lddt", "tm_score", "rmsd"]


def compute_af3_metrics(preds, evals, name):
    metrics = {}

    top_model = None
    top_confidence = -1000
    for model_id in range(5):
        # Load confidence file
        confidence_file = (
                Path(preds) / f"seed-1_sample-{model_id}" / "summary_confidences.json"
        )
        with confidence_file.open("r") as f:
            confidence_data = json.load(f)
            confidence = confidence_data["ranking_score"]
            if confidence > top_confidence:
                top_model = model_id
                top_confidence = confidence

        # Load eval file
        eval_file = Path(evals) / f"{name}_model_{model_id}.json"
        with eval_file.open("r") as f:
            eval_data = json.load(f)
            for metric_name in METRICS:
                if metric_name in eval_data:
                    metrics.setdefault(metric_name, []).append(eval_data[metric_name])

            if "dockq" in eval_data and eval_data["dockq"] is not None:
                metrics.setdefault("dockq_>0.23", []).append(
                    np.mean([float(v > 0.23) for v in eval_data["dockq"] if v is not None]))
                metrics.setdefault("dockq_>0.49", []).append(
                    np.mean([float(v > 0.49) for v in eval_data["dockq"] if v is not None]))
                metrics.setdefault("len_dockq_", []).append(
                    len([v for v in eval_data["dockq"] if v is not None]))

        eval_file = Path(evals) / f"{name}_model_{model_id}_ligand.json"
        with eval_file.open("r") as f:
            eval_data = json.load(f)
            if "lddt_pli" in eval_data:
                lddt_plis = [x["score"] for x in eval_data["lddt_pli"]["assigned_scores"]]
                for _ in eval_data["lddt_pli"]["model_ligand_unassigned_reason"].items():
                    lddt_plis.append(0)
                if not lddt_plis:
                    continue
                lddt_pli = np.mean([x for x in lddt_plis])
                metrics.setdefault("lddt_pli", []).append(lddt_pli)
                metrics.setdefault("len_lddt_pli", []).append(len(lddt_plis))

            if "rmsd" in eval_data:
                rmsds = [x["score"] for x in eval_data["rmsd"]["assigned_scores"]]
                for _ in eval_data["rmsd"]["model_ligand_unassigned_reason"].items():
                    rmsds.append(100)
                if not rmsds:
                    continue
                rmsd2 = np.mean([x < 2.0 for x in rmsds])
                rmsd5 = np.mean([x < 5.0 for x in rmsds])
                metrics.setdefault("rmsd<2", []).append(rmsd2)
                metrics.setdefault("rmsd<5", []).append(rmsd5)
                metrics.setdefault("len_rmsd", []).append(len(rmsds))

    # Get oracle
    oracle = {k: min(v) if k == "rmsd" else max(v) for k, v in metrics.items()}
    avg = {k: sum(v) / len(v) for k, v in metrics.items()}
    top1 = {k: v[top_model] for k, v in metrics.items()}

    results = {}
    for metric_name in metrics:
        if metric_name.startswith("len_"):
            continue
        if metric_name == "lddt_pli":
            l = metrics["len_lddt_pli"][0]
        elif metric_name == "rmsd<2" or metric_name == "rmsd<5":
            l = metrics["len_rmsd"][0]
        elif metric_name == "dockq_>0.23" or metric_name == "dockq_>0.49":
            l = metrics["len_dockq_"][0]
        else:
            l = 1
        results[metric_name] = {
            "oracle": oracle[metric_name],
            "average": avg[metric_name],
            "top1": top1[metric_name],
            "len": l
        }

    return results


def compute_chai_metrics(preds, evals, name):
    metrics = {}

    top_model = None
    top_confidence = 0
    for model_id in range(5):
        # Load confidence file
        confidence_file = Path(preds) / f"scores.model_idx_{model_id}.npz"
        confidence_data = np.load(confidence_file)
        confidence = confidence_data["aggregate_score"].item()
        if confidence > top_confidence:
            top_model = model_id
            top_confidence = confidence

        # Load eval file
        eval_file = Path(evals) / f"{name}_model_{model_id}.json"
        with eval_file.open("r") as f:
            eval_data = json.load(f)
            for metric_name in METRICS:
                if metric_name in eval_data:
                    metrics.setdefault(metric_name, []).append(eval_data[metric_name])

            if "dockq" in eval_data and eval_data["dockq"] is not None:
                metrics.setdefault("dockq_>0.23", []).append(
                    np.mean([float(v > 0.23) for v in eval_data["dockq"] if v is not None]))
                metrics.setdefault("dockq_>0.49", []).append(
                    np.mean([float(v > 0.49) for v in eval_data["dockq"] if v is not None]))
                metrics.setdefault("len_dockq_", []).append(
                    len([v for v in eval_data["dockq"] if v is not None]))

        eval_file = Path(evals) / f"{name}_model_{model_id}_ligand.json"
        with eval_file.open("r") as f:
            eval_data = json.load(f)
            if "lddt_pli" in eval_data:
                lddt_plis = [x["score"] for x in eval_data["lddt_pli"]["assigned_scores"]]
                for _ in eval_data["lddt_pli"]["model_ligand_unassigned_reason"].items():
                    lddt_plis.append(0)
                if not lddt_plis:
                    continue
                lddt_pli = np.mean([x for x in lddt_plis])
                metrics.setdefault("lddt_pli", []).append(lddt_pli)
                metrics.setdefault("len_lddt_pli", []).append(len(lddt_plis))

            if "rmsd" in eval_data:
                rmsds = [x["score"] for x in eval_data["rmsd"]["assigned_scores"]]
                for _ in eval_data["rmsd"]["model_ligand_unassigned_reason"].items():
                    rmsds.append(100)
                if not rmsds:
                    continue
                rmsd2 = np.mean([x < 2.0 for x in rmsds])
                rmsd5 = np.mean([x < 5.0 for x in rmsds])
                metrics.setdefault("rmsd<2", []).append(rmsd2)
                metrics.setdefault("rmsd<5", []).append(rmsd5)
                metrics.setdefault("len_rmsd", []).append(len(rmsds))

    # Get oracle
    oracle = {k: min(v) if k == "rmsd" else max(v) for k, v in metrics.items()}
    avg = {k: sum(v) / len(v) for k, v in metrics.items()}
    top1 = {k: v[top_model] for k, v in metrics.items()}

    results = {}
    for metric_name in metrics:
        if metric_name.startswith("len_"):
            continue
        if metric_name == "lddt_pli":
            l = metrics["len_lddt_pli"][0]
        elif metric_name == "rmsd<2" or metric_name == "rmsd<5":
            l = metrics["len_rmsd"][0]
        elif metric_name == "dockq_>0.23" or metric_name == "dockq_>0.49":
            l = metrics["len_dockq_"][0]
        else:
            l = 1
        results[metric_name] = {
            "oracle": oracle[metric_name],
            "average": avg[metric_name],
            "top1": top1[metric_name],
            "len": l
        }

    return results


def compute_boltz_metrics(preds, evals, name):
    metrics = {}

    top_model = None
    top_confidence = 0
    for model_id in range(5):
        # Load confidence file
        confidence_file = (
                Path(preds) / f"confidence_{Path(preds).name}_model_{model_id}.json"
        )
        with confidence_file.open("r") as f:
            confidence_data = json.load(f)
            confidence = confidence_data["confidence_score"]
            if confidence > top_confidence:
                top_model = model_id
                top_confidence = confidence

        # Load eval file
        eval_file = Path(evals) / f"{name}_model_{model_id}.json"
        with eval_file.open("r") as f:
            eval_data = json.load(f)
            for metric_name in METRICS:
                if metric_name in eval_data:
                    metrics.setdefault(metric_name, []).append(eval_data[metric_name])

            if "dockq" in eval_data and eval_data["dockq"] is not None:
                metrics.setdefault("dockq_>0.23", []).append(
                    np.mean([float(v > 0.23) for v in eval_data["dockq"] if v is not None]))
                metrics.setdefault("dockq_>0.49", []).append(
                    np.mean([float(v > 0.49) for v in eval_data["dockq"] if v is not None]))
                metrics.setdefault("len_dockq_", []).append(
                    len([v for v in eval_data["dockq"] if v is not None]))

        eval_file = Path(evals) / f"{name}_model_{model_id}_ligand.json"
        with eval_file.open("r") as f:
            eval_data = json.load(f)
            if "lddt_pli" in eval_data:
                lddt_plis = [x["score"] for x in eval_data["lddt_pli"]["assigned_scores"]]
                for _ in eval_data["lddt_pli"]["model_ligand_unassigned_reason"].items():
                    lddt_plis.append(0)
                if not lddt_plis:
                    continue
                lddt_pli = np.mean([x for x in lddt_plis])
                metrics.setdefault("lddt_pli", []).append(lddt_pli)
                metrics.setdefault("len_lddt_pli", []).append(len(lddt_plis))

            if "rmsd" in eval_data:
                rmsds = [x["score"] for x in eval_data["rmsd"]["assigned_scores"]]
                for _ in eval_data["rmsd"]["model_ligand_unassigned_reason"].items():
                    rmsds.append(100)
                if not rmsds:
                    continue
                rmsd2 = np.mean([x < 2.0 for x in rmsds])
                rmsd5 = np.mean([x < 5.0 for x in rmsds])
                metrics.setdefault("rmsd<2", []).append(rmsd2)
                metrics.setdefault("rmsd<5", []).append(rmsd5)
                metrics.setdefault("len_rmsd", []).append(len(rmsds))

    # Get oracle
    oracle = {k: min(v) if k == "rmsd" else max(v) for k, v in metrics.items()}
    avg = {k: sum(v) / len(v) for k, v in metrics.items()}
    top1 = {k: v[top_model] for k, v in metrics.items()}

    results = {}
    for metric_name in metrics:
        if metric_name.startswith("len_"):
            continue
        if metric_name == "lddt_pli":
            l = metrics["len_lddt_pli"][0]
        elif metric_name == "rmsd<2" or metric_name == "rmsd<5":
            l = metrics["len_rmsd"][0]
        elif metric_name == "dockq_>0.23" or metric_name == "dockq_>0.49":
            l = metrics["len_dockq_"][0]
        else:
            l = 1
        results[metric_name] = {
            "oracle": oracle[metric_name],
            "average": avg[metric_name],
            "top1": top1[metric_name],
            "len": l
        }

    return results


def eval_models(chai_preds, chai_evals, af3_preds, af3_evals, boltz_preds, boltz_evals):
    # Load preds and make sure we have predictions for all models
    chai_preds_names = {x.name.lower(): x for x in Path(chai_preds).iterdir() if not x.name.lower().startswith(".")}
    af3_preds_names = {x.name.lower(): x for x in Path(af3_preds).iterdir() if not x.name.lower().startswith(".")}
    boltz_preds_names = {x.name.lower(): x for x in Path(boltz_preds).iterdir() if not x.name.lower().startswith(".")}

    print("Chai preds", len(chai_preds_names))
    print("Af3 preds", len(af3_preds_names))
    print("Boltz preds", len(boltz_preds_names))

    common = (
            set(chai_preds_names.keys())
            & set(af3_preds_names.keys())
            & set(boltz_preds_names.keys())
    )

    # Remove examples in the validation set
    keys_to_remove = ['t1133', 'h1134', 'r1134s1', 't1134s2', 't1121', 't1123', 't1159']
    for key in keys_to_remove:
        if key in common:
            common.remove(key)
    print("Common", len(common))

    # Create a dataframe with the following schema:
    # tool, name, metric, oracle, average, top1
    results = []
    for name in tqdm(common):
        try:
            af3_results = compute_af3_metrics(
                af3_preds_names[name],
                af3_evals,
                name,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error evaluating AF3 {name}: {e}")
            continue
        try:
            chai_results = compute_chai_metrics(
                chai_preds_names[name],
                chai_evals,
                name,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error evaluating Chai {name}: {e}")
            continue
        try:
            boltz_results = compute_boltz_metrics(
                boltz_preds_names[name],
                boltz_evals,
                name,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error evaluating Boltz {name}: {e}")
            continue

        for metric_name in af3_results:
            if metric_name in chai_results and metric_name in boltz_results:
                if af3_results[metric_name]["len"] == chai_results[metric_name]["len"] and af3_results[metric_name]["len"] == boltz_results[metric_name]["len"]:
                    results.append({
                        "tool": "af3",
                        "target": name,
                        "metric": metric_name,
                        "oracle": af3_results[metric_name]["oracle"],
                        "average": af3_results[metric_name]["average"],
                        "top1": af3_results[metric_name]["top1"],
                    })
                    results.append({
                        "tool": "chai",
                        "target": name,
                        "metric": metric_name,
                        "oracle": chai_results[metric_name]["oracle"],
                        "average": chai_results[metric_name]["average"],
                        "top1": chai_results[metric_name]["top1"],
                    })
                    results.append({
                        "tool": "boltz",
                        "target": name,
                        "metric": metric_name,
                        "oracle": boltz_results[metric_name]["oracle"],
                        "average": boltz_results[metric_name]["average"],
                        "top1": boltz_results[metric_name]["top1"],
                    })
                else:
                    print("Different lengths", name, metric_name, af3_results[metric_name]["len"], chai_results[metric_name]["len"], boltz_results[metric_name]["len"])
            else:
                print("Missing metric", name, metric_name, metric_name in chai_results, metric_name in boltz_results)

    # Write the results to a file, ensure we only keep the target & metrics where we have all tools
    df = pd.DataFrame(results)
    df = df.groupby(["target", "metric"]).filter(lambda x: len(x["tool"]) == 3)
    return df


def main():
    eval_folder = "../../boltz_results_final/"

    # Eval the test set
    chai_preds = eval_folder + "outputs/test/chai"
    chai_evals = eval_folder + "evals/test/chai"

    af3_preds = eval_folder + "outputs/test/af3"
    af3_evals = eval_folder + "evals/test/af3"

    boltz_preds = eval_folder + "outputs/test/boltz/predictions"
    boltz_evals = eval_folder + "evals/test/boltz"

    df = eval_models(chai_preds, chai_evals, af3_preds, af3_evals, boltz_preds, boltz_evals)
    df.to_csv(eval_folder + "results_test.csv", index=False)

    print("Test results: mean")
    print(df[["tool", "metric", "oracle", "average", "top1"]].groupby(["tool", "metric"]).mean())

    # Eval CASP
    chai_preds = eval_folder + "outputs/casp15/chai"
    chai_evals = eval_folder + "evals/casp15/chai"

    af3_preds = eval_folder + "outputs/casp15/af3"
    af3_evals = eval_folder + "evals/casp15/af3"

    boltz_preds = eval_folder + "outputs/casp15/boltz/predictions"
    boltz_evals = eval_folder + "evals/casp15/boltz"

    df = eval_models(chai_preds, chai_evals, af3_preds, af3_evals, boltz_preds, boltz_evals)
    df.to_csv(eval_folder + "results_casp.csv", index=False)

    print("CASP15 results: mean")
    print(df[["tool", "metric", "oracle", "average", "top1"]].groupby(["tool", "metric"]).mean())


if __name__ == "__main__":
    main()