"""Exact label-set accuracy, per-label macro metrics, and 11-class penalty metrics."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from .data import charge_labels, article_labels

from .io import read_jsonl, unique_records


def set_metrics(truth, predictions):
    encoder = MultiLabelBinarizer().fit(truth + predictions)
    actual = encoder.transform(truth).astype(bool)
    predicted = encoder.transform(predictions).astype(bool)
    tp = (actual & predicted).sum(axis=0)
    fp = (~actual & predicted).sum(axis=0)
    fn = (actual & ~predicted).sum(axis=0)
    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros(len(numerator)), where=denominator != 0)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return {"accuracy": float((actual == predicted).all(axis=1).mean()) * 100,
            "macro_precision": float(precision.mean()) * 100,
            "macro_recall": float(recall.mean()) * 100, "macro_f1": float(f1.mean()) * 100,
            "accuracy_definition": "exact_label_set", "label_count": len(encoder.classes_)}


def evaluate(truth_path, predictions_path):
    truth = {str(r["id"]): r["labels"] for r in unique_records(read_jsonl(truth_path))}
    predicted = {str(r["id"]): r for r in unique_records(read_jsonl(predictions_path))}
    unknown = set(predicted) - set(truth)
    if unknown:
        raise ValueError(f"Predictions contain {len(unknown)} ids absent from the ground truth")
    successful = {key for key in truth if predicted.get(key, {}).get("status") == "ok"}
    report = {"expected": len(truth), "successful": len(successful),
              "failed_or_missing": len(truth) - len(successful),
              "coverage": len(successful) / len(truth) if truth else 0,
              "metric_scale": "percent", "scoring_subset": "all labeled records; failures count as incorrect"}
    for stage in ("judge_a", "judge_b"):
        report[stage] = {}
        for field in ("charge", "law_article", "term_class"):
            if not truth:
                report[stage][field] = None
                continue
            y_true, y_pred = [], []
            for key, info in truth.items():
                success = key in successful
                output = predicted[key][stage] if success else {}
                if field != "term_class":
                    normalizer = charge_labels if field == "charge" else article_labels
                    y_true.append(normalizer(info[field]))
                    y_pred.append(normalizer(output[field]) if success else [])
                else:
                    actual, prediction = info[field], output[field] if success else -1
                    if isinstance(actual, bool) or not isinstance(actual, int) or not 0 <= actual <= 10:
                        raise ValueError("Truth penalty class must be an integer from 0 to 10")
                    if success and (isinstance(prediction, bool) or not isinstance(prediction, int) or not 0 <= prediction <= 10):
                        raise ValueError("Predicted penalty class must be an integer from 0 to 10")
                    y_true.append(actual)
                    y_pred.append(prediction)
            if field != "term_class":
                report[stage][field] = set_metrics(y_true, y_pred)
            else:
                p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(11)), average="macro", zero_division=0)
                report[stage][field] = {"accuracy": 100 * float(accuracy_score(y_true, y_pred)),
                    "macro_precision": 100 * float(p), "macro_recall": 100 * float(r),
                    "macro_f1": 100 * float(f), "accuracy_definition": "interval_match", "label_count": 11}
    return report
