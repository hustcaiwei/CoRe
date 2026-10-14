"""Paper parameters; batch size and concurrency are execution settings."""
import json

DEFAULTS = {"llm_model": "deepseek-v3", "embedding_model": "text-embedding-ada-002",
            "embedding_dimension": 1536, "cee_temperature": .1, "judge_a_temperature": .1,
            "judge_b_temperature": .1, "top_m": 5000, "top_k": 19, "alpha": 60,
            "rpr_scope": "intersection", "batch_size": 16, "concurrency": 8,
            "cee_evaluation_model": "deepseek-v4-pro", "cee_repair_model": "deepseek-v4-flash",
            "cee_evaluation_temperature": 0, "cee_repair_temperature": 0,
            "cee_evaluation_max_tokens": 4096, "cee_keep_threshold": 4}


def load_config(path=None):
    config = dict(DEFAULTS)
    if path:
        with open(path, encoding="utf-8-sig") as handle:
            override = json.load(handle)
        unknown = set(override) - set(config)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config.update(override)
    for name in ("embedding_dimension", "top_m", "top_k", "batch_size", "concurrency", "cee_evaluation_max_tokens"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["alpha"] < 0 or config["rpr_scope"] not in ("union", "intersection"):
        raise ValueError("Invalid RPR configuration")
    for name in ("cee_temperature", "judge_a_temperature", "judge_b_temperature", "cee_evaluation_temperature", "cee_repair_temperature"):
        if not 0 <= config[name] <= 2:
            raise ValueError(f"{name} must lie in [0, 2]")
    if not 0 <= config["cee_keep_threshold"] <= 5:
        raise ValueError("cee_keep_threshold must lie in [0, 5]")
    return config
