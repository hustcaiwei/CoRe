"""Dataset adapters and the eleven term-of-penalty intervals used for LJP.

MultiLJP is represented at defendant level. Adapters preserve input splits;
they do not reconstruct the original paper's filtering or train/test split.
"""
import re

FACETS = ("behavior", "consequence", "fact")


def charge_name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Charge labels must be nonempty strings")
    value = value.strip()
    return value[:-1] if value.endswith("罪") else value


def term_class(months):
    months = int(months)
    if months == -1:
        return 0
    if months < -1:
        raise ValueError("Penalty months must be -1 (death) or nonnegative")
    for code, lower_bound in enumerate((120, 84, 60, 36, 24, 12, 9, 6, 0), 1):
        if months > lower_bound:
            return code
    return 10


def chinese_number(text):
    if text.isdigit():
        return int(text)
    digits = dict(zip("零一二三四五六七八九", range(10)))
    digits["两"] = 2
    if text in digits:
        return digits[text]
    if "十" in text:
        left, right = text.split("十", 1)
        return (digits[left] if left else 1) * 10 + (digits[right] if right else 0)
    raise ValueError(f"Unsupported Chinese number: {text}")


def term_months(value):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("One final penalty is required per defendant; multiple terms need defendant IDs")
        value = value[0]
    if isinstance(value, bool):
        raise ValueError("A boolean is not a penalty")
    if isinstance(value, int) or (isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip())):
        result = int(value)
        term_class(result)
        return result
    text = str(value).strip()
    # Suspension is a separate duration, never part of the custodial penalty.
    text = text.split("缓刑", 1)[0].strip(" ，,。;；")
    if "死刑" in text:
        return -1
    if "无期徒刑" in text:
        return 360
    if text in ("免予刑事处罚", "免于刑事处罚", "无刑期", "无"):
        return 0
    pattern = r"([零一二三四五六七八九十两\d]+)\s*"
    duration = re.fullmatch(r"(?:有期徒刑|拘役)?\s*(?:" + pattern + r"年)?(?:" + pattern + r"(?:个月|月))?", text)
    if duration and any(duration.groups()):
        years, months = duration.groups()
        return (chinese_number(years) * 12 if years else 0) + (chinese_number(months) if months else 0)
    days = re.search(pattern + r"日", text)
    if days and "拘役" in text:
        return max(1, chinese_number(days[1]) // 30)
    raise ValueError(f"Unsupported penalty string: {text!r}; supply explicit term_months")


def label_values(value, field):
    values = value if isinstance(value, list) else [value]
    if not values or any(isinstance(v, (list, dict, bool)) or v is None for v in values):
        raise ValueError(f"{field} must contain one or more valid labels")
    return values


def charge_labels(value):
    return sorted(set(charge_name(v) for v in label_values(value, "charge")))


def article_labels(value):
    result = []
    for item in label_values(value, "law_article"):
        if not isinstance(item, int) and not (isinstance(item, str) and re.fullmatch(r"\d+", item.strip())):
            raise ValueError("Law articles must be positive integers")
        article = int(item)
        if article <= 0:
            raise ValueError("Law articles must be positive integers")
        result.append(article)
    return sorted(set(result))


def labels(charge, law, months):
    months = term_months(months)
    return {"charge": charge_labels(charge), "law_article": article_labels(law),
            "term_months": months, "term_class": term_class(months)}


def single(value, field):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{field}: expected exactly one label; filter multi-label records explicitly")
        return value[0]
    return value


def multiljp_labels(info):
    return labels(info["accusations"], info["laws"], info["term"])


def canonical(record):
    result = {"id": str(record["id"]), "fact": record.get("fact", "")}
    for key in ("origin_id", "defendant", "dataset"):
        if key in record:
            result[key] = record[key]
    if "elements" in record:
        elements = record["elements"]
        result["elements"] = {f: elements[f] for f in FACETS}
        if not all(isinstance(v, str) for v in result["elements"].values()):
            raise ValueError("All three elements must be text strings")
    if not isinstance(result["fact"], str):
        raise ValueError("fact must be a string")
    if "labels" in record:
        info = record["labels"]
        result["labels"] = labels(info["charge"], info["law_article"], info["term_months"])
    return result


def normalize(record, source_format, charges=None, laws=None):
    """Normalize one source record; only observed ground truth becomes labels."""
    if source_format == "canonical":
        return [canonical(record)]
    if source_format in ("cail", "cail-elements"):
        result = {"id": str(record["id"]), "dataset": "cail",
                  "fact": record.get("fact", record.get("fact_cut", ""))}
        if source_format == "cail-elements":
            result["elements"] = {"behavior": record["behavior"],
                                  "consequence": record["consequence"], "fact": record["fact_summary"]}
        if "meta" in record:
            meta = record["meta"]
            penalty = meta["term_of_imprisonment"]
            months = -1 if penalty["death_penalty"] else 360 if penalty["life_imprisonment"] else penalty["imprisonment"]
            result["labels"] = labels(meta["accusation"], meta["relevant_articles"], months)
        elif all(k in record for k in ("accu", "law", "term")):
            charge = single(record["accu"], "accu")
            law = single(record["law"], "law")
            if isinstance(charge, int):
                if charges is None or laws is None:
                    raise ValueError("Indexed CAIL labels require --charges and --laws (zero-based tables)")
                charge, law = charges[charge], laws[int(law)]
            months = record["term"]
            if record.get("term_cate") == 0:
                months = -1
            elif record.get("term_cate") == 1:
                months = 360
            result["labels"] = labels(charge, law, months)
        return [canonical(result)]
    if source_format == "multiljp-elements" and "data" in record:
        info = record["data"]
        elements = info["elements"]
        result = {"id": str(record["case_id"]), "origin_id": record["origin_id"], "dataset": "multiljp",
                  "defendant": record["defendant"], "fact": record.get("fact", ""),
                  "elements": {"behavior": elements["行为"], "consequence": elements["后果"],
                               "fact": elements["犯罪事实"]}, "labels": multiljp_labels(info["crime_info"])}
        return [canonical(result)]
    if source_format in ("multiljp", "multiljp-elements"):
        extracted = source_format == "multiljp-elements"
        info = record["crime_info"] if extracted else record["criminals_info"]
        result = []
        for defendant, truth in info.items():
            name = defendant.strip("[]")
            item = {"id": f"{record['id']}:{name}", "origin_id": record["id"], "dataset": "multiljp",
                    "defendant": name, "fact": record.get("fact", ""), "labels": multiljp_labels(truth)}
            if extracted:
                elements = record["elements"][name]
                item["elements"] = {"behavior": elements["行为"], "consequence": elements["后果"],
                                    "fact": elements["犯罪事实"]}
            result.append(canonical(item))
        return result
    raise ValueError(f"Unsupported source format: {source_format}")
