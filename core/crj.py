"""CRJ: Judge A predicts; Judge B reselects precedents and refines judgment."""
import json
import re

from .data import charge_labels, article_labels, term_class
from .rpr import select_second_round

def integer(value, name):
    if isinstance(value, bool) or not (isinstance(value, int) or
            isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip())):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def parse_prediction(text, term_unit):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        from json_repair import loads
        value = loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a prediction JSON object")
    result = {"charge": charge_labels(value.get("罪名", value.get("pred_accusation"))),
              "law_article": article_labels(value.get("法条", value.get("pred_relevant_articles")))}
    penalty = value.get("刑期", value.get("pred_imprisonment_months"))
    if isinstance(penalty, list):
        if len(penalty) != 1:
            raise ValueError("Multiple penalties cannot be assigned to this defendant; output one final term")
        penalty = penalty[0]
    term = integer(penalty, "刑期")
    if term_unit == "months":
        result.update(term_months=term, term_class=term_class(term))
    elif term_unit == "class" and 0 <= term <= 10:
        result["term_class"] = term
    else:
        raise ValueError("Expected a penalty interval code from 0 through 10")
    return result


def build_prompt(query, precedents, charges, stage="a", preliminary=None):
    """Only case facts and training labels enter prompts; query labels stay out."""
    parts = ["你是一名专业刑事法官，请根据《中华人民共和国刑法》预测案件的罪名、法条和刑期。",
             "罪名库：" + "、".join(charges)]
    for rank, precedent in enumerate(precedents, 1):
        info = precedent["labels"]
        fact = precedent.get("elements", {}).get("fact", precedent["fact"])
        parts.append(f"参考先例{rank}（按相关性从高到低）：\n案件事实：{fact}\n"
                     f"判决结果：罪名：{info['charge']}；法条：{info['law_article']}；"
                     f"刑期（月）：{info['term_months']}")
    fact = (query["fact"] if query.get("defendant") or query.get("dataset") == "multiljp"
            else query.get("elements", {}).get("fact", query["fact"]))
    if not fact.strip():
        raise ValueError("The original case description is required, including for defendant-specific inference")
    parts.append("当前案件事实：\n" + fact)
    if query.get("defendant"):
        parts.append("本次仅预测被告：" + query["defendant"])
    if stage == "a":
        parts.append('根据先例与案件事实，比较犯罪行为、后果和量刑情节，逐步判断罪名、法条和刑期。'
                     '对当前被告输出一份最终判断；真实涉及多项罪名或法条时保留全部项目，不输出备选猜测。'
                     '仅输出 JSON 对象，格式：{"罪名":["故意伤害"],"法条":[234],"刑期":24}。'
                     '刑期以月为单位，每名被告一个最终刑期；死刑为-1，无期徒刑为360。'
                     '罪名不要带末尾的“罪”字，不输出额外解释。')
    else:
        parts.append("法官A的初步预测：" + json.dumps(preliminary, ensure_ascii=False))
        parts.append("你是法官B。参考重新选择的先例，核对罪名与法条，"
                     "根据行为、后果和量刑事实重新评估刑期；必要时修正法官A的预测。"
                     '保留全部真实涉及的罪名与法条，逐步比较行为、后果和明确的量刑事实；'
                     '不推断未给出的情节，可推翻法官A的预测。刑期按月输出，每名被告一个最终刑期，'
                     '死刑为-1，无期徒刑为360。'
                     '\n仅输出 JSON 对象，格式：{"罪名":["故意伤害"],"法条":[234],"刑期":24}。')
    return "\n\n".join(parts)


async def judge(query, ranked, precedents, charges, api, config):
    first_ids = [i for i, _ in ranked[:config["top_k"]]]
    text_a = await api.chat(build_prompt(query, [precedents[i] for i in first_ids], charges),
                            config["judge_a_temperature"])
    prediction_a = parse_prediction(text_a, "months")
    second_ids = select_second_round(ranked, precedents, prediction_a["charge"],
                                     prediction_a["law_article"], config["top_k"])
    text_b = await api.chat(build_prompt(query, [precedents[i] for i in second_ids], charges,
                                        "b", prediction_a), config["judge_b_temperature"])
    prediction_b = parse_prediction(text_b, "months")
    return {"id": query["id"], "status": "ok", "judge_a": prediction_a, "judge_b": prediction_b,
            "first_round_ids": [precedents[i]["id"] for i in first_ids],
            "second_round_ids": [precedents[i]["id"] for i in second_ids],
            "candidate_count": len(ranked), "raw_judge_a": text_a, "raw_judge_b": text_b}
