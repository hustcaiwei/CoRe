"""CEE: extract behavior, consequence, and fact from the case description."""
import json

from .data import FACETS


def build_prompt(record):
    if not isinstance(record.get("fact"), str) or not record["fact"].strip():
        raise ValueError("CEE requires the original case description")
    prompt = """你是一名法律信息抽取助手。根据给定的案件描述提取三个要素：
behavior：被告的具体行为，包括手段、过程、参与角色、次数等。
consequence：行为直接造成的后果，包括财物损失、伤害程度、社会影响等。
fact：删除无关实体和叙述后，保留与定罪和量刑有关的事实摘要。
仅使用描述中明确存在的信息，不推断罪名、法条或判决刑期。
没有明确行为后果时可填写“未明确描述”。仅输出 JSON：
{"behavior":"...","consequence":"...","fact":"..."}
"""
    if record.get("defendant"):
        prompt += "本次只提取被告“" + record["defendant"] + "”的要素。\n"
    return prompt + "案件描述：\n" + record["fact"]


def parse_elements(text):
    try:
        elements = json.loads(text)
    except json.JSONDecodeError:
        from json_repair import loads
        elements = loads(text)
    if not isinstance(elements, dict) or any(not isinstance(elements.get(f), str)
                                             or not elements[f].strip() for f in FACETS):
        raise ValueError("CEE must return three nonempty text elements")
    return {f: elements[f] for f in FACETS}


async def extract(record, api, temperature):
    text = await api.chat(build_prompt(record), temperature)
    return {**record, "elements": parse_elements(text), "cee_validated": False, "status": "ok"}


async def audit_and_repair(record, evaluator, config, repairer=None):
    """Evaluate on 0-5 scales, retain good elements, or perform targeted repair.

    Ground-truth judgment labels are excluded from quality prompts. Decisions
    and raw audit/repair responses are retained for reproducibility.
    """
    build_prompt(record)  # Validate the original description before an API call.
    elements = parse_elements(json.dumps(record["elements"], ensure_ascii=False))
    prompt = ('根据原始案情评价抽取结果；只使用原文，不推断罪名或判决，不使用外部知识。'
        '以下六项评分为0至5：行为忠实性behavior_faithfulness，完整性behavior_completeness，'
        '后果有效性consequence_faithfulness，事实摘要质量summary_quality，'
        '无幻觉non_hallucination，总分overall。标记严重幻觉severe_hallucination。'
        'error_types可包括hallucination、missing_key_fact、over_simplified、invalid_consequence。'
        '请输出JSON：{"scores":{"behavior_faithfulness":5,"behavior_completeness":5,'
        '"consequence_faithfulness":5,"summary_quality":5,"non_hallucination":5,"overall":5},'
        '"severe_hallucination":false,"error_types":[],"revision_advice":""}。'
        '\n原始案情：' + record["fact"] + '\n抽取结果：' + json.dumps(elements, ensure_ascii=False))
    if record.get("defendant"):
        prompt += '\n仅评价目标被告：' + record["defendant"]
    raw = await evaluator.chat(prompt, config["cee_evaluation_temperature"],
                               max_tokens=config["cee_evaluation_max_tokens"])
    from json_repair import loads
    evaluation = loads(raw)
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("scores"), dict):
        raise ValueError("CEE evaluator must return a scores object")
    for name in ("behavior_faithfulness", "behavior_completeness", "consequence_faithfulness",
                 "summary_quality", "non_hallucination", "overall"):
        score = evaluation["scores"].get(name)
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not 0 <= score <= 5:
            raise ValueError(f"Invalid CEE score: {name}")
    severe = evaluation.get("severe_hallucination")
    if not isinstance(severe, bool) or not isinstance(evaluation.get("error_types"), list):
        raise ValueError("CEE evaluator must explicitly classify severe hallucination and errors")
    errors = set(evaluation["error_types"])
    if evaluation["scores"]["overall"] >= config["cee_keep_threshold"] and not severe:
        action = "keep"
    elif severe or "hallucination" in errors:
        action = "re_extract_with_strict_evidence"
    elif errors == {"invalid_consequence"} and evaluation["scores"]["overall"] >= 3:
        action = "repair_consequence_only"
    elif evaluation["scores"]["overall"] <= 2:
        action = "automated_fallback_repair"
    else:
        action = "re_extract_with_feedback"
    result = {**record, "elements": elements, "cee_evaluation": evaluation, "cee_action": action,
              "raw_cee_evaluation": raw, "cee_evaluation_model": evaluator.model,
              "cee_validated": False, "status": "ok"}
    if action != "keep":
        repairer = repairer or evaluator
        instruction = ('仅修复consequence，输出{"consequence":"修复后的文本"}。'
                       if action == "repair_consequence_only" else
                       '重新抽取behavior、consequence、fact，输出三个非空字符串字段的JSON。')
        repair_prompt = ('依据原始案情和反馈修复抽取，不添加无原文依据的内容，不输出判决标签。'
            '原本正确的要素应保持不变。' + instruction + '\n修复方式：' + action +
            '\n原文：' + record["fact"] + '\n旧要素：' + json.dumps(elements, ensure_ascii=False) +
            '\n反馈：' + json.dumps(evaluation, ensure_ascii=False))
        if record.get("defendant"):
            repair_prompt += '\n仅修复目标被告：' + record["defendant"]
        repaired_raw = await repairer.chat(repair_prompt, config["cee_repair_temperature"],
                                           max_tokens=config["cee_evaluation_max_tokens"])
        repaired = loads(repaired_raw)
        if action == "repair_consequence_only":
            if not isinstance(repaired, dict): raise ValueError("Invalid consequence repair")
            repaired = {**elements, "consequence": repaired.get("consequence")}
        result["elements"] = parse_elements(json.dumps(repaired, ensure_ascii=False))
        result.update(raw_cee_repair=repaired_raw, cee_repair_model=repairer.model)
    result["cee_validated"] = True  # Schema-valid output after the automatic keep/repair action.
    return result
