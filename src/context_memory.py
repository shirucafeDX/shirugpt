"""原本を残したまま、モデルへ送る会話だけを階層化する。追加の生成呼び出しはしない。"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from google.genai import types

logger = logging.getLogger(__name__)

RECENT_TURNS = 2
NOTE_CHARS = 1200
OVERVIEW_CHARS = 4000
COMPACT_MIN_CHARS = 2000
MAX_OPEN_QUESTIONS = 12
QUESTION_CHARS = 300
CONCLUSION_CHARS = 600


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def text_of(content) -> str:
    return "".join(part.text for part in content.parts or [] if part.text and not part.thought)


def is_question(content) -> bool:
    return content.role == "user" and any(p.text for p in content.parts or []) and not any(p.function_response for p in content.parts or [])


def state_for(ctx) -> dict:
    state = ctx.context_state
    state.setdefault("turns", [])
    state.setdefault("results", {})
    state.setdefault("questions", {})
    return state


def sync_turns(contents, ctx) -> dict:
    """旧履歴・停止したターンも元の会話から索引化する。現在の署名は変更しない。"""
    state = state_for(ctx)
    starts = [i for i, content in enumerate(contents) if is_question(content)]
    turns = state["turns"]
    if turns and (not starts or turns[-1]["start"] > starts[-1]
                  or any(t["start"] >= len(contents) or t.get("end", 0) > len(contents)
                         or t["question"] != text_of(contents[t["start"]]) for t in turns)):
        ctx.context_state = {}
        return sync_turns(contents, ctx)
    known = {turn["start"] for turn in turns}
    for start in starts:
        if start not in known:
            turns.append({"id": f"turn_{start}", "start": start, "question": text_of(contents[start]),
                          "result_ids": [], "summary": "", "overview": ""})
    turns.sort(key=lambda turn: turn["start"])
    for i, turn in enumerate(turns):
        if i + 1 < len(turns):
            turn["end"] = turns[i + 1]["start"]
        end = turn.get("end", len(contents))
        if end > turn["start"]:
            turn["answer"] = next((text_of(c) for c in reversed(contents[turn["start"]:end])
                                   if c.role == "model" and text_of(c)), "")
    # 移行前の履歴のツール応答にも、会話内だけで通用するIDを付ける。
    for turn in turns:
        end = turn.get("end", len(contents))
        calls = []
        for i in range(turn["start"], end):
            for j, part in enumerate(contents[i].parts or []):
                if part.function_call:
                    calls.append(part.function_call)
                if part.function_response and part.function_response.name not in ("finish_answer", "select_tools"):
                    payload = part.function_response.response or {}
                    result_id = payload.get("result_id") or f"legacy_{i}_{j}"
                    if result_id not in state["results"]:
                        state["results"][result_id] = {"name": part.function_response.name, "payload": payload,
                                                       "note": "", "purpose": ""}
                        call = next((call for call in reversed(calls)
                                     if call.name == part.function_response.name
                                     and (not part.function_response.id or call.id == part.function_response.id)), None)
                        if call and call.name == "run_query" and call.args and call.args.get("sql"):
                            record = state["results"][result_id]
                            record["query_key"] = hashlib.sha256(call.args["sql"].strip().encode()).hexdigest()
                            record["purpose"] = call.args.get("purpose", "")
                    if result_id not in turn["result_ids"]:
                        turn["result_ids"].append(result_id)
    return turns[-1] if turns else {}


def result_id_for(name: str, args: dict) -> str:
    clean = {key: value for key, value in sorted(args.items()) if key != "context_note"}
    if name == "run_query":
        clean = {"sql": str(args.get("sql", "")).strip()}
    return "result_" + hashlib.sha256((name + encoded(clean)).encode()).hexdigest()[:20]


def validate_note(ctx, value: object) -> str | None:
    """未知の参照を含むメモでは原結果を短縮しない。"""
    if value is None:
        return None
    state = state_for(ctx)
    if not isinstance(value, dict):
        return "context_noteはオブジェクトで指定してください。"
    summary, refs = value.get("summary"), value.get("references", [])
    if not isinstance(summary, str) or not summary.strip() or len(summary) > NOTE_CHARS:
        return f"context_note.summaryは1〜{NOTE_CHARS}文字で指定してください。"
    if not isinstance(refs, list) or len(refs) > 12 or any(not isinstance(ref, str) or ref not in state["results"] for ref in refs):
        return "context_note.referencesには受け取ったresult_idだけを指定してください。"
    questions = state["questions"]
    turn_id = state["turns"][-1]["id"] if state["turns"] else ctx.turn_id
    updates, resolved = value.get("open_questions", []), value.get("resolved_questions", [])
    if not isinstance(updates, list) or len(updates) > MAX_OPEN_QUESTIONS or not isinstance(resolved, list) or len(resolved) > MAX_OPEN_QUESTIONS:
        return "open_questionsとresolved_questionsはそれぞれ最大12件の配列で指定してください。"
    seen = set()
    for items, field, max_chars in ((updates, "question", QUESTION_CHARS), (resolved, "conclusion", CONCLUSION_CHARS)):
        for item in items:
            if not isinstance(item, dict):
                return "未確認事項はID付きのオブジェクトで指定してください。"
            question_id, content = item.get("id"), item.get(field)
            if not isinstance(question_id, str) or re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,39}", question_id) is None:
                return "未確認事項のidは英字で始まる40文字以内の英数字・ハイフン・アンダースコアにしてください。"
            if question_id in seen:
                return "同じ未確認事項のidを重複して更新・解決できません。"
            seen.add(question_id)
            if not isinstance(content, str) or not content.strip() or len(content) > max_chars:
                return f"{field}は1〜{max_chars}文字で指定してください。"
            existing = questions.get(question_id)
            if field == "question" and existing and existing["status"] == "resolved":
                return "解決済みの未確認事項は再登録できません。新しい疑問には別のidを使ってください。"
            if field == "question" and any(item["status"] == "resolved" and item.get("turn_id", "") == turn_id and content.strip() in
                                            [item["question"], *item.get("previous_questions", [])]
                                            for item in questions.values()):
                return "同じターンで解決済みの疑問は別のidでも再登録できません。"
            if field == "conclusion" and not existing:
                return "resolved_questionsには登録済みの未確認事項のidを指定してください。"
    if len(factual_summary(value)) > NOTE_CHARS:
        return "summaryと解決した結論は合わせて1200文字以内にまとめてください。"
    remaining = {key for key, question in questions.items() if question["status"] == "open"}
    remaining.update(item["id"] for item in updates)
    remaining.difference_update(item["id"] for item in resolved)
    if len(remaining) > MAX_OPEN_QUESTIONS:
        return "未確認事項は同時に12件までです。既存の事項を解決・整理してから追加してください。"
    return None


def accept_note(ctx, value: object) -> str | None:
    warning = validate_note(ctx, value)
    if warning or value is None:
        return warning
    state = state_for(ctx)
    state['current_note'] = {'turn_id': state['turns'][-1]['id'] if state['turns'] else ctx.turn_id,
                             'summary': factual_summary(value),
                             'references': list(value.get('references', []))}
    for item in value.get("open_questions", []):
        existing = state["questions"].get(item["id"], {})
        previous = list(existing.get("previous_questions", []))
        if existing.get("question") and existing["question"] != item["question"].strip():
            previous.append(existing["question"])
        state["questions"][item["id"]] = {"question": item["question"].strip(), "status": "open",
                                          "previous_questions": list(dict.fromkeys(previous)),
                                          "turn_id": existing.get("turn_id", state["turns"][-1]["id"] if state["turns"] else ctx.turn_id)}
    for item in value.get("resolved_questions", []):
        state["questions"][item["id"]].update(status="resolved", conclusion=item["conclusion"].strip())
    for ref in value.get("references", []):
        # 記録のnoteはあくまでモデルの解釈。原数値はrecall_contextで照合する。
        state["results"][ref]["note"] = factual_summary(value)
    return None


def factual_summary(note: dict) -> str:
    """解決した結論は事実として残す。疑問文・未確認一覧は要約へ混ぜない。"""
    conclusions = [f"- {item['conclusion'].strip()}" for item in note.get("resolved_questions", [])
                   if item['conclusion'].strip() not in note["summary"]]
    return "\n".join([note["summary"].strip(), *conclusions])


def facts_only(state: dict, value: str) -> str:
    """独立した既知の疑問・未確認行を除外する。原文や確定事実の行は書き換えない。"""
    questions = [question for item in state.get("questions", {}).values()
                 for question in [item["question"], *item.get("previous_questions", [])]]
    lines = []
    for line in value.splitlines():
        body = re.sub(r"^\s*(?:[-*]|\d+[.)])?\s*", "", line)
        pending = re.match(r"(?:未確認(?:事項|点)?|未解決(?:事項|点)?|不明点|要確認|TODO)\s*[:：]", body)
        if any(body == question or (pending and question in body) for question in questions):
            continue
        lines.append(line)
    return "\n".join(lines)


def active_questions(state: dict) -> dict:
    return {"open_questions": [{"id": key, "question": item["question"]}
                               for key, item in state.get("questions", {}).items() if item["status"] == "open"],
            "instruction": "これは現在の未確認事項の全件一覧です。過去のメモ・呼び出し内の未確認事項一覧は履歴です。この一覧にない事項を未解決として再登録しないでください。"}


def register_result(ctx, name: str, args: dict, payload: dict) -> dict:
    state = state_for(ctx)
    result_id = result_id_for(name, args)
    previous = state["results"].get(result_id, {})
    record = {"name": name, "purpose": str(args.get("purpose", "")), "payload": payload,
              "note": previous.get("note", "")}
    if name == "run_query":
        record["query_key"] = hashlib.sha256(str(args.get("sql", "")).strip().encode()).hexdigest()
    elif name == "analyze_result":
        record["analysis_key"] = result_id
    elif name == "recall_context" and args.get("result_id"):
        source = state["results"].get(args["result_id"], {})
        for key in ("query_key", "analysis_key"):
            if source.get(key):
                record[key] = source[key]
    state["results"][result_id] = record
    if state["turns"] and result_id not in state["turns"][-1]["result_ids"]:
        state["turns"][-1]["result_ids"].append(result_id)
    return {**payload, "result_id": result_id}


def finish_turn(contents, ctx, summary: str = "", overview: str = "") -> None:
    turn = sync_turns(contents, ctx)
    if turn:
        turn["end"] = len(contents)
        turn["summary"] = summary
        turn["overview"] = overview[:OVERVIEW_CHARS]


def catalogue(state: dict, ids: list[str]) -> list[dict]:
    items = []
    for result_id in ids:
        record = state["results"].get(result_id, {})
        payload = record.get("payload", {})
        result = payload.get("result", {})
        item = {"result_id": result_id, "tool": record.get("name"), "purpose": record.get("purpose", "")}
        if record.get("note"):
            item["summary"] = facts_only(state, record["note"])
        if isinstance(result, dict):
            for key in ("row_count", "column_count", "truncated", "truncation_note"):
                if key in result:
                    item[key] = result[key]
        if isinstance(result, dict) and isinstance(result.get("columns"), list):
            item["columns"] = [{k: column[k] for k in ("name", "dtype") if k in column}
                               for column in result["columns"] if isinstance(column, dict)]
        for key in ("source_result_id", "total_row_count", "source_truncated", "offset", "more_rows",
                    "statistics_scope", "only_saved_samples_available", "available_sample_count",
                    "matched_row_count", "filters_applied", "returned_row_count", "next_offset"):
            if key in payload:
                item[key] = payload[key]
        if payload.get("error"):
            item["error"] = payload["error"]
        if payload.get("notes"):
            item["warnings"] = payload["notes"]
        items.append(item)
    return items


def ensure_user_ending(contents: list[types.Content]) -> list[types.Content]:
    """送信時だけ末尾を補正する。署名付きPart・保存原本・ツールの実行状態は変更しない。"""
    request = list(contents)
    # Vertexではfunction_responseの後ろにtextを並べると、role=userでも
    # model末尾と解釈される場合がある。補助説明は応答JSONへ入れてPartを分けない。
    for index, content in enumerate(request):
        parts = content.parts or []
        responses = [i for i, part in enumerate(parts) if part.function_response is not None]
        texts = [part.text for part in parts if part.text is not None]
        if content.role != "user" or not responses or not texts:
            continue
        last_response = responses[-1]
        response = parts[last_response].function_response
        payload = dict(response.response or {})
        payload["context_text"] = [*payload.get("context_text", []), *texts]
        replacement = parts[last_response].model_copy(update={
            "function_response": response.model_copy(update={"response": payload})})
        request[index] = content.model_copy(update={"parts": [
            replacement if i == last_response else part for i, part in enumerate(parts) if part.text is None]})
        logger.warning("Gemini送信会話のツール応答とtextの混在を補正しました。")
    # APIは空の末尾を無視する。空のuserを置くだけではmodel末尾のエラーを回避できない。
    while request and not any(
        (part.text and part.text.strip() and not part.thought)
        or part.function_call is not None or part.function_response is not None
        or part.inline_data is not None or part.file_data is not None
        for part in request[-1].parts or []
    ):
        request.pop()
    if not request:
        raise ValueError("送信する質問がありません。質問を入力してください。")
    if request[-1].role == "user":
        return request
    if request[-1].role != "model":
        raise ValueError("会話末尾のroleはuserまたはmodelである必要があります。")
    calls = [part.function_call for part in request[-1].parts or [] if part.function_call]
    if calls:
        # 応答が欠けた呼び出しを勝手に再実行せず、不明な状態を明示する。
        parts = []
        for call in calls:
            part = types.Part.from_function_response(name=call.name, response={
                "error": "前回のツール呼び出しの応答が履歴に残っていません。実行の成否は未確認です。保存済み結果を先に確認し、必要なら再度呼び出してください。",
                "status": "response_missing",
            })
            part.function_response.id = call.id
            parts.append(part)
    else:
        parts = [types.Part.from_text(text="直前のユーザーの依頼への回答を続けてください。すでに得られた結果と確定した条件を使い、未確認の点は推測しないでください。")]
    request.append(types.Content(role="user", parts=parts))
    logger.warning("Gemini送信会話のmodel末尾を補正しました（未応答のツール呼び出し数: %d）。", len(calls))
    return request


def build_request(contents, ctx) -> list[types.Content]:
    current = sync_turns(contents, ctx)
    if not current:
        return ensure_user_ending(contents)
    state = state_for(ctx)
    past = [turn for turn in state["turns"] if turn["start"] < current["start"]]
    older, recent = past[:-RECENT_TURNS], past[-RECENT_TURNS:]
    request = []
    if older and (not older[-1].get("overview") or any(not t.get("summary") for t in older)):
        recent = past
        older = []
    if older:
        overview = older[-1].get("overview", "")
        fallback = "\n".join(f"{t['id']}: {t['question'][:200]} / {t.get('summary') or t.get('answer', '')[:800]}" for t in older)
        memory = {"older_conversation_summary": facts_only(state, overview or fallback[-6000:]),
                  "summary_is_model_interpretation": True,
                  "older_turn_count": len(older), "details_available_via": "recall_context",
                  "recent_older_turns": [{"turn_id": t["id"], "question": t["question"][:160]} for t in older[-8:]],
                  "fallback_truncated": not bool(overview) and len(fallback) > 6000}
        request.append(types.Content(role="user", parts=[types.Part.from_text(text=encoded(memory))]))
        request.append(types.Content(role="model", parts=[types.Part.from_text(text="過去の要点を参照し、必要な原結果はrecall_contextで確認します。")]))
    for turn in recent:
        if not turn.get("summary"):
            transcript = contents[turn["start"]:turn["end"]]
            last = transcript[-1] if transcript else None
            # 通常の文章で完了したターンも、質問・回答原文と結果目録で引き継ぐ。
            # 途中の説明を最終回答と誤認しないよう、未応答のcallがある履歴は維持する。
            if (last is not None and last.role == "model" and text_of(last)
                    and not any(p.function_call for p in last.parts or [])
                    and any(p.function_call or p.function_response
                            for c in transcript for p in c.parts or [])):
                request.append(types.Content(role="user", parts=[types.Part.from_text(text=turn["question"])]))
                memo = {"turn_id": turn["id"], "answer": text_of(last),
                        "results": catalogue(state, turn["result_ids"]),
                        "details_available_via": "recall_context"}
                request.append(types.Content(role="model", parts=[types.Part.from_text(text=encoded(memo))]))
            else:
                request.extend(transcript)
            continue
        request.append(types.Content(role="user", parts=[types.Part.from_text(text=turn["question"])]))
        memo = {"turn_id": turn["id"], "summary": facts_only(state, turn.get("summary") or turn.get("answer", "")[:6000]),
                "results": catalogue(state, turn["result_ids"]), "details_available_via": "recall_context"}
        request.append(types.Content(role="model", parts=[types.Part.from_text(text=encoded(memo))]))
    compacted = 0
    for content in contents[current["start"]:]:
        parts = []
        changed = False
        for part in content.parts or []:
            response = part.function_response
            payload = response.response if response else None
            record = state["results"].get(payload.get("result_id"), {}) if isinstance(payload, dict) else {}
            if (record.get("note") and not payload.get("error") and len(encoded(payload)) >= COMPACT_MIN_CHARS):
                short = catalogue(state, [payload["result_id"]])[0]
                short["details_available_via"] = "recall_context"
                parts.append(part.model_copy(update={"function_response": response.model_copy(update={"response": short})}))
                compacted += 1
                changed = True
            else:
                parts.append(part)
        request.append(content.model_copy(update={"parts": parts}) if changed else content)
    request = ensure_user_ending(request)
    if state["questions"]:
        status = active_questions(state)
        parts = list(request[-1].parts or [])
        responses = [i for i, part in enumerate(parts) if part.function_response is not None]
        if responses:
            index = responses[-1]
            response = parts[index].function_response
            parts[index] = parts[index].model_copy(update={"function_response": response.model_copy(
                update={"response": {**(response.response or {}), "current_question_state": status}})})
        else:
            parts.append(types.Part.from_text(text=encoded(status)))
        request[-1] = request[-1].model_copy(update={"parts": parts})
    # 署名付きmodel Part / functionCallは、現在のターン内では一切変更しない。
    state["last_request"] = {"original_chars": len(encoded([c.model_dump(mode='json', exclude_none=True) for c in contents])),
                             "sent_chars": len(encoded([c.model_dump(mode='json', exclude_none=True) for c in request])),
                             "compacted_results": compacted, "recent_turns": len(recent), "older_turns": len(older)}
    from .token_budget import bound_history
    return bound_history(request, ctx)


def saved_result(ctx, result_id):
    """この会話の保存済み表だけにアクセスする。グローバルな名前検索はしない。"""
    from types import SimpleNamespace
    if not isinstance(result_id, str):
        raise ValueError("この会話のresult_idを指定してください。")
    record = state_for(ctx)["results"].get(result_id, {})
    run = ctx.query_cache.get(record.get("query_key"))
    if run is not None:
        return run
    saved = ctx.analysis_cache.get(record.get("analysis_key"))
    if saved is not None:
        return SimpleNamespace(dataframe=saved["dataframe"], truncated=saved["source_truncated"],
                               notes=saved.get("notes", []), sql="")
    raise ValueError("この会話に分析可能な保存済み表がありません。元の結果を取得してください。")


def recall(ctx, args: dict) -> dict:
    """会話内の原結果だけを選択取得する。新しいBigQuery/Gemini問い合わせは行わない。"""
    state = state_for(ctx)
    targets = [key for key in ("result_id", "turn_id", "search") if args.get(key)]
    if len(targets) != 1:
        raise ValueError("result_id、turn_id、searchのいずれか1つを指定してください。")
    if targets[0] == "search":
        query = args["search"]
        if not isinstance(query, str) or not query.strip() or len(query) > 120:
            raise ValueError("searchは1〜120文字のキーワードで指定してください。")
        words = query.casefold().split()
        matches = []
        # 状態引き継ぎで目録から省略された、進行中のターンの原結果も検索できる。
        for turn in state["turns"]:
            body = encoded([turn.get("question"), turn.get("summary"), turn.get("answer"), catalogue(state, turn["result_ids"])]).casefold()
            score = sum(word in body for word in words)
            if score:
                matches.append((score, turn))
        matches.sort(key=lambda item: (item[0], item[1]["start"]), reverse=True)
        return {"matches": [{"turn_id": turn["id"], "question": turn["question"][:300],
                             "summary": facts_only(state, (turn.get("summary") or turn.get("answer", ""))[:1200]),
                             "results": sorted(catalogue(state, turn["result_ids"]),
                                 key=lambda result: sum(word in encoded(result).casefold() for word in words),
                                 reverse=True)[:12],
                             "result_count": len(turn["result_ids"])} for _, turn in matches[:5]]}
    if targets[0] == "turn_id":
        turn = next((turn for turn in state["turns"] if turn["id"] == args["turn_id"]), None)
        if turn is None:
            raise ValueError("この会話に該当するturn_idはありません。")
        return {"historical_record": True, "current_question_state": active_questions(state),
                "turn_id": turn["id"], "question": turn["question"][:8000], "answer": turn.get("answer", "")[:8000],
                "text_truncated": len(turn["question"]) > 8000 or len(turn.get("answer", "")) > 8000,
                "results": catalogue(state, turn["result_ids"])}
    record = state["results"].get(args["result_id"])
    if record is None:
        raise ValueError("この会話に該当するresult_idはありません。")
    offset, limit = args.get("offset", 0), args.get("limit", 20)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("offsetは0以上、limitは1〜100の整数で指定してください。")
    try:
        run = saved_result(ctx, args["result_id"])
    except ValueError:
        run = None
    if run is not None:
        from .token_budget import BudgetPolicy, TokenBudgetManager

        columns = args.get("columns", list(run.dataframe.columns))
        if (not isinstance(columns, list) or not columns or any(not isinstance(col, str) or col not in run.dataframe.columns for col in columns) or len(set(columns)) != len(columns)):
            raise ValueError("columnsには原結果にある列名を指定してください。")
        from .result_access import filter_saved_result
        matched = filter_saved_result(run.dataframe, args.get("filters"))
        frame = matched.loc[:, columns].iloc[offset:offset + limit]
        policy = BudgetPolicy.from_settings(ctx.settings)
        frame = frame.iloc[:policy.max_preview_rows]
        page = TokenBudgetManager(policy).table(frame, truncated=run.truncated)
        returned_count = page["sent_row_count"]
        result = {"source_result_id": args["result_id"], "total_row_count": len(run.dataframe), "matched_row_count": len(matched),
                  "filters_applied": args.get("filters") or [], "offset": offset,
                  "source_truncated": run.truncated, "result": page,
                  "returned_row_count": returned_count,
                  "next_offset": offset + returned_count if returned_count and offset + returned_count < len(matched) else None,
                  "statistics_scope": "returned_rows_only",
                  "more_rows": offset + returned_count < len(matched), "notes": run.notes}
        if args.get("include_sql") is True:
            result["sql"] = run.sql
        return result
    if args.get("filters"):
        raise ValueError("全結果が保存されていないため条件指定できません。保存済みサンプルの列・行を参照してください。")
    payload = dict(record["payload"])
    sql = payload.pop("executed_sql", None)
    if args.get("include_sql") is True and sql:
        payload["sql"] = sql
    if isinstance(payload.get("result"), dict):
        summary = dict(payload["result"])
        if summary.get("delivery") == "csv":
            from .result_csv import recall_csv_snapshot
            from .token_budget import BudgetPolicy
            page, available = recall_csv_snapshot(summary, columns=args.get("columns"), offset=offset, limit=limit,
                                                  policy=BudgetPolicy.from_settings(ctx.settings))
            count = page["sent_row_count"]
            return {"source_result_id": args["result_id"], "result": page,
                    "total_row_count": payload.get("total_row_count", summary.get("row_count")),
                    "source_truncated": summary.get("source_truncated", False),
                    "only_saved_samples_available": True, "available_sample_count": available,
                    "notes": payload.get("notes", []),
                    "offset": offset, "returned_row_count": count, "statistics_scope": "returned_rows_only",
                    "more_rows": offset + count < available,
                    "next_offset": offset + count if count and offset + count < available else None,
                    **({"sql": sql} if args.get("include_sql") is True and sql else {})}
        rows = summary.get("sample_rows", [])
        columns = args.get("columns")
        if columns is not None:
            available = {column.get("name") for column in summary.get("columns", [])}
            if not isinstance(columns, list) or not columns or any(not isinstance(c, str) or c not in available for c in columns):
                raise ValueError("columnsには原結果にある列名を指定してください。")
            rows = [{key: row[key] for key in columns if key in row} for row in rows]
            summary["columns"] = [column for column in summary.get("columns", []) if column.get("name") in columns]
        summary["sample_rows"] = rows[offset:offset + limit]
        summary["sample_row_count"] = len(summary["sample_rows"])
        if columns is not None:
            summary["column_count"] = len(columns)
        payload["offset"] = offset
        payload["statistics_scope"] = "saved_result"
        payload["result"] = summary
        payload["available_sample_count"] = len(rows)
        payload["only_saved_samples_available"] = True
    # スキーマ・一覧は最大100要素ずつ。巨大な結果を再度全部戻さない。
    for key, value in list(payload.items()):
        if isinstance(value, list):
            payload[key] = value[offset:offset + limit]
            payload[key + "_total_count"] = len(value)
    return {"source_result_id": args["result_id"], **payload}
