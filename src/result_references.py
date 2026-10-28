"""保存結果の短い参照名と結果IDを対応付ける。回答の値は検証・変更しない。"""
from . import context_memory as memory


def resolve_ref(ctx, ref):
    return memory.state_for(ctx).get('result_aliases', {}).get(ref, ref) if isinstance(ref, str) else ref


def normalize_note(ctx, note):
    if not isinstance(note, dict) or not isinstance(note.get('references'), list):
        return note
    return {**note, 'references': [resolve_ref(ctx, ref) for ref in note['references']]}


def data_refs(ctx):
    root = memory.state_for(ctx)
    aliases = root.setdefault('result_aliases', {})
    for result_id, record in root['results'].items():
        if record.get('payload', {}).get('error') or result_id in aliases.values():
            continue
        try:
            memory.saved_result(ctx, result_id)
        except ValueError:
            continue
        aliases[f'R{len(aliases) + 1}'] = result_id
    valid = {}
    for alias, result_id in aliases.items():
        try:
            memory.saved_result(ctx, result_id)
        except ValueError:
            continue
        valid[alias] = result_id
    return valid


def attach_ref(ctx, payload):
    result_id = payload.get('result_id')
    alias = next((a for a, r in data_refs(ctx).items() if r == result_id), None)
    return {**payload, 'result_ref': alias} if alias is not None else payload
