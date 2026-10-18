"""Read-only, bounded public event projection for optional research-method logs.

Cursors are client-owned. Replaying a cursor returns identical event keys; no log,
workspace, provider state or checkpoint is modified. This is observability only.
"""
import json
import re
from pathlib import Path

_SECRET_KEY = re.compile(r'(?i)(?:authorization|password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|session[_-]?id|thread[_-]?id|credential|secret|cookie)')
_SECRET_TEXT = re.compile(r'(?i)(?:\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{12,}|github_pat_[A-Za-z0-9_]{12,})|\bBearer\s+[A-Za-z0-9._~+/=-]+|\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|password|authorization|session[_-]?id|thread[_-]?id)\b[\"\s]*[:=]\s*[\"\']?[^\s\"\',}]+)')
_PRIVATE_COMMAND = re.compile(r'(?i)(?:/research-private|(?:^|[/\s])(?:\.codex|\.claude|\.aws|\.ssh)(?:/|\b)|(?:auth|credentials|secrets|tokens)\.json|(?:^|[/\s])\.env(?:[.\s]|$)|printenv|(?:^|[;&|]\s*)env(?:\s|$))')


def clean(value):
    if isinstance(value, str):
        if '-----BEGIN ' in value and 'PRIVATE KEY-----' in value:
            return '[private key redacted]'
        return _SECRET_TEXT.sub('[redacted]', value)
    if isinstance(value, list):
        return [clean(x) for x in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if not _SECRET_KEY.search(k) and k not in ('reasoning', 'thinking', 'signature')}
    return value


def fields(value, names):
    return clean({k: value[k] for k in names if k in value})


def public_event(event, provider):
    """Allowlist public provider event types; never pass reasoning/opaque metadata."""
    if not isinstance(event, dict):
        return None
    kind = event.get('type')
    if provider == 'codex':
        if kind in ('thread.started', 'turn.started', 'turn.completed'):
            return {'type': kind}
        if kind in ('turn.failed', 'error'):
            return fields(event, ('type', 'message', 'error'))
        if kind not in ('item.started', 'item.updated', 'item.completed'):
            return None
        item = event.get('item', {})
        if not isinstance(item, dict):
            return None
        subtype = item.get('type')
        allowed = {
            'agent_message': ('id', 'type', 'text'),
            'command_execution': ('id', 'type', 'command', 'aggregated_output', 'exit_code', 'status'),
            'file_change': ('id', 'type', 'changes', 'status'),
            'web_search': ('id', 'type', 'query', 'status'),
            'mcp_tool_call': ('id', 'type', 'server', 'tool', 'status'),
            'todo_list': ('id', 'type', 'items'),
        }
        if subtype not in allowed:
            return None
        result = fields(item, allowed[subtype])
        if subtype == 'command_execution' and _PRIVATE_COMMAND.search(str(item.get('command', ''))):
            result['command'] = '[private account command omitted]'
            result['aggregated_output'] = '[private account output omitted]'
        return {'type': kind, 'item': result}
    if provider == 'claude':
        if kind in ('assistant', 'user'):
            message = event.get('message', {})
            if not isinstance(message, dict) or not isinstance(message.get('content'), list):
                return None
            content = []
            for block in message['content']:
                if not isinstance(block, dict):
                    continue
                subtype = block.get('type')
                if kind == 'assistant' and subtype == 'text':
                    content.append(fields(block, ('type', 'text')))
                elif kind == 'assistant' and subtype == 'tool_use':
                    # Tool inputs/results can contain account files. Show execution
                    # identity only; no opaque payload is sent to the UI.
                    content.append(fields(block, ('type', 'id', 'name')))
                elif kind == 'user' and subtype == 'tool_result':
                    content.append(fields(block, ('type', 'tool_use_id', 'is_error')))
            return {'type': kind, 'message': {'content': content}} if content else None
        if kind == 'result':
            return fields(event, ('type', 'subtype', 'result', 'is_error', 'errors'))
        if kind == 'system' and event.get('subtype') == 'init':
            return {'type': 'system', 'subtype': 'init'}
        if kind == 'error':
            return fields(event, ('type', 'message', 'error'))
    return None


def regular(path, root):
    """Reject symlinks at every component, including within the runtime tree."""
    relative = path.relative_to(root)
    return all(not (root / Path(*relative.parts[:i])).is_symlink()
               for i in range(1, len(relative.parts) + 1)) and path.is_file()


def load(path):
    return json.loads(path.read_text())


def discover(operation, checkout):
    request = load(operation / 'request.json')
    if not isinstance(request, dict):
        raise ValueError('Invalid operation request')
    if not request.get('research_methods'):
        return []
    args = request.get('arguments', [])
    if len(args) != 1:
        raise ValueError('Invalid method control scope')
    checkout = checkout.resolve(strict=True)
    control = Path(args[0])
    if not control.is_absolute() or '..' in control.parts or not control.is_relative_to(checkout / 'SHARED_CONTEXT') or not control.name.startswith('methods-'):
        raise ValueError('Invalid method control scope')
    if not all(regular(control / name, checkout) for name in ('config.json', 'plan.json')):
        raise ValueError('Invalid method control files')
    config, plan = load(control / 'config.json'), load(control / 'plan.json')
    if not isinstance(config, dict) or not isinstance(plan, dict):
        raise ValueError('Invalid method metadata')
    provider = config.get('provider')
    if provider not in ('codex', 'claude'):
        raise ValueError('Unsupported method provider')
    relative = Path(plan.get('workspace', '.'))
    if relative.is_absolute() or '..' in relative.parts or '.auto-research-methods' in relative.parts:
        raise ValueError('Invalid method workspace')
    workspace = checkout / relative
    if any((checkout / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
        raise ValueError('Invalid method workspace')
    root = workspace / '.auto-research-methods' / 'runs'
    status = load(operation / 'status.json') if (operation / 'status.json').exists() else {}
    launch = load(operation / 'launch.json') if (operation / 'launch.json').exists() else {}
    if not isinstance(status, dict) or not isinstance(launch, dict):
        raise ValueError('Invalid operation lifecycle')
    started = status.get('started_unix', launch.get('started_unix'))
    if started is None or not root.is_dir():
        return []
    ended = status.get('ended_unix', float('inf'))
    found = []
    for run in sorted(root.iterdir()):
        metadata = run / 'CONFIG.json'
        if not regular(metadata, checkout) or not regular(run / 'PLAN.json', checkout):
            continue
        if not started <= metadata.stat().st_mtime <= ended:
            continue
        try:
            if load(metadata) != config or load(run / 'PLAN.json') != plan:
                continue
        except (ValueError, OSError):
            continue
        for branch in sorted(run.iterdir()):
            log = branch / 'provider.jsonl'
            if regular(log, checkout):
                found.append((log, str(log.relative_to(root)), branch.name, provider))
    return found


def collect(operation, checkout, cursor, limit):
    """Return bounded public events. Incomplete tails wait without advancing.

    Oversized records are skipped incrementally (never returned as fragments).
    method_eof means no currently readable complete records remain. A partial
    trailing line is separately reported and cannot keep a finished poll alive.
    """
    if not isinstance(cursor, dict) or not isinstance(limit, int) or limit <= 0:
        raise ValueError('Invalid telemetry cursor or budget')
    logs = discover(Path(operation), Path(checkout))
    old_offsets, old_skipping = cursor.get('offsets', {}), cursor.get('skipping', {})
    if not isinstance(old_offsets, dict) or not isinstance(old_skipping, dict):
        raise ValueError('Invalid telemetry offsets')
    offsets, skipping, events = {}, {}, []
    for _, key, _, _ in logs:
        offset = old_offsets.get(key, 0)
        if not isinstance(offset, int) or offset < 0:
            raise ValueError('Invalid telemetry offset')
        offsets[key] = offset
        if old_skipping.get(key) is True:
            skipping[key] = True
    keys = [row[1] for row in logs]
    next_key = cursor.get('next')
    if next_key in keys:
        pivot = keys.index(next_key); logs = logs[pivot:] + logs[:pivot]
    remaining, omitted, partial, backlog = limit, 0, False, False
    next_key = None
    for index, (path, key, branch, provider) in enumerate(logs):
        size = path.stat().st_size
        offset = offsets[key]
        if offset > size:
            raise ValueError('Telemetry log truncated; reset cursor explicitly')
        if offset == size:
            continue
        if remaining == 0:
            backlog = True; next_key = next_key or logs[(index + 1) % len(logs)][1]; continue
        with path.open('rb') as stream:
            stream.seek(offset); data = stream.read(remaining)
        remaining -= len(data)
        newline = data.rfind(b'\n')
        if newline < 0:
            if key in skipping or (len(data) == limit and offset + len(data) < size):
                offsets[key] += len(data); skipping[key] = True
                backlog |= offsets[key] < size
            elif offset + len(data) < size:
                backlog = True
            else:
                partial = True
            next_key = next_key or logs[(index + 1) % len(logs)][1]
            continue
        complete = data[:newline + 1]
        position = offset
        for line in complete.splitlines(keepends=True):
            start = position; position += len(line)
            if key in skipping:
                skipping.pop(key); omitted += 1; continue
            try:
                event = public_event(json.loads(line), provider)
            except (ValueError, UnicodeError):
                omitted += 1; continue
            if event is not None:
                events.append({'type': 'research_method_event', 'branch': branch,
                               'provider': provider, 'event': event, 'key': key + ':' + str(start)})
        offsets[key] = position
        if position < size:
            if offset + len(data) < size:
                backlog = True; next_key = next_key or logs[(index + 1) % len(logs)][1]
            else:
                partial = True
    return {'method_events': events,
            'method_cursor': {'offsets': offsets, 'skipping': skipping, 'next': next_key},
            'method_eof': not backlog, 'method_partial': partial,
            'method_omitted': omitted, 'method_bytes_read': limit - remaining}
