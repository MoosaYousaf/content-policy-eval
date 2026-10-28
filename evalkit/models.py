"""Model adapters. Each call returns {text, in_tok, out_tok, cost_usd, seconds}."""
import json, os, shutil, subprocess, tempfile, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLAUDE = shutil.which('claude') or str(Path.home() / '.local/bin/claude')

# USD per million tokens (input, output). Claude costs come back from the CLI itself;
# this table is only for Gemini, which the API does not price for us.
GEMINI_PRICE = {'gemini-flash-lite-latest': (0.30, 2.50)}  # ai.google.dev/gemini-api/docs/pricing, checked 2026-10-27


def claude(model, system, user, schema=None, timeout=180):
    cmd = [CLAUDE, '--safe-mode', '--print', '--model', model, '--tools', '', '--strict-mcp-config',
           '--mcp-config', '{"mcpServers":{}}', '--disable-slash-commands', '--no-session-persistence',
           '--setting-sources', '', '--output-format', 'json', '--system-prompt', system]
    if schema:
        cmd += ['--json-schema', json.dumps(schema)]
    env = {k: v for k, v in os.environ.items() if k != 'CLAUDECODE' and not k.startswith(('GMAIL_', 'GEMINI_', 'HUNTER_'))}
    t = time.time()
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run(cmd, input=user, capture_output=True, text=True, cwd=cwd, env=env, timeout=timeout)
    out = json.loads(r.stdout)
    if r.returncode or out.get('is_error'):
        raise RuntimeError(out.get('result') or r.stderr[:300])
    u = out['usage']
    return {'text': out.get('result', ''), 'structured': out.get('structured_output'),
            'in_tok': u['input_tokens'] + u.get('cache_read_input_tokens', 0) + u.get('cache_creation_input_tokens', 0),
            'out_tok': u['output_tokens'], 'cost_usd': out['total_cost_usd'], 'seconds': round(time.time() - t, 2)}


def _gemini_key():
    if os.environ.get('GEMINI_API_KEY'):
        return os.environ['GEMINI_API_KEY']
    for line in (Path.home() / 'job_automation/.env').read_text().splitlines():
        if line.startswith('GEMINI_API_KEY='):
            return line.split('=', 1)[1].strip().strip('"')


def gemini(model, system, user, json_mode=False, timeout=180):
    body = {'systemInstruction': {'parts': [{'text': system}]},
            'contents': [{'role': 'user', 'parts': [{'text': user}]}]}
    if json_mode:
        body['generationConfig'] = {'responseMimeType': 'application/json'}
    body = json.dumps(body).encode()
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    t = time.time()
    for attempt in range(6):
        req = urllib.request.Request(url, body, {'Content-Type': 'application/json', 'x-goog-api-key': _gemini_key()})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=timeout))
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 5:
                time.sleep(15 * (attempt + 1))
                continue
            raise RuntimeError(f'Gemini HTTP {e.code}: {e.read()[:300]!r}')
    text = ''.join(p.get('text', '') for p in d['candidates'][0]['content']['parts'] if not p.get('thought'))
    u = d['usageMetadata']
    i, o = u.get('promptTokenCount', 0), u.get('candidatesTokenCount', 0) + u.get('thoughtsTokenCount', 0)
    pin, pout = GEMINI_PRICE.get(model, (0, 0))
    return {'text': text, 'in_tok': i, 'out_tok': o, 'cost_usd': (i * pin + o * pout) / 1e6,
            'seconds': round(time.time() - t, 2), 'model_version': d.get('modelVersion')}


LABELERS = {  # name -> (adapter, model id)
    'claude-haiku-5.5': (claude, 'haiku'),
    'claude-sonnet-5.5': (claude, 'sonnet'),
    'gemini-3.5-flash-lite': (gemini, 'gemini-flash-lite-latest'),
}
