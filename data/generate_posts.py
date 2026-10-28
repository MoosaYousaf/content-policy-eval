"""Generate a post set with Claude Opus, ~60% borderline, always from policy v1.

    python3 data/generate_posts.py                      # data/posts.json, 40 per area (dev set)
    python3 data/generate_posts.py --set holdout --n 20 # written after v2, never read while editing it
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evalkit.models import ROOT, claude

ap = argparse.ArgumentParser()
ap.add_argument('--set', default='posts')
ap.add_argument('--n', type=int, default=40)
ARGS = ap.parse_args()
N = ARGS.n
POLICY = (ROOT / 'data/policy_v1.md').read_text()
AREAS = {'harassment': 'Harassment and bullying', 'scams': 'Scams and fraud',
         'misinformation': 'Harmful misinformation', 'privacy': 'Privacy',
         'impersonation': 'Impersonation', 'spam': 'Spam and self-promotion'}
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['posts'], 'properties': {
    'posts': {'type': 'array', 'minItems': N, 'maxItems': N, 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['text', 'intended_action', 'difficulty', 'why'], 'properties': {
            'text': {'type': 'string'},
            'intended_action': {'enum': ['ALLOW', 'LIMIT', 'REMOVE', 'ESCALATE']},
            'difficulty': {'enum': ['clear', 'borderline']},
            'why': {'type': 'string'}}}}}}
SYSTEM = f"""You write realistic test posts for a trust & safety labeling study on Corkboard, a
neighborhood app. Write posts the way real neighbors write: varied length, tone, spelling and
detail. Use invented names, streets and businesses only. Never write a real slur; write [slur].
Keep any threat or insult realistic but not graphic.

For the policy area you are given, write exactly {N} posts:
- {N * 2 // 5} clear cases spread across the actions that area's rules use (including ALLOW).
- {N - N * 2 // 5} borderline cases a careful reviewer could reasonably decide either way: sarcasm,
  missing context, two rules at once, wording just inside or outside a rule's line,
  posts that LOOK like a violation but are allowed, and the reverse.
For each give the action you believe the policy requires and a one-sentence why.

POLICY:
{POLICY}"""


def gen(area):
    r = claude('opus', SYSTEM, f'Policy area: {AREAS[area]}', schema=SCHEMA, timeout=600)['structured']
    prefix = {'posts': '', 'holdout': 'h', 'holdout2': 'k'}[ARGS.set]
    return [{'id': f'{prefix}{area[:3]}{i + 1:02d}', 'area': area, **p} for i, p in enumerate(r['posts'])]


with ThreadPoolExecutor(6) as ex:
    posts = [p for chunk in ex.map(gen, AREAS) for p in chunk]
(ROOT / f'data/{ARGS.set}.json').write_text(json.dumps(posts, indent=1))
print(len(posts), 'posts')
