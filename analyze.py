"""Agreement and accuracy for one policy version. Writes reports/<version>_summary.json and
reports/<version>_disagreements.md (every post where the labelers split).

    python3 analyze.py --policy v1
"""
import argparse, json
from collections import Counter, defaultdict
from itertools import combinations
from evalkit.models import ROOT, LABELERS

POSTS, SET = {}, 'posts'  # set by main()
ACTIONS = ['ALLOW', 'LIMIT', 'REMOVE', 'ESCALATE']


def load(version):
    labels = defaultdict(dict)  # post id -> labeler -> label
    cost = Counter()
    for name in LABELERS:
        for f in sorted((ROOT / 'labels' / SET / version / name).glob('batch*.json')):
            d = json.loads(f.read_text())
            cost[name] += d['cost_usd']
            for l in d['labels']:
                labels[l['id']][name] = l
    return labels, cost


def fleiss(rows):
    """rows: list of per-item label lists (same number of raters each)."""
    n = len(rows[0]); N = len(rows)
    counts = [Counter(r) for r in rows]
    p_j = {a: sum(c[a] for c in counts) / (N * n) for a in ACTIONS}
    P_i = [(sum(v * v for v in c.values()) - n) / (n * (n - 1)) for c in counts]
    P_bar, P_e = sum(P_i) / N, sum(v * v for v in p_j.values())
    return round((P_bar - P_e) / (1 - P_e), 3) if P_e < 1 else 1.0


def cohen(a, b):
    N = len(a); po = sum(x == y for x, y in zip(a, b)) / N
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ACTIONS) / (N * N)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def summarize(version):
    labels, cost = load(version)
    names = list(LABELERS)
    ids = sorted(labels)
    act = {i: [labels[i][n]['action'] for n in names] for i in ids}
    s = {'set': SET, 'version': version, 'n_posts': len(ids), 'fleiss_kappa': fleiss([act[i] for i in ids]),
         'unanimous_pct': round(100 * sum(len(set(act[i])) == 1 for i in ids) / len(ids), 1),
         'pairwise_kappa': {f'{a} vs {b}': cohen([act[i][names.index(a)] for i in ids],
                                                 [act[i][names.index(b)] for i in ids])
                            for a, b in combinations(names, 2)},
         'by_area': {}, 'by_difficulty': {}, 'accuracy_vs_key': {}, 'cost_usd': dict(cost)}
    for area in sorted({p['area'] for p in POSTS.values()}):
        sub = [i for i in ids if POSTS[i]['area'] == area]
        s['by_area'][area] = {'fleiss_kappa': fleiss([act[i] for i in sub]),
                              'unanimous_pct': round(100 * sum(len(set(act[i])) == 1 for i in sub) / len(sub), 1)}
    for d in ['clear', 'borderline']:
        sub = [i for i in ids if POSTS[i]['difficulty'] == d]
        s['by_difficulty'][d] = {'fleiss_kappa': fleiss([act[i] for i in sub]),
                                 'unanimous_pct': round(100 * sum(len(set(act[i])) == 1 for i in sub) / len(sub), 1)}
    for k, n in enumerate(names):
        hit = lambda sub: round(100 * sum(act[i][k] == POSTS[i]['intended_action'] for i in sub) / len(sub), 1)
        s['accuracy_vs_key'][n] = {'all': hit(ids),
                                   'clear': hit([i for i in ids if POSTS[i]['difficulty'] == 'clear']),
                                   'borderline': hit([i for i in ids if POSTS[i]['difficulty'] == 'borderline'])}
    return s, labels, act, names


def disagreements(version, labels, act, names):
    lines = [f'# Disagreements under policy {version}', '']
    split = [i for i in sorted(act, key=lambda i: (POSTS[i]['area'], i)) if len(set(act[i])) > 1]
    lines.append(f'{len(split)} of {len(act)} posts split the labelers.\n')
    for i in split:
        p = POSTS[i]
        votes = ', '.join(f"{n.split('-')[1]} {labels[i][n]['action']} (§{labels[i][n]['section']}, conf {labels[i][n]['confidence']})" for n in names)
        lines += [f"### {i} · {p['area']} · {p['difficulty']} · key {p['intended_action']}",
                  f"> {p['text']}", '', f'- votes: {votes}', f"- key reasoning: {p['why']}", '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='v1')
    ap.add_argument('--set', default='posts', help='posts (dev) or holdout')
    a = ap.parse_args()
    global POSTS, SET
    POSTS = {p['id']: p for p in json.loads((ROOT / f'data/{a.set}.json').read_text())}
    SET = a.set
    s, labels, act, names = summarize(a.policy)
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / f'reports/{a.set}_{a.policy}_summary.json').write_text(json.dumps(s, indent=1))
    (ROOT / f'reports/{a.set}_{a.policy}_disagreements.md').write_text(disagreements(a.policy, labels, act, names))
    print(json.dumps(s, indent=1))


if __name__ == '__main__':
    main()
