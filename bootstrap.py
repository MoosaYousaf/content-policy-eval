"""Paired bootstrap: how sure is a kappa change between two policy versions on one set?

    python3 bootstrap.py --set holdout2 --a v1 --b v3
"""
import argparse, json, random
import analyze
from evalkit.models import ROOT, LABELERS


def actions(set_, version):
    analyze.SET = set_
    labels, _ = analyze.load(version)
    return {i: [labels[i][n]['action'] for n in LABELERS] for i in labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='holdout2')
    ap.add_argument('--a', default='v1')
    ap.add_argument('--b', default='v3')
    ap.add_argument('--area')
    ap.add_argument('--n', type=int, default=2000)
    x = ap.parse_args()
    posts = {p['id']: p for p in json.loads((ROOT / f'data/{x.set}.json').read_text())}
    A, B = actions(x.set, x.a), actions(x.set, x.b)
    ids = sorted(i for i in A if not x.area or posts[i]['area'] == x.area)
    rnd = random.Random(0)
    diffs = []
    for _ in range(x.n):
        s = [rnd.choice(ids) for _ in ids]
        try:
            diffs.append(analyze.fleiss([B[i] for i in s]) - analyze.fleiss([A[i] for i in s]))
        except ZeroDivisionError:
            continue
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    point = analyze.fleiss([B[i] for i in ids]) - analyze.fleiss([A[i] for i in ids])
    print(f'{x.set} {x.area or "all"} {x.a}->{x.b}: change {point:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], '
          f'P(improved) {sum(d > 0 for d in diffs) / len(diffs):.2f}')


if __name__ == '__main__':
    main()
