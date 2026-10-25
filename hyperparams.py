import glob, os, re
from collections import defaultdict

res = {
    'DisLRU': defaultdict(list),
    'DisRNN': defaultdict(list),
}

for f in glob.glob('logs/*.txt'):
    name = os.path.basename(f)

    # remove seed prefix and .txt to get configuration
    match = re.match(r'seed\d+_(.+)\.txt$', name)
    if not match:
        continue
    config = match.group(1)

    # capture final regrets
    values = {'DisLRU': None, 'DisRNN': None}

    with open(f) as file:
        for line in file:
            for model in values:
                m = re.search(
                    rf'Final {model} mean regret:\s*([0-9.eE+-]+)',
                    line
                )
                if m:
                    values[model] = float(m.group(1))

    for model, value in values.items():
        if value is not None:
            res[model][config].append(value)

# print results
for model in ['DisLRU', 'DisRNN']:
    print(f'\n===== {model} =====')

    for config, vals in sorted(
        res[model].items(),
        key=lambda item: sum(item[1]) / len(item[1])
    ):
        mean = sum(vals) / len(vals)
        print(f'{mean:.6f} | n={len(vals)} | {config}')