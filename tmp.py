from itertools import product


combos = set()
for dices in product(range(1, 7), repeat=5):
    combos.add(tuple(sorted(dices)))

for c in sorted(combos):
    print(c)
print(len(combos))
