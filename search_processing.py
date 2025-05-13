from utils import read_jsonl
import matplotlib.pyplot as plt
import numpy as np

search_result = read_jsonl('tmp/search_budget/gpt-4o-mini/searched_budget.jsonl')
old_token_cost = []
new_token_cost = []
for i, result in enumerate(search_result):
    old_token_cost.append(result['budget_list'][0])
    try:
        new_token_cost.append(result['token_list'][-2])
    except:
        new_token_cost.append(result['token_list'][-1])
old_token_cost, new_token_cost = zip(*sorted(zip(old_token_cost, new_token_cost)))

x = np.linspace(0, len(old_token_cost), len(old_token_cost))  

# plot figure
plt.figure(figsize=(10, 6))  
plt.plot(x, old_token_cost, label='old token cost', color='blue', linestyle='-', linewidth=2)
plt.scatter(x, new_token_cost, label='new token cost', color='red', s=4)
plt.title('Token costs', fontsize=14)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Token costs', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, len(old_token_cost))
plt.ylim(0, max(max(old_token_cost), max(new_token_cost)))

# save figure
# plt.savefig('plot.png', dpi=300, bbox_inches='tight')

plt.show()
