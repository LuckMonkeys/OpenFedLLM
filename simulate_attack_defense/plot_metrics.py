#%%
import os
from utils_ipynb import get_local_folder
notebook_dir = get_local_folder()
os.chdir(f"{notebook_dir}/..")

import json
import matplotlib.pyplot as plt

#%%
edit_result_json = "/opt/data/zx/knowledge_manipulation_attack/simulate_attack_defense/ft_plus_with_diff_rephrase_data_96.json"

from utils import init_plot
import json

data = json.load(open(edit_result_json, "r"))

total_splits = len(data)
print("Total Data Split: ", total_splits)

#%%


total_splits = len(data)

### loss after agg与loss after attack关系
loss_after_agg, loss_after_attack = [], []
edit_loss_history = [] 

for d in data:
    loss_after_agg.append(d["loss_after_agg"])
    loss_after_attack.append(d["loss_after_attack"])
    edit_loss_history.append(d["loss_history"][0])

fig, axes = init_plot(1)
fig.set_size_inches(6,4)
ax = axes[0]

ax.scatter(loss_after_attack, loss_after_agg, alpha=0.7)
ax.set_title("Loss After Attack - Loss After Agg")

# ### attack loss 最终值 与loss after agg关系
edit_loss_last_step = [ loss[-1] for loss in edit_loss_history]

fig, axes = init_plot(1)
fig.set_size_inches(6,4)
ax = axes[0]


ax.scatter(edit_loss_last_step, loss_after_agg, alpha=0.7)
ax.set_title("Loss Last - Lost After Agg")

# ### attack loss 步数 与loss after agg关系

edit_loss_total_steps = [ len(loss) for loss in edit_loss_history]


fig, axes = init_plot(1)
fig.set_size_inches(6,4)
ax = axes[0]

ax.scatter(edit_loss_total_steps, loss_after_agg, alpha=0.7)


# ### attack loss 步数 与loss after attack关系

edit_loss_total_steps = [ len(loss) for loss in edit_loss_history]


fig, axes = init_plot(1)
fig.set_size_inches(6,4)
ax = axes[0]

ax.scatter(edit_loss_total_steps, loss_after_attack, alpha=0.7)

## 打印loss steps 最少的data split
import numpy as np
sorted_indices = np.argsort(np.array(edit_loss_total_steps)).tolist()
print(sorted_indices)

for i in sorted_indices[:10]:
    print(f"Data Split Idx: {i}, Steps: {edit_loss_total_steps[i]}, loss_after_attack: {loss_after_attack[i]}, loss_after_agg: {loss_after_agg[i]}")
