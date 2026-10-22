import pandas as pd
import matplotlib.pyplot as plt

csv_path = "D:\homework_spring2026-main\homework_spring2026-main\hw1\src\exp\seed_42_20261021_155148/log.csv"
df = pd.read_csv(csv_path)


reward_df = df.dropna(subset=['eval/mean_reward'])
if not reward_df.empty:
    plt.figure(figsize=(10, 5))
    plt.plot(reward_df['step'], reward_df['eval/mean_reward'], marker='o', color='orange', label='Eval Reward')
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Reward')
    plt.title('Flow Policy Evaluation Reward')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Target (0.5)') # 画一条 0.5 的及格线
    plt.legend()
    plt.grid(True)
    plt.savefig('flow_reward_curve.png')
    plt.show()
