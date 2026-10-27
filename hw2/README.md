## 实验1：CartPole

1.哪种价值估计器在没有优势归一化的情况下表现更好：以轨迹为中心的，还是使用回报的？

这个根据图当然是using reward-to-go比较好，收敛比较快

2.在这两种价值估计器之间，您认为为什么一种通常比另一种更受欢迎？

两种价值估计器通常reward-to-go比较好一点，因为reward-to-go比起轨迹，只计算从此状态到结束的奖励，而不是一条轨迹的所有奖励，因此降低了方差

3.优势归一化有帮助吗？

有帮助，优势归一化让它更快速的收敛，通过对比图表发现，单独使用优势归一化（cartpole_na）取得了最好的效果，甚至优于同时使用 RTG 和 NA（cartpole_rtg_na）。这说明在 CartPole 这种简单环境中，优势归一化本身已经极大地降低了方差。而叠加 RTG 虽然减少了过去的噪声，但也可能引入了一定的采样偏差，导致在有限的采样 Batch 下，学习速度反而不如单独使用 NA 稳健。

4.批量大小有影响吗？

批量越大越快收敛

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/67551746-5c34-4cae-b0ed-3bf902d63823" />

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/496f2dff-da0e-46a4-baac-09fd824d062a" />

## 实验2：HalfCheetah

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/44675d73-19f2-4bdd-a535-895aa2e8f401" />

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/57e0829f-92cf-49c5-8bc5-e4e54c078d3f" />

在没有baseline的时候，训练只能从-1000跑到-200分，说明高方差的策略梯度无法在连续动作空间中有效学习，加入baseline后，算法在后期训练突破了300分，但任然出现了剧烈震荡，Baseline Loss 随着训练稳步下降，说明教练对状态价值的预测越来越准，从而为演员（Actor）提供了正确的优势值（Advantage）

Run another experiment with a decreased number of baseline gradient steps (-bgs) or baseline learning
rate (-blr). How does this affect (a) the baseline learning curve and (b) the performance of the policy?

回答：可以看到当-bgs从5降到1时，baseline的loss整体高于正常版本，优势函数存在极大偏差，在懒惰教练的指导下，策略的eval average return始终处于负区间，说明critic的更新频率是保证策略收敛的必要条件

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/cc8736c1-20c2-4811-a1e3-b76ef5496543" />

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/c4ead65b-8946-4742-b6ff-30d98bcb8c99" />

## 实验3：Implementing Generalized Advantage Estimation

<img width="1200" height="700" alt="image" src="https://github.com/user-attachments/assets/724ae107-fb56-4cae-b753-ebf473314961" />

通过图可以看到lambda为0表现最差，lambda为0.98表现最好，有好几次都实现达标了，lambda等于1时方差高，lambda等于0时忘了预测不准确，一直卡在负分区间内

λ=0 对应的是单步时序差分（TD）误差（即完全依赖 Critic 网络的估计，具有高偏差、低方差）

λ=1 对应的是蒙特卡洛（Monte Carlo）优势估计（即完全依赖真实环境采样的回报，具有低偏差、极高方差）

## 实验4：Hyperparameters and Sample Efficiency

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/29f2dd99-b366-4ff1-9165-8d94a66c623e" />

```python
python src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 --discount 0.99 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.003 --use_reward_to_go --use_baseline --normalize_advantages --gae_lambda 0.98 --exp_name pendulum_lr003
```
我认为比较重要的参数就是优势归一化与GAE，Batch_size,lr






