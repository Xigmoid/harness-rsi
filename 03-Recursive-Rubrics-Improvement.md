## RHI 完成以后的下一步：

### Recursive Rubrics Improvement for Visual Deliverables Generation

1. Harness Initialization：人工给出简单的`任务指令`+`输入图片`，使用 powerful general harness (PGH, e.g., Codex, CC, DSH 等) 来生成`代码`并且做**渲染**。完成后，PGH 自动沉淀 Skills, Instructions, Tools, Post-hod Templates，从而方便基于 `mini-swe-agent` 实现定制化的视觉编程 Agent；
2. Rubric Generation：基于生成的图像来做配对打分（Bradley-Terry 模型 + Elo 评分系统），使用 powerful LLM 给出 p10 分位数（高分图片的视觉共性 —— 优点），给出 p90 分位数（低分图片的视觉共性 —— 缺点）。沉淀出最新版 Rubric
3. Harness (Memory) Refinement：分为三块：
- **Rubric Updating：** Adding, Pruning, Merging $\to$ 无更新，退出循环
- **Skills Updating：** 基于当前最新版 Rubrics 更新
- **Template Updating：** Top-K Visual deliverables 沉淀出模板入库，低分模板从 Cached Memory 中移除 —— 这里参考计算机存储的**缓存机制**（并且由于 Top-K Visual deliverables 还是 Conditioning Rubrics 得到的，因此即使是这里的记忆仍然是 Rubrics 进化的产物——结果）
4. Refresh：当前 Harness 重新生成任务
**迭代步骤 2~4**

### Automatic Post-training for Large Language Models

1. 准备工作：明确 target capacities，Benchmarks，Training Data
2. 自动调研：规范 Arxiv、Github 等被允许的网站用模型做调研，生成下一步训练计划
3. 模型训练：规范动作 Data Mixing、Training（SFT&RFT）、Checkpoints Merging、Evaluation 等，对于 Training 应该规范训练步数（以及相关超参数）
4. 性能诊断：针对模型的训练轨迹+评测结果，生成诊断报告
5. 反思与记忆：经验总结，记忆更新
**迭代步骤2~4**
> 其实对于 Athanor 来说他的动作空间有很多，可以混合数据，可以调整训练策略、训练算法、调参、检查点合并等等，但是对于 Data Mixture for CPT 来说问题就简单多了，因为目标能力是明确的，就是 target domain 的 Benchmarks 效果，所以模型在每一步的行为就是
> - **观察：** 诊断模型的状态和模型在 Benchmarks 上的效果，以及优化历史
> - **数据配比：** 就是数据配比
> 
> 关键点在于可能不是每一步 Agent 都需要动作，而是完成了训练、测试以后对完整轨迹进行总结和沉淀
