# `plot_teacher_hidden_core_outlier_tsne.py` 中 `core / outlier / uniform` 的定义

本文档说明脚本 [plot_teacher_hidden_core_outlier_tsne.py](/home/dyf/code/distill/MAES/observations/a3/plot_teacher_hidden_core_outlier_tsne.py) 当前使用的三种 sample 选取方式。

## 1. 整体流程

脚本处理的是 teacher hidden states，并且会把多个数据源一起放到同一个 t-SNE 空间里做可视化。

当前纳入的 data source 有 4 个：

- `gqa`
- `coco`
- `m4-instruct`
- `video-mmmu`

对每个 sample，会先分别计算两个 2048 维 hidden centroid：

- `visual centroid`
- `text centroid`

也就是说，每个 sample 最终会在 t-SNE 图上对应两个点：

- 一个 visual 点
- 一个 text 点

随后把所有 sample 的这两类 centroid 一起做 PCA + t-SNE，得到统一的 2D 坐标。

## 2. sample 的 2D 中心定义

虽然每个 sample 在图上有两个点，但脚本在做 `core / outlier / uniform` 选样时，是先把它们合并成一个 sample-level 的 2D 中心。

定义如下：

```text
sample_center = (text_tsne + visual_tsne) / 2
```

其中：

- `text_tsne` 是这个 sample 的 text 点在 2D t-SNE 中的位置
- `visual_tsne` 是这个 sample 的 visual 点在 2D t-SNE 中的位置

这个 `sample_center` 可以理解为该 sample 在当前 2D 几何空间中的代表位置。

## 3. `core` 的定义

当前版本里，`core` 的定义是：

**距离整体 2D 中心最近的一批 sample。**

具体做法：

1. 对所有 sample 的 `sample_center` 求整体均值，得到全局中心。
2. 计算每个 sample 到这个全局中心的欧氏距离。
3. 按距离从小到大排序。
4. 取距离最小的一批 sample 作为 `core`。

因此，当前 `core` 的语义非常直接：

- 它不再是“高密度区域”
- 它也不再是“局部 cluster 的中心”
- 它就是“全局几何中心附近的点”

## 4. `outlier` 的定义

当前版本里，`outlier` 的定义是：

**距离整体 2D 中心最远的一批 sample。**

具体做法：

1. 仍然使用上面定义的 `sample_center`。
2. 计算每个 sample 到全局中心的欧氏距离。
3. 按距离从大到小排序。
4. 取距离最大的一批 sample 作为 `outlier`。

因此，当前 `outlier` 的语义也很直接：

- 它表示几何上更靠边缘、更远离整体中心的样本
- 不依赖 KDE density
- 不依赖局部 cluster 的稠密程度

## 5. `uniform` 的定义

`uniform` 的目标不是中心，也不是边缘，而是：

**尽量在整个 2D 分布上均匀覆盖。**

当前实现方法是基于 sample 的 2D 中心做贪心 farthest-point sampling：

1. 仍然先计算每个 sample 的 `sample_center`。
2. 先选一个离全局中心最近的 sample 作为起点。
3. 之后每一步都选一个“离当前已选集合最远”的 sample。
4. 重复这个过程，直到选够目标数量。

这个策略的效果是：

- 选出来的点会尽量铺开
- 不会过度集中在某个 cluster
- 适合作为“分布覆盖”对照组

## 6. 图中实际展示方式

三联图中的三个子图分别是：

- 图 1：`Core Hidden`
- 图 2：`Outlier Hidden`
- 图 3：`Uniform Hidden`

每个子图里都会：

1. 先把所有点作为浅色背景画出来。
2. 再把当前选中的那一批 sample 对应的 `visual/text` 点高亮出来。

因此你在每个 panel 中看到的“被强调的点”，就是该 panel 选中的 sample 集合。

## 7. 三个 panel 的样本数量关系

在当前默认参数下：

- `core_ratio = 0.20`
- `outlier_ratio = 0.20`

因此：

- `core` 和 `outlier` 的 sample 数量是一样的
- `uniform` 的 sample 数量被显式设成和 `core` 一样

所以在当前这张图里：

- 图 1、图 2、图 3 选取的 sample 数量是完全一样多的

注意：

如果以后手动把参数改成：

- `core_ratio != outlier_ratio`

那么：

- 图 1 和图 2 的 sample 数量就不一定相同
- 图 3 仍然会跟图 1（也就是 `core`）保持相同数量

## 8. 当前版本和旧版本的区别

旧版本里，`core / outlier` 用的是 `KDE density + radius` 的混合打分：

- `core` 更偏向“高密度 + 靠近各自模态中心”
- `outlier` 更偏向“低密度 + 远离中心”

这种定义不一定对应“全局中心区域 vs 全局边缘区域”。

当前版本已经改成纯几何定义：

- `core` = 全局中心附近
- `outlier` = 全局边缘区域

这更符合直观解释，也更符合你现在对图的预期。
