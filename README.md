# pairwise-lockon-target-softlock

## 领域

游戏、图形与多媒体

## 主题

目标锁定与镜头软锁

## 说明

逐帧视线、句柄代际与查询序号。

初始状态下 `gameplay/lockon_target_softlock.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
