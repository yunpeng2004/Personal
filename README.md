# WKSPB-v2 (Region-KAN only)

这版是按“只把改进型 KAN 精确放到 Region 生长上”的思路整理的完整代码。

## 你需要的目录
把数据按下面放：

```text
wkspb_v2/
├── images/
├── masks/
├── main.py
└── ...
```

## 安装
```bash
pip install -r requirements.txt
```

## 运行
```bash
python main.py
```

## 这版的核心变化
1. Encoder 改成 ResNet34
2. SPB 保留 Wavelet + Region-KAN
3. GDU 改成 Lite 版：只有 Region 分支用 KAN
4. Boundary / Confidence 改回轻量卷积
5. 训练仍然是 early stopping
6. 训练结束后自动在 val 上搜索最佳 threshold，再拿这个 threshold 去测 test

## 为什么这么改
不是删掉 KAN 创新，而是把 KAN 集中放在“最需要高阶状态映射”的地方：
- SPB 的高置信核心初始化
- GDU 的 Region 生长更新

这样比“所有分支都重 KAN”更稳，也更容易把 Dice 提上去。
