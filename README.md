

# 线性回归示例

使用纯 Python 实现一元线性回归（最小二乘法），不需要安装第三方依赖。

运行示例：

```bash
python linear_regression.py
```

`fit_linear_regression(x_values, y_values)` 会返回拟合模型，模型提供 `slope`、`intercept` 和 `predict(x_values)`。输入需要是长度相同、至少包含两个数值且自变量不全相同的序列。
