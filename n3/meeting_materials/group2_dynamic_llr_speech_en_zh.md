# English speaker script

Good morning. This slide shows how we choose the dynamic LLR scale.

First, we evaluate the carriers at different frequencies one by one. For carrier k, `H_k` is the carrier response. `e_k` is the mean squared residual error, calculated over the repeated symbols for that carrier. It is an error-power measure, not a separate sample-variance calculation. A smaller `e_k` is better because it means less unexplained error.

Next, we calculate the carrier reliability as the squared magnitude of the carrier response divided by this error. A larger `|H_k|` means that the useful carrier component is stronger after the acoustic channel, so its useful power `|H_k|²` is larger. Therefore, when other conditions are similar, a larger `|H_k|` and a smaller `e_k` are better. The ratio `|H_k|² / e_k` combines both effects. We compare each carrier with the median reliability. This gives a relative weight for each frequency carrier.

We then divide by the median and take the square root: `w_k = √(ρ_k / ρ̄)`. The square root keeps the ordering of the carriers but avoids making the LLR values too extreme.

The final per-carrier scale is `s_k = b · w_k`. The slide focuses on this frequency-dependent weight. We use the resulting scale to make the two LLR values from each QPSK carrier, and the LDPC decoder uses these LLR values and the parity checks.

With this dynamic scale, the BER after LDPC became zero, and we recovered the original JPG file.

# 中文演讲稿

大家好。这一页说明动态 LLR scale 是怎样计算的。

首先，我们逐个评估不同频率的载波。对于第 k 个载波，`H_k` 表示该载波的响应，`e_k` 表示该载波重复符号上的平均平方残差。它表示误差功率，不是另外做了去均值的样本方差计算。`e_k` 越小越好，因为这表示无法解释的误差越少。

然后，我们用“载波响应幅度的平方除以误差”计算载波可靠度。`|H_k|` 越大，表示该频率载波经过声学通道后保留的有效幅度越强，对应的有效功率 `|H_k|²` 也越大。因此，在其他条件相近时，`|H_k|` 越大、`e_k` 越小越好；`|H_k|²/e_k` 这个比值同时考虑了这两个因素。之后再把每个载波和所有载波的可靠度中位数进行比较，得到每个载波的相对权重。

然后，我们用可靠度中位数进行归一化，再开平方：`w_k = √(ρ_k / ρ̄)`。开平方保留了载波之间的大小规律，同时避免 LLR 数值过于极端。

第 k 个载波的最终 scale 是 `s_k = b · w_k`。这一页重点展示的是随频率变化的权重 `w_k`。我们使用最终 scale 生成每个 QPSK 载波的两个 LLR，然后把这些 LLR 和奇偶校验关系一起交给 LDPC 解码器。

使用动态 scale 后，LDPC 后 BER 达到 0%，原始 JPG 文件成功恢复。
