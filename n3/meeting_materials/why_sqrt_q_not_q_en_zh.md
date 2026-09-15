# Why use √q instead of q directly?

## English answer

For carrier `k`, we first define its relative reliability:

`q_k = ρ_k / ρ̄`

where `ρ̄` is the median reliability of all carriers. We then use:

`w_k = √q_k`

instead of using `q_k` directly.

The reason is that `q_k` can have a large range. If we use `q_k` directly, a very good carrier can receive an excessively large weight, while a weak carrier can receive an excessively small weight. The LDPC decoder may then become overconfident in a few carriers.

The square root compresses this range but keeps the correct order:

- `q_k = 1` gives `w_k = 1`, so a median carrier is unchanged.
- `q_k = 4` gives `w_k = 2`, so a good carrier is still strengthened, but not by four times.
- `q_k = 0.25` gives `w_k = 0.5`, so a weak carrier is reduced, but not pushed too close to zero.

Thus, `√q_k` keeps the reliability difference while avoiding extreme LLR values. It is a practical balance between ignoring carrier quality and trusting it too much.

## 中文说明

对于第 `k` 个载波，我们先定义相对可靠度：

`q_k = ρ_k / ρ̄`

其中 `ρ̄` 是所有载波可靠度的中位数。然后使用：

`w_k = √q_k`

而不是直接使用 `q_k`。

原因是 `q_k` 的范围可能很大。如果直接使用 `q_k`，特别好的载波会得到过大的权重，较差的载波会得到过小的权重，LDPC 解码器可能会过度相信少数载波。

开平方可以压缩这个范围，同时保留正确的大小规律：

- `q_k = 1` 时，`w_k = 1`，中等载波不变。
- `q_k = 4` 时，`w_k = 2`，好的载波仍然被增强，但不是增强 4 倍。
- `q_k = 0.25` 时，`w_k = 0.5`，较差的载波被减弱，但不会被压得接近 0。

所以，`√q_k` 在“完全忽略载波质量”和“过度相信载波质量”之间取得了平衡，可以避免 LLR 数值过于极端。
