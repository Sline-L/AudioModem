# What is LLR scale?

## English answer

LLR means **log-likelihood ratio**. It is a number that tells the LDPC decoder two things:

- The sign gives the bit decision. A positive value favors one bit value, and a negative value favors the other.
- The absolute value gives the confidence. A large absolute value means more confidence; a value close to zero means less confidence.

The LLR scale controls the size of this confidence. For carrier `k`, we use:

`LLR_scaled,k = s_k × LLR_raw,k`

Here, `s_k` is the scale for that carrier. It does not change the hard bit decision when it is positive. It changes how strongly the LDPC decoder trusts that decision.

If the scale is too large, the decoder may trust a wrong bit too much. If it is too small, the decoder may not use a reliable bit strongly enough. We therefore use a different scale for carriers at different frequencies: a weak carrier gets a smaller scale, and a reliable carrier gets a larger scale.

## 中文说明

LLR 是 **对数似然比**。它告诉 LDPC 解码器两件事：

- 符号表示比特判决方向。正值倾向于一个比特值，负值倾向于另一个比特值。
- 绝对值表示置信度。绝对值越大，表示越有把握；接近 0，表示不确定。

LLR scale 控制这个置信度的大小。对于第 `k` 个载波：

`LLR_scaled,k = s_k × LLR_raw,k`

这里的 `s_k` 是该载波的 scale。只要 scale 为正，它不会改变硬判决的比特方向，但会改变 LDPC 解码器对这个判决的信任程度。

scale 太大时，解码器可能会过度相信错误比特；scale 太小时，又可能没有充分利用可靠比特。因此我们针对不同频率的载波使用不同 scale：较差的载波使用较小 scale，可靠的载波使用较大 scale。
