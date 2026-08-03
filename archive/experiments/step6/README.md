# Step6 Profiles / Step6 Profile 归档

Step6 compared the Step4 frequency selections under N=512/CP=256 and
N=1024/CP=256. All three sender/receiver pairs and their local modem modules are
kept here unchanged.

Step6 比较了 Step4 频点在 N=512/CP=256 与 N=1024/CP=256 下的表现。三组发送、接收和
modem 模块均按原样保存。

Example / 示例：

```bash
python archive/experiments/step6/tx_step6_step4.py
python archive/experiments/step6/rx_step6_step4.py \
  data/step6_newexp/file_combo_step4trimmed_n512_cp256.wav \
  --out runs/legacy_step6_offline
```
