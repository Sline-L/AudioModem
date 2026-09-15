# English speaker script

Good morning. We aligned our audio modem receiver with the Standardization reference and then tested it on the WAV file uploaded by Group 2 to Google Drive. We played the WAV file from a phone and recorded the acoustic signal with a computer microphone. The receiver successfully recovered the original JPG file.

The main problem appeared in the LDPC soft-decision stage. Our first implementation used a fixed LLR scale, so it treated every frequency carrier as equally reliable. The raw coded BER was about six to seven percent. After fixed-scale LDPC decoding, the BER only fell to about four percent, and the recovered JPG still could not be opened.

We changed the receiver to use a dynamic, per-carrier LLR scale. We evaluated the quality of carriers at different frequencies from their residual error and the squared magnitude of the carrier response. Weak carriers received smaller LLR magnitudes, while reliable carriers received larger magnitudes. This gave the LDPC decoder better reliability information without changing the standard wire format or the transmitter. With this change, the post-LDPC BER reached zero, and the original JPG opened successfully.

# 中文演讲稿

大家好。我们首先根据 Standardization 文件夹中的标准参考实现调整了接收端代码，然后测试了 Group 2 上传到 Google Drive 的 WAV 文件。我们使用手机播放 WAV 文件，再用电脑麦克风录音，最后成功恢复出了原始 JPG 文件。

最初的问题出现在 LDPC 软判决阶段。第一版使用固定的 LLR scale，因此默认所有频段的载波都同样可靠。LDPC 之前的编码比特 BER 大约为 6% 到 7%，使用固定 scale 的 LDPC 后仍然约为 4%，恢复出的 payload 仍有错误，所以 JPG 无法打开。

随后我们把接收端改成动态的逐载波 LLR scale。我们根据不同频率载波的残差误差和载波响应幅度的平方，分别评估每个载波的质量。对于较差的载波，我们降低 LLR 的绝对值；对于可靠的载波，我们提高 LLR 的绝对值。这样 LDPC 解码器可以获得更准确的可靠度信息，同时没有改变标准帧格式和发射端。改进后，LDPC 后 BER 达到 0%，原始 JPG 成功打开。

## Presentation note

The slide reports approximate values from the receiver runs: raw coded BER about 6.7%, fixed-scale LDPC about 4.1%, and adaptive LLR plus LDPC at 0%. The adaptive result recovered the JPG with `file_match=True`.
