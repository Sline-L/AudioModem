# N3.2 独立标准声学调制解调器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** 在 \`n3_2/\` 中从零实现一个不依赖 N3/N3.1、发射线格式严格复现标准示例的 OFDM 文件调制解调器。

**Architecture:** \`modem.py\` 包含固定物理层、帧构造和无状态协议原语；\`ldpc_codec.py\` 将仓库 LDPC 库封装为可移植的软判决编解码器；\`sync.py\` 专职声道选择、chirp/training 定位与 SFO 估计。 \`tx.py\` 是唯一标准发射 CLI，\`rx.py\` 用同步结果驱动解调、Header 校验、文件恢复和诊断落盘。

**Tech Stack:** Python 3、NumPy、SciPy、标准库 \`wave\`/\`json\`/\`zlib\`、仓库已有 \`lib/ldpc\` 的 C DLL。

**Spec:** \`docs/superpowers/specs/2026-09-15-n3_2-standalone-design.md\`

## Global Constraints

- 仅在 \`n3_2/\`、N3.2 测试和文档内新增或修改；不得删除或修改 N1、N2、N3、N3.1 和归档数据。
- N3.2 源码不得导入、调用或运行时依赖 \`n3\`、\`n3_1\`；只允许内部相对导入、NumPy、SciPy 与 \`lib/ldpc\`。
- 发射协议以 \`C:\Users\Administrator\Desktop\Standardization\example_tx.py\` 为准：48 kHz、FFT 8192、CP 2048、bins 400..2391、未归一化 QPSK、一个 Header 信息块、seed=80 的 16 个 training 和线性 3 秒 chirp。
- 每个码字使用 IEEE 802.16、rate \`1/2\`、Z=166；每个码字均以 \`0x5A4D\` 重置扰码器。
- 接收端仅支持 16-bit、48 kHz PCM WAV；支持 mono/stereo，并在 stereo 情况选择评分最高的一路。
- 新生成 WAV 及其 sidecar 仅写入 \`data/n3_2/\`；接收输出写入用户指定目录。
- 每项先写失败测试，再写最小实现；参考文件不得作为 N3.2 的运行时依赖。

---

### Task 1: 创建独立包和协议黄金原语

**Files:**
- Create: \`n3_2/__init__.py\`
- Create: \`n3_2/modem.py\`
- Create: \`tests/test_n3_2_protocol.py\`

**Interfaces:**
- Consumes: 无。
- Produces: 常量 \`FS=48000, N=8192, CP=2048, L=10240, K0=400, K1=2391, M=1992\`，以及 \`header_bytes(name, size, payload_symbols) -> bytes\`、\`parse_header(raw) -> dict\`、\`scramble_bits(bits, seed=0x5A4D) -> ndarray\`、\`training_symbols() -> ndarray\`、\`qpsk_map(bits) -> ndarray\`、\`qpsk_llr(symbols, noise_var) -> ndarray\`、\`ofdm_time(carriers) -> ndarray\`、\`linear_chirp() -> ndarray\`、\`read_pcm16_wav(path) -> tuple[int, ndarray]\`、\`write_pcm16_wav(path, samples) -> None\`。

- [ ] **Step 1: 写失败测试**

    def test_reference_golden_vectors():
        raw = header_bytes("x.bin", 5, 1)
        assert hashlib.sha256(raw).hexdigest() == (
            "e158f3dff4d6c8b8494c3e9cc48ebd03cdbcb4350d45e9dbb991cc6ce7793a4e"
        )
        assert raw[:18].hex() == "5048000000000500000001782e62696e00d0"
        assert "".join(map(str, scramble_bits(np.zeros(32, np.uint8)))) == "10110100100110111011101101011001"
        t = training_symbols()
        assert t.shape == (16, 8192)
        assert hashlib.sha256(t.tobytes()).hexdigest() == "1eaf58db08942730d41a049d5f77b1ba9e0e672d225773b69530538887f36404"

    def test_qpsk_cp_and_chirp():
        assert np.array_equal(qpsk_map(np.array([0, 0, 0, 1], np.uint8)), np.array([1+1j, -1+1j]))
        x = ofdm_time(np.ones((1, M), np.complex128))
        assert x.shape == (1, L)
        assert np.allclose(x[0, :CP], x[0, -CP:])
        assert linear_chirp().shape == (3 * FS,)

- [ ] **Step 2: 运行失败测试**

Run: \`python -m pytest tests/test_n3_2_protocol.py -v\`

Expected: FAIL，因为 \`n3_2\` 和接口尚不存在。

- [ ] **Step 3: 写最小独立实现**

    def qpsk_map(bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.uint8).reshape(-1, 2)
        return (1 - 2 * b[:, 1]) + 1j * (1 - 2 * b[:, 0])

    def ofdm_time(carriers: np.ndarray) -> np.ndarray:
        f = np.zeros((len(carriers), N), dtype=np.complex128)
        f[:, K0:K1 + 1] = carriers
        f[:, N - K1:N - K0 + 1] = np.conj(carriers[:, ::-1])
        x = np.fft.ifft(f, axis=1).real
        return np.concatenate((x[:, -CP:], x), axis=1)

按 \`header_gen.py\` 写 Header、CRC 覆盖范围和字段验证；按 \`scrambler.py\` 写 15-bit LFSR；用局部 \`random.Random(80)\` 复现 \`training_symbol_gen.py\`。WAV 读写实现精确 16-bit 48 kHz PCM 验证。

- [ ] **Step 4: 运行通过测试**

Run: \`python -m pytest tests/test_n3_2_protocol.py -v\`

Expected: PASS；Header、扰码、training、QPSK、CP 和 chirp 均吻合黄金向量。

- [ ] **Step 5: 提交**

    git add n3_2/__init__.py n3_2/modem.py tests/test_n3_2_protocol.py
    git commit -m "n3_2 standard: add protocol primitives"

### Task 2: 封装可移植的 IEEE 802.16 LDPC 软判决编解码

**Files:**
- Create: \`n3_2/ldpc_codec.py\`
- Create: \`tests/test_n3_2_ldpc.py\`

**Interfaces:**
- Consumes: \`n3_2.modem.scramble_bits\`、\`lib/ldpc/py/ldpc.py\`。
- Produces: \`INFO_BITS=1992\`、\`CODE_BITS=3984\`、\`StandardLdpc.encode(info_bits) -> ndarray\`、\`StandardLdpc.decode_llr(scrambled_llr) -> tuple[ndarray, bool]\`、\`StandardLdpc.syndrome(bits) -> ndarray\`。

- [ ] **Step 1: 写失败测试**

    def test_noiseless_ldpc_roundtrip_and_syndrome():
        codec = StandardLdpc()
        info = np.random.default_rng(12).integers(0, 2, 1992, dtype=np.uint8)
        coded = codec.encode(info)
        assert coded.shape == (3984,)
        assert not codec.syndrome(coded).any()
        decoded, ok = codec.decode_llr((1 - 2 * coded).astype(float) * 8)
        assert ok and np.array_equal(decoded, info)

    def test_soft_descrambling_precedes_decode():
        codec = StandardLdpc()
        info = np.zeros(1992, np.uint8)
        coded = codec.encode(info)
        decoded, ok = codec.decode_llr((1 - 2 * scramble_bits(coded)).astype(float) * 6)
        assert ok and np.array_equal(decoded, info)

- [ ] **Step 2: 运行失败测试**

Run: \`python -m pytest tests/test_n3_2_ldpc.py -v\`

Expected: FAIL，提示 \`n3_2.ldpc_codec\` 不存在。

- [ ] **Step 3: 写最小实现**

    class PortableCode(ldpc.code):
        def _load_library(self):
            root = Path(ldpc.__file__).resolve().parents[1]
            return ctypes.CDLL(str(root / "bin" / f"c_ldpc{suffix}"))

    class StandardLdpc:
        def __init__(self):
            self.code = PortableCode(standard="802.16", rate="1/2", z=166)

将 1992 info bits 大端 pack 为 249 bytes；编码展平为 3984 bits。解码前生成同一 reset seed 的扰码序列，并将 LLR 乘 \`1 - 2*sequence\`，再调用 soft sum-product。用输出码字的 syndrome 是否全零确定 \`ok\`。不得引入 N2 的 802.11n interleaver。

- [ ] **Step 4: 运行通过测试**

Run: \`python -m pytest tests/test_n3_2_ldpc.py -v\`

Expected: PASS；无噪声编解码正确、syndrome 为零、软域解扰正确。

- [ ] **Step 5: 提交**

    git add n3_2/ldpc_codec.py tests/test_n3_2_ldpc.py
    git commit -m "n3_2 standard: add portable ldpc codec"

### Task 3: 实现严格标准格式的独立发射端

**Files:**
- Create: \`n3_2/tx.py\`
- Create: \`tests/test_n3_2_tx.py\`

**Interfaces:**
- Consumes: Task 1 的 Header/training/QPSK/OFDM/chirp/WAV 和 Task 2 的 \`StandardLdpc.encode\`。
- Produces: \`build_frame(source: bytes, name: str) -> tuple[ndarray, bytes, dict]\`、\`run_tx(source: Path, out: Path | None) -> Path\`、CLI \`python -m n3_2.tx INPUT --out OUTPUT\`。

- [ ] **Step 1: 写失败测试**

    def test_tx_frame_layout_and_sidecars(tmp_path):
        src = tmp_path / "a.bin"
        src.write_bytes(b"abc")
        out = run_tx(src, tmp_path / "a.wav")
        fs, x = read_pcm16_wav(out)
        assert fs == 48000
        assert len(x) == 2 * 3 * FS + FS + (8 + 2 + 8) * L
        assert out.with_suffix(".training.npy").exists()
        assert out.with_suffix(".header.bin").read_bytes()[:2] == b"PH"
        assert json.loads(out.with_suffix(".meta.json").read_text())["payload_symbols"] == 1

- [ ] **Step 2: 运行失败测试**

Run: \`python -m pytest tests/test_n3_2_tx.py -v\`

Expected: FAIL，提示 \`n3_2.tx\` 或 \`run_tx\` 不存在。

- [ ] **Step 3: 写最小发射实现**

    stream = header + source
    stream += bytes((-len(stream)) % 498)
    info_blocks = np.frombuffer(stream, np.uint8).reshape(-1, 249)
    # 每块：LDPC -> reset-seed scramble -> QPSK -> conjugate mirror IFFT/CP
    frame = np.concatenate([chirp, silence, front_training, payload,
                            tail_training, silence, chirp])

每个 249-byte 块必须 LDPC、扰码并映射到 1992 个 carriers；Header 与 payload 合流后先补零至 498-byte 整数倍。安全量化 int16，并创建 WAV、\`.training.npy\`、\`.header.bin\`、\`.meta.json\`。CLI 不提供非标准 modulation、training 或 chirp 参数。

- [ ] **Step 4: 运行通过测试**

Run: \`python -m pytest tests/test_n3_2_tx.py -v\`

Expected: PASS；帧段顺序、样本数、Header 和 sidecar 正确。

- [ ] **Step 5: 提交**

    git add n3_2/tx.py tests/test_n3_2_tx.py
    git commit -m "n3_2 standard: add reference frame transmitter"

### Task 4: 实现同步、立体声择优与 SFO 估计

**Files:**
- Create: \`n3_2/sync.py\`
- Create: \`tests/test_n3_2_sync.py\`

**Interfaces:**
- Consumes: Task 1 的 chirp/training/常量。
- Produces: \`ChannelChoice(index, score, samples)\`、\`SyncResult(start, training_start, payload_start, sfo, score)\`、\`choose_channel(samples) -> ChannelChoice\`、\`synchronize(samples) -> SyncResult\`、\`SyncError(stage, message)\`。

- [ ] **Step 1: 写失败测试**

    def test_stereo_selects_cleaner_channel_and_finds_training():
        clean = known_standard_frame()
        noisy = np.random.default_rng(4).normal(0, 0.3, len(clean))
        choice = choose_channel(np.column_stack([noisy, clean]))
        result = synchronize(choice.samples)
        assert choice.index == 1
        assert result.training_start == 3 * FS + FS // 2
        assert result.score > 0.9

    def test_sync_reports_specific_chirp_failure():
        with pytest.raises(SyncError, match="chirp"):
            synchronize(np.zeros(10 * FS))

- [ ] **Step 2: 运行失败测试**

Run: \`python -m pytest tests/test_n3_2_sync.py -v\`

Expected: FAIL，提示 \`n3_2.sync\` 不存在。

- [ ] **Step 3: 写最小同步实现**

    def choose_channel(samples):
        channels = [samples] if samples.ndim == 1 else [samples[:, i] for i in range(samples.shape[1])]
        scores = [chirp_pair_score(x) for x in channels]
        i = int(np.argmax(scores))
        return ChannelChoice(i, float(scores[i]), np.asarray(channels[i]))

每一声道以归一化匹配滤波寻找合法前后 chirp pair，并检查间隔是否与帧结构相容；不把 stereo 平均为 mono。在预期位置附近用 training 频域相关定位 symbol 边界，随后先 bounded coarse SFO search，再以 training phase-vs-bin 线性拟合微调。失败必须用 \`channel\`、\`chirp\`、\`training\` 或 \`sfo\` 标识阶段。

- [ ] **Step 4: 运行通过测试**

Run: \`python -m pytest tests/test_n3_2_sync.py -v\`

Expected: PASS；右声道获选、training 定位正确、无 chirp 明确报 chirp 失败。

- [ ] **Step 5: 提交**

    git add n3_2/sync.py tests/test_n3_2_sync.py
    git commit -m "n3_2 standard: add channel-aware synchronization"

### Task 5: 实现 Header 接收、明确诊断与安全失败

**Files:**
- Create: \`n3_2/rx.py\`
- Create: \`tests/test_n3_2_rx.py\`

**Interfaces:**
- Consumes: Task 1 的 WAV/LLR/Header，Task 2 的 LDPC，Task 4 的 channel/sync。
- Produces: \`decode_header(path, source=None) -> tuple[dict, dict]\`、\`run_rx(path, out, source=None) -> Path\`、CLI \`python -m n3_2.rx INPUT --out DIR [--source FILE]\`、\`DecodeError(stage, message, metrics)\`。

- [ ] **Step 1: 写失败测试**

    def test_rx_decodes_header_and_reports_stage(tmp_path):
        wav = make_standard_wav(tmp_path, b"hello", "hello.bin")
        header, diag = decode_header(wav)
        assert header["filename"] == "hello.bin"
        assert header["file_size"] == 5
        assert diag["stage"] == "header_ok"

    def test_pcm_and_header_failures_are_distinct(tmp_path):
        with pytest.raises(DecodeError, match="wav_format"):
            decode_header(write_non_48k_wav(tmp_path / "bad.wav"))
        with pytest.raises(DecodeError, match="header_crc|header_fields|ldpc"):
            decode_header(make_corrupt_header_wav(tmp_path))

- [ ] **Step 2: 运行失败测试**

Run: \`python -m pytest tests/test_n3_2_rx.py -v\`

Expected: FAIL，提示 \`n3_2.rx\` 不存在。

- [ ] **Step 3: 写最小 Header 解调实现**

    def decode_codeword(symbol, h):
        y = np.fft.fft(symbol[CP:CP + N])[K0:K1 + 1]
        return qpsk_llr(y / h, noise_var)

以前置 8 个 training 的复数均值估计 \`H\`，均衡第一个码字，得到 LLR，软解扰后 LDPC decode。用 \`parse_header\` 校验 magic、version、size、payload_symbols、UTF-8 filename 和 CRC。无论成败都写 \`metrics.json\`、\`header_debug.json\`、\`decoded_header.bin\`、\`H.npy\`、\`training_phase_fit.npy\`、\`summary.csv\`；失败绝不写恢复文件。传 \`--source\` 时，只在 debug 中增加 Header BER，不改变判定。

- [ ] **Step 4: 运行通过测试**

Run: \`python -m pytest tests/test_n3_2_rx.py -v\`

Expected: PASS；Header 恢复正确，WAV 格式失败和 Header 失败可区分。

- [ ] **Step 5: 提交**

    git add n3_2/rx.py tests/test_n3_2_rx.py
    git commit -m "n3_2 standard: decode header with diagnostics"

### Task 6: 完成 payload 恢复和端到端鲁棒性

**Files:**
- Modify: \`n3_2/rx.py\`
- Modify: \`tests/test_n3_2_rx.py\`

**Interfaces:**
- Consumes: Task 5 的 Header、同步、均衡和 block decode。
- Produces: 恢复文件、\`decoded_payload.bin\`、\`payload_symbols.npy\`；metrics 含 \`channel_index\`、\`chirp_score\`、\`training_score\`、\`sfo\` 和 \`header_ok\`。

- [ ] **Step 1: 写失败测试**

    @pytest.mark.parametrize("transform", [
        lambda x: x,
        lambda x: np.pad(0.65 * x, (317, 0)),
        lambda x: x + np.random.default_rng(9).normal(0, 0.002, len(x)),
    ])
    def test_rx_recovers_benign_channels(tmp_path, transform):
        source = tmp_path / "payload.bin"
        source.write_bytes(bytes(range(256)) * 3)
        wav = make_standard_wav(tmp_path, source.read_bytes(), source.name)
        recovered = run_rx(rewrite_wav(tmp_path / "channel.wav", transform(read_pcm16_wav(wav)[1])),
                           tmp_path / "out", source=source)
        assert recovered.read_bytes() == source.read_bytes()

    def test_rx_recovers_dual_mono_stereo(tmp_path):
        wav, source = make_stereo_duplicate_fixture(tmp_path)
        assert run_rx(wav, tmp_path / "out").read_bytes() == source.read_bytes()

- [ ] **Step 2: 运行失败测试**

Run: \`python -m pytest tests/test_n3_2_rx.py -v\`

Expected: FAIL，因为 payload blocks、tail-training 验证和恢复文件尚未实现。

- [ ] **Step 3: 写最小 payload 恢复实现**

    blocks = []
    for i in range(header["payload_symbols"]):
        bits, ok = codec.decode_llr(decode_codeword(payload_symbol(i), h))
        if not ok:
            raise DecodeError("ldpc_payload", f"payload block {i} syndrome failed")
        blocks.append(np.packbits(bits, bitorder="big").tobytes())
    payload = b"".join(blocks)[:header["file_size"]]

记录尾 training 一致性分数，需要时只在限定范围修正 SFO/相位。只有全部 Header 指定 payload block 通过且截断后长度符合 Header 才写 \`<out>/<Path(filename).name>\`，防止路径穿越；同时写 payload symbols 和 decoded bytes。

- [ ] **Step 4: 运行通过测试**

Run: \`python -m pytest tests/test_n3_2_rx.py -v\`

Expected: PASS；无损、延迟/增益、轻噪声和 dual-mono stereo 都逐字节恢复。

- [ ] **Step 5: 提交**

    git add n3_2/rx.py tests/test_n3_2_rx.py
    git commit -m "n3_2 standard: recover payload blocks"

### Task 7: 增加 BER 工具、独立性防护与中文文档

**Files:**
- Create: \`n3_2/ber_test.py\`
- Create: \`n3_2/README.md\`
- Create: \`tests/test_n3_2_independence.py\`
- Modify: \`docs/code_guide.md\`

**Interfaces:**
- Consumes: \`run_tx\`、\`run_rx\`、\`decode_header\`。
- Produces: CLI \`python -m n3_2.ber_test INPUT --out DIR\`，中文 CLI/产物/诊断文档。

- [ ] **Step 1: 写失败测试**

    def test_n3_2_has_no_old_runtime_dependency():
        banned = re.compile(r"^\s*(?:from|import)\s+(?:n3(?:_1)?)(?:\s|\.|$)", re.M)
        for source in Path("n3_2").glob("*.py"):
            assert not banned.search(source.read_text(encoding="utf-8")), source

    def test_ber_cli_reports_clean_loopback(tmp_path):
        report = run_ber_test(make_source(tmp_path), tmp_path / "result")
        assert report["header_bit_errors"] == 0
        assert report["payload_byte_errors"] == 0

- [ ] **Step 2: 运行失败测试**

Run: \`python -m pytest tests/test_n3_2_independence.py -v\`

Expected: FAIL，提示 \`ber_test\` 或其接口尚不存在。

- [ ] **Step 3: 写最小 BER 工具与中文文档**

    def run_ber_test(source, out):
        wav = out / "tx.wav"
        run_tx(source, wav)
        recovered = run_rx(wav, out / "rx", source=source)
        original, got = source.read_bytes(), recovered.read_bytes()
        return {"payload_byte_errors": sum(a != b for a, b in zip(original, got)),
                "payload_bytes": len(original)}

README 和 \`docs/code_guide.md\` 用中文解释唯一 TX/RX 命令、标准范围、产物路径、stereo 择优，以及如何读取 \`metrics.json\` 的 \`wav_format\`、\`channel\`、\`chirp\`、\`training\`、\`sfo\`、\`ldpc\`、\`header_crc\`、\`header_fields\` 阶段；不得改写既有版本说明。

- [ ] **Step 4: 运行通过测试**

Run: \`python -m pytest tests/test_n3_2_independence.py -v\`

Expected: PASS；源码不导入旧模块，干净回环 BER 为零。

- [ ] **Step 5: 提交**

    git add n3_2/ber_test.py n3_2/README.md tests/test_n3_2_independence.py docs/code_guide.md
    git commit -m "n3_2 standard: document standalone workflow"

### Task 8: 执行全量验收并记录实测结果

**Files:**
- Modify: \`n3_2/README.md\`

**Interfaces:**
- Consumes: Tasks 1–7 的模块和 CLI。
- Produces: README 中可复现命令和实际离线验收结论。

- [ ] **Step 1: 写编译检查测试**

    def test_all_n3_2_modules_compile():
        paths = [str(p) for p in Path("n3_2").glob("*.py")]
        subprocess.run([sys.executable, "-m", "py_compile", *paths], check=True)

- [ ] **Step 2: 运行测试，确认实施期间的失败会被暴露**

Run: \`python -m pytest tests/test_n3_2_protocol.py tests/test_n3_2_ldpc.py tests/test_n3_2_tx.py tests/test_n3_2_sync.py tests/test_n3_2_rx.py tests/test_n3_2_independence.py -v\`

Expected: 前序任务未完成时 FAIL；此时如有失败，回到对应最小实现修复，不能跳过。

- [ ] **Step 3: 写入并运行验收命令**

    python -m py_compile n3_2/__init__.py n3_2/modem.py n3_2/ldpc_codec.py n3_2/sync.py n3_2/tx.py n3_2/rx.py n3_2/ber_test.py
    python -m pytest tests/test_n3_2_protocol.py tests/test_n3_2_ldpc.py tests/test_n3_2_tx.py tests/test_n3_2_sync.py tests/test_n3_2_rx.py tests/test_n3_2_independence.py -v
    python -m n3_2.tx data/source/file16_test.txt --out data/n3_2/file16_test.wav
    python -m n3_2.rx data/n3_2/file16_test.wav --out runs/n3_2_file16 --source data/source/file16_test.txt

在 README 记录实际的测试命令及 SHA-256 或逐字节比较结论。不得把失败写作通过。

- [ ] **Step 4: 再次运行验收测试**

Run: \`python -m pytest tests/test_n3_2_protocol.py tests/test_n3_2_ldpc.py tests/test_n3_2_tx.py tests/test_n3_2_sync.py tests/test_n3_2_rx.py tests/test_n3_2_independence.py -v\`

Expected: PASS；协议黄金向量、LDPC、帧结构、同步、接收和独立性测试全部通过。

- [ ] **Step 5: 提交**

    git add n3_2/README.md
    git commit -m "n3_2 standard: verify standalone modem"

