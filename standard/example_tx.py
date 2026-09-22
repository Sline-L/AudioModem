import numpy as np
import sounddevice as sd
import ldpc
import zlib
import random

from scipy.io import wavfile

from pathlib import Path


class Ofdm:
    DEFAULTS = {
            "send_file": "files/cute.jpg",
            "CP": 2048,
            "N": 8192,
            "start_index": 400,
            "chunk_size": 498,
            "preamable_count": 8,
            "received_audio": "saved.wav",
            "ldpc_en": True,
            "start_freq": 100,
            "end_freq": 20000,
            "chirp_duration": 3.0,
            "amp": 0.8,
            "byte_len": 8,
            "sample_rate": 48000,
            "silence_duration": 0.5
            }

    def __init__(self, **kwargs):
        for key, default_value in self.DEFAULTS.items():
            setattr(self, key, kwargs.get(key, default_value))
        assert self.chunk_size % 6 == 0
        self.coder = ldpc.code(standard="802.16", rate="1/2", z=self.chunk_size//3)

    def log_chirp_gen(self):
        t = np.linspace(0, self.chirp_duration,
                        int(self.sample_rate * self.chirp_duration),
                        endpoint=False)
        f_ratio = self.end_freq / self.start_freq
        B = self.end_freq - self.start_freq
        # phi = 2 * np.pi * self.start_freq * self.chirp_duration / np.log(f_ratio) * (f_ratio ** (t / self.chirp_duration) - 1)
        phi = 2 * np.pi * self.start_freq * t + t*t*np.pi*B/self.chirp_duration
        sweep = self.amp * np.cos(phi)
        return sweep

    def save_sound(self, audio_arr, name='Saved_audio.wav'):
        array_int16 = (audio_arr*32767).astype(np.int16)
        wavfile.write(name, self.sample_rate, array_int16)

    def mapp(self, chunk_arr):
        symbol_data_f = []
        chunk_arr = chunk_arr.astype(int)
        for data_chunk in chunk_arr:
            arr = np.zeros(self.N, dtype=complex)
            for i in range(4*self.chunk_size):
                d_i = i//4
                b_i = 3 - i % 4
                arr[self.start_index+i] = -2*((data_chunk[d_i] >> (b_i*2)) & 1) + 1 + \
                         -2*((data_chunk[d_i] >> (b_i*2+1)) & 1) * 1j + 1j
                arr[self.N-self.start_index-i] = np.conj(arr[self.start_index+i])
            symbol_data_f.append(arr)
        return np.array(symbol_data_f)

    def sequence(self, length, seed=0x5A4D):
        state = int(seed)
        result = np.empty(length, int)
        for i in range(length):
            result[i] = (state >> 14) & 1
            feedback = ((state >> 14) ^ (state >> 13)) & 1
            state = ((state << 1) & 0x7FFF) | feedback
        return result

    def train_symbol_gen(self,
                         start_index,
                         end_index,
                         symbol_count = 16,
                         N = 8192,
                         rand_seed = 80):
        '''
            start_index: 起始频率对应的列表索引
            end_index: 结束频率对应的列表索引，（不包含本索引）
            symbol_count: 生成的symbol数量
            N: symbol长度
            rand_seed: 随机数种子

            返回值：
                arr:     二维numpy列表，symbol_count行，N列，频域信号，幅值为根号2
                time_arr:二维numpy列表，symbol_count行，N列，时域信号，未归一化
            '''
        random.seed(rand_seed)
        arr = np.zeros((symbol_count, N), dtype=complex)
        for i in range(symbol_count):
            for j in range(start_index, end_index):
                rand_num = random.randint(0, 3)
                arr[i, j] = -2*(rand_num & 1) + 1 - 2*((rand_num >> 1) & 1)*1j + 1j
                arr[i, N - j] = np.conj(arr[i][j])
        time_arr = np.fft.ifft(arr, axis=1).real
        last_cp = time_arr[:, -self.CP:]
        arr_with_cp = np.hstack((last_cp, time_arr))

        return arr, arr_with_cp



class Ofdm_tx(Ofdm):
    def __init__(self, tx_mode="save", **kwargs):
        super().__init__(**kwargs)
        self.tx_mode = tx_mode

    def header_gen(self, file_name, file_size: int, symbol_count: int, availible_bytes):
        '''
        整体结构
        PH(magic number)    0x00(version)    文件本体大小(以字节计)(占4bytes)    payloadsymbol数量(占4bytes)  filename  0x00(文件名结束标识)  CRC32值(4bytes)  补0

        函数输入:
            file_name:文件名字符串
            file_size:文件有多少个bytes
            symbol_count:共计发送了多少个symbols
            availible_bytes:最终会补0到多少bytes

        返回值：
            ba:bytearray对象，长度为availible_bytes
        '''
        ba = bytearray(availible_bytes)
        ba[0:2] = b'PH'
        ba[2] = 0
        ba[3:7] = file_size.to_bytes(4)
        ba[7:11] = symbol_count.to_bytes(4)
        name_string = file_name.encode('utf-8')
        ba[11:11+len(name_string)] = name_string
        crc_result = zlib.crc32(ba[:11+len(name_string)])
        ba[12+len(name_string):16+len(name_string)] = crc_result.to_bytes(4)
        return ba

    def file_process(self):
        f = open(self.send_file, "rb")
        file_byte = f.read()
        size = len(file_byte)
        f.close()
        header_bytes = self.header_gen(Path(self.send_file).name, size, int(np.ceil(2*size/self.chunk_size)), self.chunk_size//2)
        byte = b''.join([bytes(header_bytes),
                         file_byte])
        return byte

    def ldpc_encoding(self, arr_uncoded):
        if self.ldpc_en:
            total_len = ((len(arr_uncoded) + self.chunk_size - 1) // self.chunk_size) * self.chunk_size
            padded_data = arr_uncoded.ljust(total_len, b'\x00')
            n_chunks = 2*len(padded_data)//self.chunk_size
            n_bytes = self.chunk_size//2
            chunk_lis = []
            for i in range(n_chunks):
                chunk = padded_data[i*n_bytes:(i+1)*n_bytes]
                data_lis = []
                for j in range(n_bytes):
                    byte = np.array([(chunk[j]&128) // 128,
                            (chunk[j]&64) // 64,
                            (chunk[j]&32) // 32,
                            (chunk[j]&16) // 16,
                            (chunk[j]&8) // 8,
                            (chunk[j]&4) // 4,
                            (chunk[j]&2) // 2,
                            chunk[j]&1], dtype=int)
                    data_lis.append(byte)
                raw_bin = np.concatenate(data_lis)
                encoded_bin = self.coder.encode(raw_bin)
                scambler = self.sequence(self.chunk_size*8)
                scambled_bin = encoded_bin^scambler
                assert len(encoded_bin)//8 == self.chunk_size
                encoded = np.zeros(self.chunk_size, dtype=np.uint8)
                for j in range(self.chunk_size):
                    encoded[j] = scambled_bin[8*j]*128 + \
                                 scambled_bin[8*j+1]*64 + \
                                 scambled_bin[8*j+2]*32 + \
                                 scambled_bin[8*j+3]*16 + \
                                 scambled_bin[8*j+4]*8 + \
                                 scambled_bin[8*j+5]*4 + \
                                 scambled_bin[8*j+6]*2 + \
                                 scambled_bin[8*j+7]
                chunk_lis.append(encoded)
            return np.array(chunk_lis)

        else:
            total_len = ((len(arr_uncoded) + self.chunk_size - 1) // self.chunk_size) * self.chunk_size
            padded_data = arr_uncoded.ljust(total_len, b'\x00')

            return np.frombuffer(padded_data, dtype=np.uint8).reshape(-1, self.chunk_size)

    def play_sound(self, audio_arr):
        sd.play(audio_arr, samplerate=self.sample_rate)
        sd.wait()

    def time_domain_convert(self, freq_data):
        time_arr = np.fft.ifft(freq_data, axis=1).real
        last_cp = time_arr[:, -self.CP:]
        arr_with_cp = np.hstack((last_cp, time_arr))
        return np.concatenate(arr_with_cp)

    def tx(self, saved_name="send_audio.wav"):
        file_bytes = self.file_process()
        silence = np.zeros(int(self.sample_rate*self.silence_duration))
        coded_bytes = self.ldpc_encoding(file_bytes)
        f = open("send.tif", "wb")
        f.write(coded_bytes)
        f.close()
        mapped = self.mapp(coded_bytes)
        ofdm_symbols = self.time_domain_convert(mapped)
        print(ofdm_symbols.shape)
        _, training_symbols = self.train_symbol_gen(self.start_index, self.start_index+4*self.chunk_size, N=self.N, symbol_count=2*self.preamable_count)
        start_symbols = np.concatenate(training_symbols[:self.preamable_count])
        end_symbols = np.concatenate(training_symbols[self.preamable_count:])
        all_symbols = np.concatenate((start_symbols, ofdm_symbols, end_symbols))
        all_symbols = 0.8*all_symbols/np.max(all_symbols)
        chirp = self.log_chirp_gen()
        final_arr = np.concatenate((chirp, silence, all_symbols, silence, chirp))
        if self.tx_mode == "save":
            self.save_sound(final_arr, name=saved_name)
        elif self.tx_mode == "play":
            self.play_sound(final_arr)
        else:
            raise NameError("tx_mode must either be save or play")


if __name__ == "__main__":
    t = Ofdm_tx()
    t.tx()
