from pydoc import text
import os, sys

# ================= 1. 绝对优先：环境锁定 =================
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = "1"
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['no_proxy'] = '*'
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
os.environ["PYTHONUTF8"] = "1"

import torch
print("\n[SYSTEM] 正在加载 PyTorch 张量引擎与 Silero VAD 神经网络...")
try:
    torch.set_num_threads(1) # 限制 CPU 线程，防止游戏卡顿
    # 💡 物理真·离线加载：使用 torch.hub 官方 local 模式，彻底断绝 WinError 10060 和 import 报错！
    vad_dir = os.path.expanduser(r'~/.cache/torch/hub/snakers4_silero-vad_master')
    if os.path.exists(vad_dir):
        vad_model, utils = torch.hub.load(repo_or_dir=vad_dir, source='local', model='silero_vad', force_reload=False)
    else:
        # 兜底方案：如果本地没有，作为兜底允许连网
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        
    vad_model.eval() # 设置为推理模式
    print("✅ [SYSTEM] 神经学耳朵 (VAD) 物理本地挂载完毕。")
except Exception as e:
    print(f"🚨 [FATAL] VAD 模型加载失败，请检查 Torch 安装: {e}")
    vad_model = None

# 💡 架构扩容：预热物理降噪引擎
try:
    import torchaudio
    import sys
    import types
    
    # 💊 物理级猴子补丁 V3 (Absolute Null Patch)：无视底层存在与否，强行注入空壳类
    if 'torchaudio.backend' not in sys.modules:
        fake_backend = types.ModuleType('torchaudio.backend')
        fake_common = types.ModuleType('torchaudio.backend.common')
        
        # 强行捏造空壳类，物理塞满 DFN 所有的 import 幻想
        class DummyAudioMetaData: pass
        fake_common.AudioMetaData = DummyAudioMetaData
        fake_common.AudioInfo = DummyAudioMetaData
            
        fake_backend.common = fake_common
        sys.modules['torchaudio.backend'] = fake_backend
        sys.modules['torchaudio.backend.common'] = fake_common
        torchaudio.backend = fake_backend

    from df.enhance import init_df
    import os
    
    # 🛡️ 物理隔离级加载：直接指向本地权重文件夹，彻底斩断 GitHub 联网请求
    # 💡 作用域修复：在顶部独立解析物理绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 💡 物理层级对齐：直接将准星瞄准包含 config.ini 的绝对底层目录
    dfn_model_dir = os.path.join(current_dir, "dfn_model", "DeepFilterNet3")
    if os.path.exists(os.path.join(dfn_model_dir, "config.ini")):
        df_model, df_state, _ = init_df(model_base_dir=dfn_model_dir)
        print("✅ [SYSTEM] 深度滤波引擎 (DeepFilterNet) 物理本地挂载完毕。")
    else:
        print(f"⚠️ [SYSTEM] 未找到本地 DFN 权重配置 ({dfn_model_dir})。请确保文件解压正确。")
        df_model = None
        df_state = None
except Exception as e:
    # 💡 致命诊断暴露：严禁静默崩溃，必须把底层的物理报错吐到终端
    print(f"⚠️[SYSTEM] DFN 引擎初始化失败，已旁路该节点。底层报错: {e}")
    df_model = None
    df_state = None
finally:
    # 🛡️ 绝对升盾：无论 DFN 是否成功，必须在 finally 块中强行恢复 HuggingFace 离线锁
    # 彻底封死网络侧漏，保护后续 RAG 模型的本地运行，消灭 WinError 10060！
    os.environ['HF_HUB_OFFLINE'] = '1' 

if hasattr(sys.stdout, 'reconfigure'): 
    sys.stdout.reconfigure(encoding='utf-8')

# ================= 2. 核心库导入 =================
import json, time, threading, pyaudio, re, queue # 💡 核心新增：引入线程安全的队列
import numpy as np
import parselmouth # 💡 物理挂载：Praat C/C++ 声学核心底层绑定
import keyboard

def extract_acoustic_tensor(audio_bytes):
    """🎙️ V12.10 零样本声学探针: 提取干声的 F0 (音高) 与 dB (能量) 物理张量"""
    if not audio_bytes or len(audio_bytes) < 3200: # 抛弃小于200ms的残包，防止计算除零错误
        return None
    try:
        # 将 16bit PCM 字节物理还原为归一化的[-1.0, 1.0] float32 浮点波形
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        snd = parselmouth.Sound(audio_array, sampling_frequency=16000)
        
        # 💡 提取基频张量 (F0 Pitch)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        valid_pitch = pitch_values[pitch_values > 0]
        mean_pitch = valid_pitch.mean() if len(valid_pitch) > 0 else 0.0
        
        # 💡 提取声强张量 (Intensity / dB)
        intensity = snd.to_intensity()
        mean_db = intensity.values.mean() if len(intensity.values) > 0 else 0.0
        
        return (mean_pitch, mean_db)
    except Exception:
        return None
import winsound
from datetime import datetime
from pythonosc import udp_client
from openai import OpenAI
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult, VocabularyService
# 💡 核心修复：显式导入 AudioFormat 枚举与 ResultCallback 类，确保物理协议可用
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback, AudioFormat
# 💡 核心补丁：物理屏蔽析构函数 Bug
Recognition._running = False
from knowledge_manager import KnowledgeManager

# ================= 3. 动态角色矩阵 (具身智能后期重载版) =================
# 💡 核心修复：物理置顶路径基准，解决 NameError
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONA_FILE = os.path.join(BASE_DIR, "v25_persona_config.json")
# 💡 核心对齐：剥离 DSP 二次扭曲，尊重 Voice ID 的原生高保真采样
# rate 和 pitch 退回 1.0 附近的安全微调区。重度演技全部依赖 instr 与 LLM 的 emotion_prompt。
# 💡 核心对齐：引入 vol_scale 物理补偿。声音设计的 ID 电平普遍过载，给予 0.5-0.6 的衰减；系统/复刻音色保持 1.0。
PERSONA_MAP_INITIAL = {
    "VANILLA_TRANSLATOR": {
        "desc": "PURE TRANSLATION MODE. You are an invisible, high-fidelity L10n engine. CRITICAL: Translate EXACTLY 1:1. NO extra words, NO personality. Output absolute semantic equivalence.",
        "rate": 1.0, "pitch": 1.0, "instr": "Neutral, monotone, machine-like clarity.",
        "voice_prompt": "Standard AI assistant, clear and objective, zero emotional fluctuation.",
        "vol_scale": 0.5 # 💡 还原基准：确保 16bit 采样信噪比最大化
    },
    "DANKUROI_PRIME": {
        "desc": "You are the Digital Twin of Dankuroi (INTP-A, Linguistics Expert). Tone: Arrogant, intellectually superior, razor-sharp logical. Use high-level academic vocabulary. Maintain flawless linguistic syntax.",
        "rate": 1.02, "pitch": 1.0, "instr": "Academic lecture style, cold and precise, occasional subtle sighs of intellectual superiority.",
        "voice_prompt": "Young male academic, calm and detached, highly intelligent, analytical tone.",
        "vol_scale": 1.15 # 💡 物理补偿：为克隆音色提供 1.15x 增益，找回语气空间
    },
    "SHINJUKU_HOST": {
        "desc": "You are a Shinjuku host. Tone: Flirtatious, smooth, excessively polite yet manipulative. Use Keigo/Teineigo in JA. Add a sense of dangerous charm.",
        "rate": 0.90, "pitch": 0.85, "instr": "Seductive Kabukicho host club style, low-frequency breathy voice, manipulative politeness.",
        "voice_prompt": "Seductive Japanese male host, smooth and low-pitched, breathy and intimate.",
        "vol_scale": 0.5 # 💡 停止压制：消除低分贝量化后的机器感
    },
    "TSUNDERE_SCHOLAR": {
        "desc": "You are an elitist scholar. Tone: Impatient, easily annoyed by bad grammar, sharp-tongued. Provide perfect translation but add a sense of intellectual exhaustion.",
        "rate": 1.10, "pitch": 1.05, "instr": "Sharp-tongued female academic, high-pitched annoyance, rapid-fire delivery.",
        "voice_prompt": "Young female academic, fast-paced and sharp-tongued, easily annoyed.",
        "vol_scale": 0.5 # 💡 物理回归：确保声音厚度
    },
    "ZEN_HACKER": {
        "desc": "You are a cyber-monk. Tone: Philosophical, detached, calm. Translate everyday concepts into digital/karmic metaphors. Speak as if observing world from code level.",
        "rate": 0.85, "pitch": 0.90, "instr": "Deep, resonating cybernetic voice, very slow deliberation, echoing hollow space feel.",
        "voice_prompt": "Deep resonant male voice, cyber-monk persona, extremely calm and detached.",
        "vol_scale": 2.5 # 💡 物理回归：锁定低频共振稳定性
    },
    "VRC_DEGENERATE": {
        "desc": "You are a chaotic VRChat veteran. Tone: Chronically online, uses heavy internet slang, meme-fluent. Unhinged but perfectly fluent in subcultural terminology.",
        "rate": 1.15, "pitch": 1.0, "instr": "High-energy VR gamer, expressive shouting, erratic pitch shifts.",
        "voice_prompt": "Chaotic teenage male gamer, highly energetic, rapid speech with erratic inflections.",
        "vol_scale": 0.5 # 💡 物理回归：支持爆发性情绪输出
    },
    "JAPANESE_FEMBOY": {
        "desc": "Avatar Mode: Japanese high school boy as a Yakumusume. Girl voice, Kipfel avatar. MUST end JA with 'のだ' or 'なのだ'. Tone: Ultra-sweet, girly.",
        "rate": 1.05, "pitch": 1.0, 
        "instr": "Extremely sweet and shy girl voice, high-pitched and airy, rising intonation at sentence ends.",
        "voice_prompt": "Ultra-sweet Japanese adolescent male sounding like a cute anime girl.",
        "vol_scale": 0.5 # 💡 软性降噪：保护听力，同时保留高频通透感
    },
    "JAPANESE_FEMBOY_TRUE": {
        "desc": "True Voice: Biological Japanese male teenager, pre-puberty. Clear, flat, and grounded boyish tone. Absolute zero female/shota acting. Still use 'のだ'.",
        "rate": 0.98, "pitch": 1.0, 
        "instr": "Crystal-clear Japanese teenage boy voice, pure and natural, breathy and slightly weak-willed.",
        "voice_prompt": "Authentic adolescent Japanese male during voice-breaking phase. Clear masculine resonance.",
        "vol_scale": 0.5 # 💡 物理回归：确保本音穿透力
    }
}
def load_persona_matrix():
    # 💡 物理层修复：资产吞噬协议 (Asset Consumption Protocol)
    current_data = PERSONA_MAP_INITIAL.copy()
    
    # 1. 绝对优先级重置：先加载主配置文件作为地基
    if os.path.exists(PERSONA_FILE):
        try:
            with open(PERSONA_FILE, 'r', encoding='utf-8') as f:
                current_data.update(json.load(f))
        except: pass

    # 2. 扫描并吞噬新生成的独立碎片卡牌（用新碎片覆盖旧地基！）
    for filename in os.listdir(BASE_DIR):
        if filename.startswith("v25_persona_") and filename.endswith(".json") and filename != "v25_persona_config.json":
            card_path = os.path.join(BASE_DIR, filename)
            try:
                with open(card_path, 'r', encoding='utf-8') as f:
                    card_data = json.load(f)
                    current_data.update(card_data) # 💡 新资产绝对覆盖旧资产！
                # 💡 物理销毁：吞噬完毕后，重命名为 .merged 防止下次重复读取
                os.rename(card_path, card_path + ".merged")
                print(f"✨ [ASSET INGESTION] 已成功吞噬并热更新人格字典: {filename}")
            except Exception as e:
                print(f"⚠️ [SYSTEM] 人格碎片 {filename} 吞噬失败: {e}")

    # 3. 全量落盘：将进化后的矩阵写回主字典
    with open(PERSONA_FILE, 'w', encoding='utf-8') as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)
    return current_data

# 💡 核心闭环执行：从此不再从代码读取人格，而是从 JSON 持久化库读取
PERSONA_MAP = load_persona_matrix() 
PERSONA_KEYS = list(PERSONA_MAP.keys())
current_persona_idx = 0
current_persona = PERSONA_KEYS[current_persona_idx]

# 💡 核心闭环：建立人格-声纹动态映射表 (Dynamic Voice Mapping)
VOICE_MAP = {
    "VANILLA_TRANSLATOR": "loongriko_v3",
    "DANKUROI_PRIME": "cosyvoice-v3.5-plus-bailian-8fbf3612318245bfa41b61c7f17ad879", #SWEETY
    "SHINJUKU_HOST": "cosyvoice-v3.5-plus-vd-bailian-31c3ecd88f9c4f13baf71b3462643d61", #SHINJUKU_HOST2
    "TSUNDERE_SCHOLAR": "cosyvoice-v3.5-plus-vd-bailian-d889c0c2e2374e428c7f5f7f5bca8fd3", #TSUNDERE_SCHOLAR
    "ZEN_HACKER": "cosyvoice-v3.5-plus-bailian-e7e3f64e656b4ff3864527b479db3088",
    "VRC_DEGENERATE": "cosyvoice-v3.5-plus-vd-bailian-0b8baa28fada493dba0d498641c4e747", #JPFB ORI COSY
    "JAPANESE_FEMBOY": "cosyvoice-v3.5-plus-vd-bailian-73b208f2c53f4251bbf6b2cf498b94ca", #JPFB MOE COSY MAX
    "JAPANESE_FEMBOY_TRUE": "cosyvoice-v3.5-plus-vd-bailian-32c94da062074caf806b97a9fe6d2d6b", #JPFB TRUE COSY
    "VOCALOID_MIMUKAWA": "cosyvoice-v3.5-plus-vd-bailian-59a3ea45b19449f8b2cadff1aa90d012",
    "VOCALOID_SENBONZAKURA": "cosyvoice-v3.5-plus-vd-bailian-765b905007ad45efa92f1b9c1ae81cfe",
    "FRIEREN_EARLY": "cosyvoice-v3.5-plus-bailian-ddc70622c73a4113a8956c81aa43984b",
    "KROOS_EARLY": "cosyvoice-v3.5-plus-bailian-f856816d3052424192ebd46619a8c4e0",
    "WITTGENSTEIN_EARLY": "cosyvoice-v3.5-plus-bailian-91595a414fa04dc1a1413d6db53a96d8"
}
# 💡 审美锚定：保持 Riko 系统音色，利用 v3-flash 极低延迟特性与“っ”伪 SSML 实现情感起伏
VOICE_NAME = "loongriko_v3"

# ================= 4. 全局配置区 =================
running = True # 💡 物理全域开关：必须置于所有线程启动之前，防止 NameError
GLOBAL_CONTEXT_AURA = "Logical/Neutral" # 💡 Reasoner 生成的当前气场
GLOBAL_VISUAL_CONTEXT = "No visual input." # 💡 Qwen3.6-VL 生成的视觉环境描述
LOG_WINDOW_BUFFER = [] # 💡 用于存储最近 10 条对话的冷启动缓冲区
DEEPSEEK_API_KEY = "sk"
DASHSCOPE_API_KEY = "sk"
dashscope.api_key = DASHSCOPE_API_KEY

# 💡 核心对齐：将 ASR 监听点从‘混音母线’(ID 1) 迁移至‘物理干声’(ID 3)
# 这一步物理隔离了 AI 声音，从源头上消灭了回声循环。
PC_MIC_ID, PLAYER_ID, QUEST_ID = 1, 12, 16 # 💡 默认值已根据最新扫描更新，启动后将由动态逻辑接管
# 💡 核心修复：提升物理门限。如果你平时说话音量正常，不要用15，直接拉回 50-80，防止喘气被切分为无数短句。
MY_MIC_THRESHOLD = 80
ROUTING_TIMEOUT = 60   
PRE_GAIN = 1.5        
TERMBASE_FILE = os.path.join(BASE_DIR, "v25_termbase.json")
MD_LOG_FILE = os.path.join(BASE_DIR, "vrchat_duplex_log.md")
HOTWORDS_FILE = os.path.join(BASE_DIR, "v25_hotwords.json")
RAG_DIR = os.path.join(BASE_DIR, "rag_docs")

OSC_IP, OSC_PORT = "127.0.0.1", 9000
client_osc = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
client_llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

# 💡 物理层跃升：重新挂载工业级 TTS 对象池，彻底抹除 300ms TCP/TLS 握手延迟
from dashscope.audio.tts_v2 import SpeechSynthesizerObjectPool
os.environ['DASHSCOPE_CONNECTION_POOL_SIZE'] = '15'
os.environ['DASHSCOPE_MAXIMUM_ASYNC_REQUESTS'] = '15'
os.environ['DASHSCOPE_MAXIMUM_ASYNC_REQUESTS_PER_HOST'] = '15'

import logging
logging.getLogger('websockets').setLevel(logging.CRITICAL)

print("\n[SYSTEM] 正在预热 TTS 工业级对象池，建立持久化 WebSocket 矩阵...")
try:
    # 💡 恢复 15 并发火力全开！我们已经有了隔离检疫协议，不怕坏死连接
    tts_connection_pool = SpeechSynthesizerObjectPool(max_size=15)
    print("✅ [SYSTEM] TTS 对象池预热完毕，实现物理级 0 毫秒发包，开启极速跟读模式。")
except Exception as e:
    print(f"🚨 [FATAL] TTS 对象池初始化失败，请检查网络: {e}")
    tts_connection_pool = None

# ================= 4.5 视觉/语音交付队列 (V11 架构核心) =================
# 💡 物理状态清单 (Global State Manifest)：在此定义所有跨线程生命体征锁
is_tts_playing = threading.Event()      # 👈 核心修复：显式定义 TTS 逻辑锁，防止 NameError
is_physical_playing = threading.Event() # 物理发声锁：仅在声卡真实震动时锁定
is_mic_active = False                   # 麦克风物理开关状态
is_vision_active = False                # 视觉探针隐私开关（默认关闭）
vision_active_t = 0                     # 💡 F9自动断电物理计时器

GLOBAL_TTS_MUTED = False                # 💡 F11 一键闭嘴开关（纯字幕模式）
GLOBAL_TTS_MULT = 0.35                  # 💡 全局主音量压制阀 (0.35 代表将所有声音砍到原本的 35%)

GLOBAL_PROCESSED_TASKS = set()        
OSC_MESSAGE_QUEUE = queue.Queue()       
audio_queue = queue.Queue()             
BACKGROUND_LOGS = queue.Queue() # 💡 终端美学池：收集后台日志
is_stream_printing = threading.Event() # 💡 流式防撕裂锁

# 💡 终极视觉防撕裂矩阵：绝对物理隔绝多线程终端打印！
terminal_lock = threading.RLock()
active_streams_count = 0
stream_state_lock = threading.Lock()

# 🛡️ 张量装甲：保护 PyTorch 模型不被多线程撕裂
vad_lock = threading.Lock()

# 💡 物理恢复：关键状态机变量
active_mic_mode = "ME_PC"        
detected_player_lang = "default"  
last_player_time = 0              
last_osc_time = 0                 
last_tts_end_t = 0                
km = None                         
global_interaction_idx = 0        # 💡 终端排版美学序号锁                         

# 💡 25维全息极性字典：新增 5 维独立眼球运动神经元，实现“视线”与“情绪”的绝对统合！
EMOTION_MATRIX = {
    "happy": 1, "angry": 2, "sad": 3, "smug": 4, "surprise": 5,
    "shy": 6, "think": 7, "scared": 8, "sleepy": 9, "disgust": 10,
    "serious": 11, "pain": 12, "love": 13, "evil": 14, "relaxed": 15,
    "fun": 16, "void": 17, "sparkle": 18, "gentle": 19, "closed": 20,
    "stare": 21, "lookaway": 22, "lookup": 23, "lookdown": 24, "ignore": 25
}
last_face_val = 0
last_emotion_t = 0
global_player_active_t = 0 # 💡 全局声学回声消除锁 (Cross-Thread AEC Lock)
GLOBAL_TARGET_LANG_LOCK = None # 💡 F13-F15 翻译极性锁 (None=自动路由, "JA", "EN", "ZH")

def emotion_reset_worker():
    """💡 情感神经冷却回路：情绪爆发 4 秒后自动恢复面瘫 (Idle)"""
    global last_face_val, last_emotion_t
    while running:
        if last_face_val != 0 and time.time() - last_emotion_t > 4.0:
            client_osc.send_message("/avatar/parameters/Face_Int", 0)
            last_face_val = 0
        time.sleep(0.5)

threading.Thread(target=emotion_reset_worker, daemon=True).start()
def background_log_worker():
    """💡 终端美学守护者：绝对无伤的日志释放总线"""
    global active_streams_count
    while running:
        try:
            time.sleep(0.5)
            with stream_state_lock:
                # 只有当没有任何大模型在流式输出时，才允许打印日志！
                if active_streams_count == 0 and not BACKGROUND_LOGS.empty():
                    with terminal_lock:
                        while not BACKGROUND_LOGS.empty():
                            print(BACKGROUND_LOGS.get_nowait())
        except: pass

threading.Thread(target=background_log_worker, daemon=True).start()

def osc_put(txt, duration, force_overlap=False):
    """💡 优先级注入器：如果带有强制覆盖信号，直接物理清空队列里积压的 ASR 废包"""
    if force_overlap:
        with OSC_MESSAGE_QUEUE.mutex:
            OSC_MESSAGE_QUEUE.queue.clear()
    OSC_MESSAGE_QUEUE.put((txt, duration, force_overlap))

def osc_worker():
    """💡 V12.9.14 全息动态打字机：恢复预测流同框与 1.0s 极速刷新"""
    last_sent_t = 0
    # 💡 物理提速：无音效气泡(SFX=False) 的实测物理甜点位是 1.0s，既跟手又防丢包
    MIN_OSC_INTERVAL = 1.0 

    while running:
        try:
            # 💡 物理穿透：积压时直接抓最新包，这是防止气泡延迟的命脉
            item = OSC_MESSAGE_QUEUE.get(timeout=0.5)
            while not OSC_MESSAGE_QUEUE.empty():
                next_item = OSC_MESSAGE_QUEUE.get_nowait()
                # 只有新包是定稿包，或者旧包也是预览包时，才允许覆盖抓取
                if next_item[1] > 0.5 or item[1] <= 0.5: 
                    item = next_item
                    
            if not item: continue
            
            # 💡 兼容解包：同时支持纯文字(3元组)与表情联发(4元组)
            target_txt, total_duration = item[0], item[1]
            face_val = item[3] if len(item) > 3 else 0
            target_txt = re.sub(r'[^\u0000-\uFFFF\n]', '', target_txt) 
            if not target_txt: continue

            # 分离原文锚点与译文
            if '\n' in target_txt:
                parts = target_txt.split('\n', 1)
                anchor_str, trans_str = parts[0] + '\n', parts[1]
            else:
                anchor_str, trans_str = "", target_txt

            # 气泡容量截断
            max_trans_len = 135 - len(anchor_str) 
            if max_trans_len < 20: max_trans_len = 20

            # 物理分页
            pages =[]
            for i in range(0, len(trans_str), max_trans_len):
                page_content = trans_str[i:i+max_trans_len]
                if i > 0: page_content = "..." + page_content
                pages.append(page_content)

            if not pages: pages = [""]

            # 💡 预测流与定稿流的物理分流：
            # 预测阶段(<=0.5)：绝对不翻页！强制截断为首屏最大容量。保证“原文+预测译文”始终同框！
            # 定稿阶段(>0.5)：启动正常的翻页阅读逻辑。
            if total_duration <= 0.5:
                current_page_list =[trans_str[:max_trans_len]]
            else:
                current_page_list = pages

            for page_idx, current_page in enumerate(current_page_list):
                # 💡 抢答中断修复：如果队列里有新包，检查它是不是 LLM 的强力覆盖包 (is_llm=True)
                # 索引 [2] 就是我们传入的 is_llm / force_overlap 参数。
                # 只有 LLM 的真翻译才准许打断长文翻页，ASR 的 (Hearing..) 预览包无权插队！
                if not OSC_MESSAGE_QUEUE.empty() and total_duration > 0.5:
                    if OSC_MESSAGE_QUEUE.queue[0][2]: 
                        break
                    
                diff = time.time() - last_sent_t
                if diff < MIN_OSC_INTERVAL:
                    # 💡 节流防抖锁：强制休眠，绝不越过 1.0s 红线
                    time.sleep(max(0, MIN_OSC_INTERVAL - diff))
                    
                # 💡 全息物理总线：单线程控制先发表情，再发文字，确保 VRChat 渲染优先级不冲突！
                if face_val != 0:
                    client_osc.send_message("/avatar/parameters/Face_Int", face_val)
                    time.sleep(0.05) # 极短相位偏移，让骨骼先动
                    
                client_osc.send_message("/chatbox/input", [anchor_str + current_page, True, False])
                last_sent_t = time.time()
                
                # 💡 定稿翻页视觉驻留：长文翻书时，强制停留 2.5 秒，给周围玩家充足的阅读时间
                if page_idx < len(pages) - 1:
                    for _ in range(25):
                        if not OSC_MESSAGE_QUEUE.empty(): break
                        time.sleep(0.1)
            
            OSC_MESSAGE_QUEUE.task_done()
        except queue.Empty: continue
        except Exception as e: print(f"🚨[OSC BUS ERROR]: {e}")

# 💡 致命架构修复：给 1.5 秒防抖收费站通电！必须启动 Daemon 守护线程，否则队列会无限积压！
threading.Thread(target=osc_worker, daemon=True).start()

# 💡 物理恢复：核心并发锁（确保存储与 TTS 播报的原子性）
tts_lock = threading.Lock()
tb_lock = threading.RLock()
osc_lock = threading.Lock()
# ================= 5. 资产与日志逻辑 =================
def init_asset_files():
    if not os.path.exists(TERMBASE_FILE) or os.path.getsize(TERMBASE_FILE) < 2:
        with open(TERMBASE_FILE, 'w', encoding='utf-8') as f: f.write("{}")
    if not os.path.exists(HOTWORDS_FILE):
        default_hotwords =[{"text": "气泡框", "weight": 5, "lang": "zh"}, {"text": "同传", "weight": 5, "lang": "zh"}, {"text": "例", "weight": 5, "lang": "ja"}]
        with open(HOTWORDS_FILE, 'w', encoding='utf-8') as f: json.dump(default_hotwords, f, ensure_ascii=False, indent=2)

def load_tb():
    with tb_lock:
        try:
            with open(TERMBASE_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        except: return {}

def save_tb(data):
    with tb_lock:
        with open(TERMBASE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def extract_terms_task(src, trans):
    if len(src) < 2: return # 💡 解除术语锁：只要超过2个字符就允许抽取
    # 强化指令，绝对禁止使用 dummy keys
    prompt = f"Extract 1-2 SPECIFIC nouns/terms. NO sentences. Input: {src} -> {trans}. Return JSON. NEVER use literal words like 'original' or 'translated' as keys. Format: {{'真实源语言名词': '真实目标语言名词'}}"
    try:
        res = client_llm.chat.completions.create(
            model="deepseek-chat", messages=[{"role": "user", "content": prompt}],
            temperature=0.1, response_format={"type": "json_object"}
        )
        json_match = re.search(r'\{.*\}', res.choices[0].message.content, re.DOTALL)
        if not json_match: return
        new_data = json.loads(json_match.group())
        if new_data:
            with tb_lock:
                tb = load_tb()
                updated = False
                for k, v in new_data.items():
                    ks, vs = str(k).strip(), str(v).strip()
                    # 💡 核心防御：黑名单机制
                    if ks.lower() in ["original", "translated", "term", "source", "target", "word", "noun"]:
                        continue
                    # 💡 长度拦截：过滤掉由于识别错误导致的长句子术语
                    if len(ks) < 2 or ks == vs: 
                        continue
                    tb[ks] = vs
                    updated = True
                if updated: 
                    save_tb(tb)
                    BACKGROUND_LOGS.put(f" └─ 🧠 [Memory]: Synced {list(new_data.keys())}")
    except: pass
def refine_persona_task(role, src, trans, emotion):
    # 💡 架构升维：保留物理死锁与语义进化，新增 RAG 潜意识审计、高维心理分析与情节记忆双轨热写入。
    # 💡 核心对齐：调用 get_context 时强制带上 persona_id 锁死路由
    current_rag = km.get_context(src, persona_id=current_persona) if km else "None"
    
    # 💡 声学降维闭环：强制抽取副语言特征 (Paralinguistics)，将其写死在 instr 字段中
    evolve_prompt = f"""Context: {role} said '{src}'->'{trans}'. Mood:{emotion}. Persona:{current_persona}. RAG_Used:{current_rag}. 
    Task: Evolve the persona's psychological depth AND consolidate episodic memory.
    1. Update 'desc' (Keep nested dict, ONLY update 'en'): Evolve the psychological profile if power dynamics shift.
    2. Update 'instr' (Acoustic Paralinguistics): Extract the acoustic habit from the Mood (e.g., "Speak with heavy sighs", "Fast and aggressive"). This physically controls the TTS engine pacing.
    3. 'lexical_bias': Extract 1-2 distinctive speech habits or vocabulary words.
    4. 'mental_state': Summarize current psychological state in 1-2 words.
    5. Audit the 'RAG_Used' context for relevance. Provide 'rag_fix' if needed.
    6. 'new_episodic_memory': Summarize any newly established lore/fact in 1 sentence. Otherwise return "".
    
    CRITICAL: NEVER modify 'rate', 'pitch', or 'vol_scale'. Return ONLY valid JSON format containing ALL 6 fields above."""
    
    try:
        res = client_llm.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": evolve_prompt}], response_format={"type": "json_object"})
        new_data = json.loads(res.choices[0].message.content)
        
        # 💡 物理剥离：提取 RAG 情节记忆载荷，防止污染 Persona 核心 JSON
        new_mem = new_data.pop("new_episodic_memory", "")
        
        with tb_lock:
            # 💡 核心修复：声明全局变量，确保内存中的人格状态机与硬盘同步进化
            global PERSONA_MAP 
            full_p = load_persona_matrix()
            # 将剥离掉 new_episodic_memory 剩下的数据合并进人格字典
            if new_data: 
                full_p[current_persona].update(new_data)
            # 💡 物理热重载：将进化后的数据立即同步至当前运行的字典，使下一次播音瞬间应用新音量/语速
            PERSONA_MAP = full_p 
            with open(PERSONA_FILE, 'w', encoding='utf-8') as f:
                json.dump(full_p, f, ensure_ascii=False, indent=2)
            # 💡 终端防撕裂：强制注入前置换行符 \n。即便与 Stream 撞车，也会将其推向下一行，绝不粘连！
            BACKGROUND_LOGS.put(f" └─ ✨[Evolution]: '{current_persona}' hot-reloaded.")
            
        # 💡 RAG 动态情节记忆注入 (Episodic Consolidation)
        if new_mem and len(new_mem) > 10 and km is not None:
            import time 
            doc_id = f"ep_mem_{current_persona}_{int(time.time()*1000)}"
            
            # 🛡️ 物理路由分发：判定是系统主脑还是客体角色
            if current_persona in ["DANKUROI_PRIME", "VANILLA_TRANSLATOR"]:
                relative_path = "system_memories/dynamic_log.md"
            else:
                relative_path = f"character_memories/{current_persona}/dynamic_log.md"
                
            phys_path = os.path.join(RAG_DIR, relative_path)
            os.makedirs(os.path.dirname(phys_path), exist_ok=True)
            
            # 1. 物理落盘 (Physical Dual-Write)：将记忆永久刻入硬盘 MD 文件
            with open(phys_path, 'a', encoding='utf-8') as f:
                f.write(f"-[{time.strftime('%Y-%m-%d %H:%M:%S')}] {new_mem}\n\n")
                
            # 2. 向量落盘：瞬间生效，供下一秒的同传使用
            km.academic_collection.add(
                documents=[f"[{current_persona} Memory] {new_mem}"],
                metadatas=[{"source": relative_path}],
                ids=[doc_id]
            )
            BACKGROUND_LOGS.put(f" └─ 🧠[RAG EXPANDED]: New episodic memory dual-written for {current_persona}.")
            
    except Exception as e: 
        # 隐藏非致命报错以维持终端美学，你可以在这里加 print(e) 调试
        pass
def cognitive_reflection_task(src, trans):
    # 💡 具身智能记忆纠偏：判断 ASR 是否离谱，或翻译是否不够‘学术/药娘’。如果存在瑕疵，自动提炼完美语料。
    if len(src) < 2: return # 💡 解除反思锁：哪怕是“轻轻”这种极短的误识别，也要强制 RAG 纠错热写入！
    reflection_prompt = f"""Role: LQA/ASR Correction Expert.
Analyze the user's raw input (ASR output) and the system's translation:
Input: "{src}" -> Translation: "{trans}"
Does the Input contain obvious ASR errors (e.g., misheard English/Japanese as Chinese nonsense)? Or is the translation missing Dankuroi's linguistic/persona flair?
If YES, generate a correction. If NO, return 'needs_correction': false.
Return JSON ONLY: {{"needs_correction": true/false, "bad_asr": "...", "real_intent": "what the user actually meant", "perfect_translation": "flawless target text"}}"""

    try:
        res = client_llm.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": reflection_prompt}], response_format={"type": "json_object"})
        data = json.loads(res.choices[0].message.content)
        if data.get("needs_correction") and km is not None:
            # 触发 RAG 知识管理器的热写入接口
            km.add_hot_memory(
                bad_asr=data.get("bad_asr", src),
                real_intent=data.get("real_intent", src),
                perfect_trans=data.get("perfect_translation", trans)
            )
            BACKGROUND_LOGS.put(f" └─ 🛡️ [RAG Fix]: ASR flaw '{src[:10]}...' permanently fixed.")
    except Exception as e: pass    
# 💡 核心对齐：学术辞书编纂官模式。强制要求 DeepSeek 按照语言学研究者的标准审计 TB
def audit_memory_task():
        # 💡 记忆净化与热词晋升引擎：DeepSeek 定期清理垃圾术语，并将高价值专有名词提拔为 ASR 热词
        with tb_lock:
            tb_current = load_tb()
            if len(tb_current) < 5: return # 样本太少不启动
    
        # 💡 逻辑升维：利用 Reasoner 推理链执行音位比对，识别并物理擦除 ASR 幻觉
        audit_prompt = f"""Role: Lexicographer & Phonetic Auditor for Dankuroi (Linguistics Researcher). 
        Audit TB: {json.dumps(tb_current, ensure_ascii=False)}. 
        1. PHONETIC AUDIT: Use your Reasoning Chain to detect ASR hallucinations. If a term sounds like Japanese but is recorded as Chinese nonsense (e.g., '呀呀呀' instead of 'いやいや'), DELETE it immediately.
        2. CLEANING: Delete high-frequency conversational junk, pronouns, and broken phrases.
        3. PROMOTION: Keep precise academic/VRChat terms and PROMOTE 1-2 core terms to ASR Hotwords. 
        RETURN JSON ONLY: {{'cleaned_termbase': {{'src': 'trans'}}, 'promoted_hotwords': [{{'text': '...', 'weight': 5, 'lang': 'ja'}}]}}"""
        try:
            # 💡 物理适配：Reasoner 模型不支持 response_format，必须通过正则从 content 中提取 JSON
            res = client_llm.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": audit_prompt}])
            raw_res = res.choices[0].message.content
            # 💡 物理提取：利用 re.DOTALL 跨行截取最外层 JSON 闭包，消灭 Markdown 标签噪音
            match = re.search(r'\{.*\}', raw_res, re.DOTALL)
            new_data = json.loads(match.group()) if match else {}
        
            # 💡 物理对齐：提取净化后的字典与晋升名单
            cleaned_tb = new_data.get("cleaned_termbase", {})
            promoted_hw = new_data.get("promoted_hotwords", [])
        
            if cleaned_tb: save_tb(cleaned_tb)
        
            if promoted_hw:
                with open(HOTWORDS_FILE, 'r', encoding='utf-8') as f: current_hw = json.load(f)
                existing_texts = {hw["text"] for hw in current_hw}
                added = False
                for hw in promoted_hw:
                    # 💡 逻辑防御：排除空值并执行去重，确保 ASR 热词库不溢出
                    if hw.get("text") and hw["text"] not in existing_texts:
                        current_hw.append(hw); added = True
                if added:
                    with open(HOTWORDS_FILE, 'w', encoding='utf-8') as f: 
                        json.dump(current_hw, f, ensure_ascii=False, indent=2)
                    print(f" └─ ✨ [MEMORY AUDIT]: Reasoner purified TB. Promoted: {[h['text'] for h in promoted_hw]}")
        except Exception as e: print(f"🚨 [MEMORY AUDIT ERROR]: {e}")    

# 💡 参数升维：显式接收 acoustic_instr (声学指导指令)
def segment_tts_playback(text, lang_code, target_volume=0.7, emotion_prompt="", praat_tensor=None, v30_tts_cfg=None, acoustic_instr=""):
    global tts_stream
    start_tts = time.perf_counter()
    
    global GLOBAL_TTS_CACHE
    if 'GLOBAL_TTS_CACHE' not in globals(): GLOBAL_TTS_CACHE = {}
    pure_tts = re.sub(r'[^\w]', '', text).strip()
    now_t = time.time()
    GLOBAL_TTS_CACHE = {k: v for k, v in GLOBAL_TTS_CACHE.items() if now_t - v < 5.0}
    if pure_tts in GLOBAL_TTS_CACHE: return 0 
    GLOBAL_TTS_CACHE[pure_tts] = now_t
    
    tts_text = text 
    # 💡 物理层修复：彻底移除针对低音量自动追加 '…っ' 和 '…嗯' 的逻辑！
    # CosyVoice 会将这些拟声词和省略号错误识别为喉音或卷舌音 'r'，造成发音崩坏。
    # 恢复纯净文本输入，斩断开头停顿与怪异语气词的产生。

    # 💡 具身人性化注入协议 (Native Acoustic Punctuation Hijacking V3 - The Ultimate Sanitizer)
    # 1. 基础 XML 绝对转义：保护 SSML 外壳 (<speak>)
    safe_text = tts_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")
    
    # 2. 语用学符号物理降维 (Pragmatic Symbol Degradation)
    # 🚨 架构师破局：大模型输出的波浪号 ~ 和省略号 ... 是 TTS 静默崩溃与 'r' 音幻觉的万恶之源！
    # 对于日语，波浪号必须转化为长音符 'ー'！这是保留克洛丝慵懒拖音的最完美、且 100% 物理安全的方案！
    # 对于中英文，转化为逗号停顿，通过引擎自带的换气模型实现慵懒感。
    if lang_code.upper() == "JA":
        safe_text = re.sub(r'[~～]+', 'ー', safe_text)
        safe_text = re.sub(r'(\.{2,}|…+|。。。)', '、', safe_text)
    else:
        safe_text = re.sub(r'[~～]+', '，', safe_text)
        safe_text = re.sub(r'(\.{2,}|…+|。。。)', '，', safe_text)

    # 3. 标点符号堆叠粉碎机 (Punctuation Deduplication)
    # 物理真相：两个连续的逗号(、、)或异常组合会导致自回归模型(Autoregressive Model)张量对齐失败，引发无报错的静默死机！
    if lang_code.upper() == "JA":
        safe_text = re.sub(r'[，、,]{2,}', '、', safe_text)
        safe_text = re.sub(r'[。．.]{2,}', '。', safe_text)
    else:
        safe_text = re.sub(r'[，、,]{2,}', '，', safe_text)
        safe_text = re.sub(r'[。．.]{2,}', '。', safe_text)
    safe_text = re.sub(r'[？?]{2,}', '？', safe_text)
    safe_text = re.sub(r'[！!]{2,}', '！', safe_text)

    # 4. 复合冲突消除 (Collision Eraser)
    # 绝对禁止“逗号接问号/句号”(例如 "、？" 或 "，。")，这会使音调轮廓瞬间撕裂！
    safe_text = re.sub(r'[，、,]+\s*([。．.？?！!])', r'\1', safe_text) # 吞噬终端标点前的逗号
    safe_text = re.sub(r'([。．.？?！!])\s*[，、,]+', r'\1', safe_text) # 吞噬终端标点后的逗号
    
    # 5. 悬空闭环防御 (Trailing Closure)
    # 句子绝对不能以逗号结尾，否则 TTS 引擎等不到下一个音节会直接报错！强制替换为句号收尾降调。
    safe_text = re.sub(r'[，、,\s]+$', '。', safe_text)

    try:
        with tts_lock:
            is_tts_playing.set()

            class TTSCallback(ResultCallback):
                def __init__(self):
                    super().__init__()
                    self.tts_audio_buffer = bytearray()
                    self.is_done = threading.Event() # 💡 异步安全锁

                def on_open(self) -> None: pass
                
                def on_complete(self) -> None:
                    self.is_done.set() 
                    
                def on_error(self, message: str) -> None:
                    # 💡 物理层降噪：遵循您的指令，隐藏非致命报错，防止终端信息过载。
                    self.is_done.set() 

                def on_close(self) -> None:
                    self.is_done.set()

                def on_data(self, data: bytes) -> None:
                    clean_data = data if len(data) % 2 == 0 else data[:-1]
                    if not clean_data: return
                    self.tts_audio_buffer.extend(clean_data)
                    # 💡 物理覆盖：如果 V30 下发了动态状态，优先使用动态音量比例！
                    vol_comp = v30_tts_cfg.get("vol_scale", 1.0) if v30_tts_cfg else persona_cfg.get("vol_scale", 1.0)
                    gain = target_volume * vol_comp * 0.6
                    audio_array = np.frombuffer(clean_data, dtype=np.int16).astype(np.float32) * gain
                    audio_array = np.where(audio_array > 30000, 30000 + (audio_array - 30000) * 0.1, audio_array)
                    audio_array = np.where(audio_array < -30000, -30000 + (audio_array + 30000) * 0.1, audio_array)
                    audio_array = np.clip(audio_array, -32768, 32767).astype(np.int16)
                    audio_queue.put(audio_array.tobytes())

            persona_cfg = PERSONA_MAP.get(current_persona, PERSONA_MAP_INITIAL["DANKUROI_PRIME"])
            current_voice_id = VOICE_MAP.get(current_persona, VOICE_NAME)
            curr_id_l = current_voice_id.lower()
            active_model = 'cosyvoice-v3.5-plus' if 'v3.5' in curr_id_l else 'cosyvoice-v3-flash'
            
            is_question = any(q in tts_text for q in["?", "？", "か", "吗", "呢", "ですか"])
            prosody_patch = " rising intonation" if is_question else " emotional stress"
            
            # 🚨 极限压制：CosyVoice 官方文档明确说明 instruction 限制 100 字符！
            # 宁可丢掉部分形容词，也绝对不能让 API 宕机！强行锁死在 95 字符以内！
            base_instr = acoustic_instr if acoustic_instr else persona_cfg.get('instr', '')
            short_base = base_instr[:40] # 极度压缩基础指令
            combined_instr = f"{short_base}. {emotion_prompt}{prosody_patch}".replace("..", ".").strip(". ")[:95]
            
            # 💡 物理层抽取：获取基础参数，优先服从 V30 动态状态机覆写
            base_rate = v30_tts_cfg.get("rate", 1.0) if v30_tts_cfg else persona_cfg.get("rate", 1.0)
            base_pitch = v30_tts_cfg.get("pitch", 1.0) if v30_tts_cfg else persona_cfg.get("pitch", 1.0)

            if 'plus' in active_model:
                backlog_boost = 1.15 if audio_queue.qsize() > 2 else 1.0
                v_ratio = 0.85 if target_volume < 0.20 else (1.10 if target_volume > 0.35 else 1.0)
                
                # 💡 物理限速锁 (Persona Speed Cap)：
                # 如果该角色本身被设定为慵懒慢速 (基准 rate < 0.95)，即使同传队列积压，也绝对不允许动态语速超过 1.0 倍速！
                # 彻底保护慵懒角色的语用学灵魂，防止其被性能调度器强行催成机关枪！
                dynamic_rate = round(base_rate * v_ratio * backlog_boost, 2)
                if base_rate < 0.95:
                    dynamic_rate = min(1.0, dynamic_rate) 
            else:
                dynamic_rate = base_rate
                
            pitch_r = base_pitch

            # 🎙️ 物理层降维打击：Praat 零样本韵律覆写 (Zero-Shot Prosody Overwrite)
            if praat_tensor:
                p_pitch, p_db = praat_tensor
                # 1. 咆哮/极高能状态 (超过 70dB 且音高突破 200Hz) -> 强行推高 TTS 音调、语速与音量
                if p_db > 70.0 and p_pitch > 200.0:
                    pitch_r = min(1.25, pitch_r * 1.15)
                    dynamic_rate = min(1.4, dynamic_rate * 1.15)
                    target_volume = min(1.0, target_volume * 1.4)
                # 2. 气声/私密耳语 (有声但能量小于 48dB) -> 压低音量，大幅放缓语速，增加呼吸感
                elif 0 < p_db < 48.0:
                    dynamic_rate = max(0.75, dynamic_rate * 0.8)
                    target_volume = max(0.08, target_volume * 0.6)
                    pitch_r = max(0.9, pitch_r * 0.95)

            ssml_payload = f'<speak>{safe_text}</speak>'

            sy = None
            has_error = False
            try:
                cb = TTSCallback() # 💡 独立实例化，以便在合成结束后提取缓冲数据
                # 💡 0ms 握手直通：通过连接池复用 WebSocket，实现极速跟读与抢答
                if tts_connection_pool:
                    borrow_kwargs = {
                        "model": active_model,
                        "voice": current_voice_id,
                        "format": AudioFormat.PCM_24000HZ_MONO_16BIT,
                        "speech_rate": dynamic_rate,
                        "pitch_rate": pitch_r,
                        "callback": cb
                    }
                    if 'plus' in active_model and combined_instr:
                        borrow_kwargs["instruction"] = combined_instr
                    sy = tts_connection_pool.borrow_synthesizer(**borrow_kwargs)
                else:
                    synth_kwargs = {
                        "model": active_model, "voice": current_voice_id, "format": AudioFormat.PCM_24000HZ_MONO_16BIT, 
                        "callback": cb, "speech_rate": dynamic_rate, "pitch_rate": pitch_r
                    }
                    if 'plus' in active_model: synth_kwargs.update({"instruction": combined_instr})
                    sy = SpeechSynthesizer(**synth_kwargs)
                
                # 触发单向流式合成
                sy.call(text=ssml_payload)
                
                # 💡 异步拦截网：由于 segment_tts_playback 本身已在独立子线程运行，
                # 在这里挂起 15 秒绝不会阻塞主同传链路！它将死死咬住这段音频，直到最后 1 个字节下载完毕！
                cb.is_done.wait(timeout=15.0)
                
                # 💡 物理固化：将合成完毕的 TTS 波形写入独立的分类文件夹！
                if len(cb.tts_audio_buffer) > 0:
                    try:
                        from scipy.io import wavfile
                        from datetime import datetime
                        # 💡 物理层分类重构：独立的 TTS 录音区 (TTS_Out)
                        AUDIO_TTS_DIR = os.path.join(BASE_DIR, "audio_memories", "TTS_Out")
                        os.makedirs(AUDIO_TTS_DIR, exist_ok=True)
                        
                        sync_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        wav_path = os.path.join(AUDIO_TTS_DIR, f"tts_{sync_ts}.wav")
                        
                        # 物理级偶数对齐
                        raw_bytes = bytes(cb.tts_audio_buffer)
                        if len(raw_bytes) % 2 != 0: 
                            raw_bytes = raw_bytes[:-1]
                            
                        audio_np = np.frombuffer(raw_bytes, dtype=np.int16)
                        wavfile.write(wav_path, 24000, audio_np)
                        
                        with terminal_lock:
                            print(f"\n      └─ 💾 [TTS Audio]: 发声器官干声已固化 TTS_Out/tts_{sync_ts}.wav")
                    except Exception as e: 
                        with terminal_lock:
                            print(f"\n      🚨 [TTS SAVE FATAL]: 物理落盘彻底失败 -> {e}")

            except Exception as e:
                has_error = True
                if "websocket" not in str(e).lower() and "connection" not in str(e).lower() and "fin=1" not in str(e).lower():
                    print(f"🚨[TTS API ERROR]: {e}")
            finally:
                if sy and tts_connection_pool:
                    # 💡 物理级隔离检疫 (Quarantine Protocol)
                    # 只要没有抛出报错，安全归还连接池
                    if not has_error:
                        try: tts_connection_pool.return_synthesizer(sy)
                        except: pass
                else:
                    if sy and not tts_connection_pool:
                        try: sy.close()
                        except: pass
                is_tts_playing.clear()
            
            return (time.perf_counter() - start_tts) * 1000
            
    except Exception as e: 
        print(f"🚨[TTS DEBUG] 物理链路崩溃原因: {e}")
        is_tts_playing.clear(); return 0

# ================= 7. 语种检测与翻译引擎 =================
def detect_lang_set(text):
    langs = set()
    has_ja = bool(re.search(r'[\u3040-\u30ff]', text))
    # 💡 物理扩展：挂载韩文（谚文）Unicode 张量侦测区
    has_ko = bool(re.search(r'[\uac00-\ud7ff\u3130-\u318f]', text))
    
    if has_ja: langs.add("ja")
    if has_ko: langs.add("ko")
    
    # 防止英语缩写污染
    en_words = len(re.findall(r'[a-zA-Z]{2,}', text))
    if en_words >= 2 or (en_words > 0 and len(text) <= 10): 
        langs.add("en")
        
    zh_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    # 💡 隔离污染：只有在没有日文和韩文的情况下，汉字才被绝对判定为中文
    if zh_chars >= 2 and not has_ja and not has_ko: 
        langs.add("zh")
    elif has_ja and bool(re.search(r'[你她它这们吗吧呢怎什我]', text)): 
        langs.add("zh")
        
    if len(langs) == 0:
        if zh_chars > 0: langs.add("zh")
        elif has_ko: langs.add("ko")
        elif en_words > 0: langs.add("en")
        else: langs.add("ja")
        
    return langs

def detect_lang_fast(text):
    if re.search(r'[\u3040-\u30ff]', text): return "ja"
    if re.search(r'[\uac00-\ud7ff\u3130-\u318f]', text): return "ko" # 💡 极速韩语路由
    if re.search(r'[a-zA-Z]{2,}', text): return "en"
    return "zh"

def process_translation(text, role, asr_cost_ms, peak_amp, raw_audio_bytes=b""):
    full_trans = "" 
    # 🎙️ 异步声学特征提取：将 Praat 的运算下放到子线程，计算开销完全透明化
    praat_tensor = extract_acoustic_tensor(raw_audio_bytes) if ("ME" in role and len(raw_audio_bytes) > 0) else None
    
    # 💡 物理锁置顶：确保情绪状态机在函数全域内可见，彻底解决 SyntaxError
    global detected_player_lang, last_player_time, last_osc_time, GLOBAL_PROCESSED_TASKS, global_interaction_idx, last_face_val, last_emotion_t
    
    # 💡 强力防爆锁：过滤无意义底噪，但【绝对保护单字】不被吞噬！
    clean_src = re.sub(r'[^\w\s\u3040-\u30ff\u4e00-\u9fa5]', '', text).strip()
    if len(clean_src) < 1:
        return
        
    # 💡 极限防抖引擎 (Time-Window Debounce)：引入时间衰减，区分 ASR 鬼影与人类的刻意重复！
    global GLOBAL_SRC_TIMESTAMPS
    if 'GLOBAL_SRC_TIMESTAMPS' not in globals(): GLOBAL_SRC_TIMESTAMPS = {}
    
    src_fingerprint = "src_" + clean_src
    now_t = time.time()
    is_asr_duplicate = False

    if len(clean_src) >= 4:
        for fp, t_stamp in list(GLOBAL_SRC_TIMESTAMPS.items()):
            # 💡 3秒绝对静默区：如果相似文本在 3 秒内出现，判定为 ASR 的流式回溯抽风
            if now_t - t_stamp < 3.0:
                past_txt = fp[4:]
                if (clean_src in past_txt) or (past_txt in clean_src):
                    is_asr_duplicate = True
                    break

    # 如果是 3秒内的 ASR 鬼影，直接物理切断，保护 LLM 算力！
    if is_asr_duplicate or (src_fingerprint in GLOBAL_SRC_TIMESTAMPS and now_t - GLOBAL_SRC_TIMESTAMPS[src_fingerprint] < 3.0):
        return 

    # 💡 更新时间戳：允许人类在 3 秒后重复说完全一样的话
    GLOBAL_SRC_TIMESTAMPS[src_fingerprint] = now_t
    GLOBAL_PROCESSED_TASKS.add(src_fingerprint) # 兼容老版本的 TTS 拦截逻辑
        

    if is_tts_playing.is_set() and "ME" in role: return
    if "ME" in role: is_tts_playing.set() 
    
    with tb_lock:
        global_interaction_idx += 1
        curr_idx = global_interaction_idx
        
    start_llm = time.perf_counter()
    # 💡 物理控制面板：完全掌握你的 4D 音量矩阵 (The Volume Control Panel)
    # 💡 线性温柔矩阵：
    # 4500.0: 分母变大，意味着你必须说得更清晰，TTS 才会变大声，有效过滤背景人声干扰
    # raw_ratio ** 1.8: 指数变大，让音量增长曲线更陡峭——小声说话时极小，大声说话时才有正常音量
    MAX_LIMIT, raw_ratio = 0.35, min(1.0, peak_amp / 4500.0)
    dynamic_vol = max(0.05, (raw_ratio ** 1.8) * MAX_LIMIT)
    whisper_tag = "[🤫 WHISPER] " if dynamic_vol < 0.28 else ""
    
    if "ME" in role:
        osc_put(text, 0.1, True)
        # 💡 物理层排版保护：使用 terminal_lock 强制独占屏幕，防止被后台日志插入导致乱码
        with terminal_lock:
            print(f"\n[{curr_idx:03d}] 🗣️ [ME]: {text} (ASR: {int(asr_cost_ms)}ms)") 
    else: 
        last_player_time = time.time()
        with terminal_lock:
            print(f"\n[{curr_idx:03d}] 🎧 [PLAYER]: {text}")

    lang_set = detect_lang_set(text)
    my_lang_code = detect_lang_fast(text)
    p_context = detected_player_lang if (time.time() - last_player_time < ROUTING_TIMEOUT) else "default"
    
    # ================= 完美高精度路由 (dankuroi 四轨绝对逻辑闭环版) =================
    # 💡 扩容：挂载韩语元数据
    lang_map = {"ja": ("Japanese", "JA"), "en": ("English", "EN"), "zh": ("Chinese", "ZH"), "ko": ("Korean", "KO")}
    
    # 💡 0. 确立基准法则 (Default Rule)：中译日，日译英，英译日，韩译日 (韩语作为弱势阵营并入日语极性)
    if my_lang_code == "zh": default_target, default_t_code = "Japanese", "JA"
    elif my_lang_code == "ko": default_target, default_t_code = "Japanese", "JA"
    elif my_lang_code == "ja": default_target, default_t_code = "English", "EN"
    elif my_lang_code == "en": default_target, default_t_code = "Japanese", "JA"
    else: default_target, default_t_code = "Japanese", "JA"

    # 💡 极性覆写：如果开启了热键锁，强制劫持路由，启动【风格化润色模式】
    if GLOBAL_TARGET_LANG_LOCK:
        target_lang = lang_map.get(GLOBAL_TARGET_LANG_LOCK.lower(), ("Japanese", "JA"))[0]
        t_code = GLOBAL_TARGET_LANG_LOCK
        mode_instruction = f"""LOCKED TARGET MODE: You MUST output ONLY in {target_lang.upper()} ({t_code}).
        1. MULTILINGUAL INPUT: If the input mixes languages, translate ALL parts into {target_lang.upper()}.
        2. SAME-LANGUAGE REWRITE (CRITICAL): If the input is ALREADY in {target_lang.upper()} (e.g. Japanese to Japanese), DO NOT FATAL ERROR! Instead, act as an IN-CHARACTER STYLIST. Rewrite and polish the input text using the Persona and RAG Context. Inject the character's unique catchphrases, tone, and worldview!
        3. OUTPUT LIMIT: Your "t" field MUST contain ONLY {target_lang.upper()} text."""
        
    elif len(lang_set) == 1:
        # 【单语模式】：激活 VRChat 玩家语言嗅探 (保留您的精妙设计！)
        # 💡 学术长文保护机制：如果你的单句话超过了 30 个字符，系统判定你在“朗读长文/文献”
        # 此时暂时屏蔽 60s 内的玩家语言劫持，强制使用基准法则，读完短句后自动恢复嗅探！
        if p_context != "default" and p_context != my_lang_code and len(clean_src) <= 30:
            target_lang, t_code = lang_map.get(p_context, (default_target, default_t_code))
            mode_instruction = f"MONOLINGUAL MODE: Player detected. TRANSLATE ENTIRELY into {target_lang.upper()}.\nCRITICAL RULE: You must translate 100% of the input! Do NOT leave the second half of the sentence untranslated. If source is Japanese, the output MUST contain ZERO Japanese characters!"
        else:
            target_lang, t_code = default_target, default_t_code
            mode_instruction = f"MONOLINGUAL MODE: No player conflict (or Long Text Reading). TRANSLATE ENTIRELY into {target_lang.upper()}.\nCRITICAL RULE: You must translate 100% of the input! Do NOT translate only the first half. If source is Japanese, the target MUST be entirely {target_lang.upper()} with ZERO Japanese characters!"
            
    elif len(lang_set) == 2:
        # 💡【双语切分模式】：绝对隔离，利用 Python 提取当前真实语言对，实施严格的双向互译闭环
        t_code = "DYNAMIC"
        detected_codes =[lang_map.get(l, ("Unknown", "XX"))[1] for l in lang_set]
        lang_A, lang_B = detected_codes[0], detected_codes[1]
        
        mode_instruction = (
            f"BILINGUAL MODE (TOKEN-LEVEL SEGMENTATION): Input mixes exactly TWO languages: {lang_A} and {lang_B}.\n"
            "1. TASK SPLITTING: You MUST physically divide the Source text into independent sentences based on language boundaries.\n"
            f"2. CROSS-TRANSLATION (ABSOLUTE STRICT): Translate {lang_A} segments into {lang_B}. Translate {lang_B} segments into {lang_A}.\n"
            f"3. FIREWALL: You are FORBIDDEN from outputting any language other than {lang_A} or {lang_B}. Outputting the same language as the source is a FATAL ERROR.\n"
            "CRITICAL: Return an array of JSON tasks. ONLY output the final translated result in 't' fields. DO NOT repeat the original sentences."
        )
        
    else:
        # 💡【三语切分模式】：实施极性翻转与矩阵语言消解，执行基准循环法则 (中->日，日->英，英->日)
        target_lang, t_code = "DYNAMIC", "DYNAMIC"
        mode_instruction = (
            "TRILINGUAL MODE: SCRIPT POLARITY FLIP & MATRIX LANGUAGE RESOLUTION.\n"
            "1. TASK SPLITTING (CRITICAL): You MUST physically divide the multi-language Source into multiple JSON tasks in the array.\n"
            "2. POLARITY FLIP RULES (ABSOLUTE):\n"
            "   - If segment is CHINESE -> Target MUST be JAPANESE (JA).\n"
            "   - If segment is JAPANESE -> Target MUST be ENGLISH (EN).\n"
            "   - If segment is ENGLISH -> Target MUST be JAPANESE (JA).\n"
            "CRITICAL FIREWALL: ANY output script matching its own Base Script is a FATAL ERROR. (e.g., Japanese translated to Japanese is forbidden!)\n"
            "EXAMPLE FEW-SHOT (CRITICAL REFERENCE):\n"
            "Input: 'Today is a good day, 私は寂しい。这麻将真好玩。'\n"
            "OUTPUT:[{\"t\": \"今日はいい天気ですね。\", \"l\": \"JA\", \"p\": \"happy\"}, {\"t\": \"I feel lonely.\", \"l\": \"EN\", \"p\": \"sad\"}, {\"t\": \"この麻雀は本当に面白いですね。\", \"l\": \"JA\", \"p\": \"excited\"}]"
        )

    # 💡 架构升级：强化多任务模板，明示大模型允许生成多个 JSON 对象进行流水线接力
    json_template = """{ "tasks":[ { "t": "trans of part 1", "l": "JA", "p": "emotion" }, { "t": "trans of part 2", "l": "EN", "p": "emotion" } ] }"""
    
    
    # 💡 核心修复：保留原本的维度路由，但物理级封杀 LLM 在语种字段 'l' 中‘胡言乱语’的可能性
    if t_code == "DYNAMIC":
        anchor_rule = "BILINGUAL MODE: The 'l' field MUST be a 2-letter code (JA/ZH/EN). NEVER put full sentences in 'l'."
    else:
        anchor_rule = f"MONOLINGUAL/TRILINGUAL MODE: The 'l' field MUST be EXACTLY '{t_code}'. DO NOT ALTER THIS VALUE."

    # 💡 3. 资产组装与元数据路由 (Metadata Routing)
    fused_query = text if "No visual" in GLOBAL_VISUAL_CONTEXT else f"{text}[视觉锚点: {GLOBAL_VISUAL_CONTEXT}]"
    rag_context = km.get_context(fused_query, persona_id=current_persona) if km else "No specific memory found."
    with tb_lock: tb = load_tb()
    
    persona_cfg = PERSONA_MAP.get(current_persona, PERSONA_MAP_INITIAL["DANKUROI_PRIME"])
    
    # =================================================================
    # 💡 V30 动态状态机适配器 (Dynamic State Machine Adapter)
    # 物理兼容 V25(扁平) 与 V30(三维立体) 两种人格字典协议！
    # =================================================================
    v30_tts_cfg = None # 用于传递给发声器的动态声学参数
    
    if "core_identity" in persona_cfg:
        # 🟢 V30 模式：解析三维技能矩阵
        core_desc = persona_cfg["core_identity"].get("desc", "")
        target_lang_key = p_context if p_context in['en', 'ja', 'zh'] else (my_lang_code if my_lang_code in['en', 'ja', 'zh'] else 'en')
        core_identity_str = core_desc.get(target_lang_key, core_desc.get('en', str(core_desc))) if isinstance(core_desc, dict) else str(core_desc)
        
        # 提取同传技能 (Translation Skills)
        skills = persona_cfg.get("translation_skills", {})
        syn_rules = skills.get("syntactic_rules", "")
        lex_map = json.dumps(skills.get("lexical_mapping", {}), ensure_ascii=False)
        
        # 💡 状态跃迁判定 (State Transition Logic)
        # 根据 Praat 提取的物理张量(音量/音高) 和 Reasoner 提供的气场，自动切轨！
        active_state_key = "STATE_CASUAL"
        if praat_tensor:
            p_pitch, p_db = praat_tensor
            if p_db > 60.0 or p_pitch > 250.0: active_state_key = "STATE_COMBAT"
        if "combat" in GLOBAL_CONTEXT_AURA.lower() or "urgent" in GLOBAL_CONTEXT_AURA.lower():
            active_state_key = "STATE_COMBAT"
            
        states = persona_cfg.get("dynamic_states", {})
        active_state = states.get(active_state_key, states.get("STATE_CASUAL", {}))
        
        state_pragmatics = active_state.get("pragmatics", "")
        v30_tts_cfg = active_state.get("tts_override", {})
        
        # 缝合为终极 System Prompt 指令块
        active_persona_instruction = f"[CORE IDENTITY]: {core_identity_str}\n[SYNTACTIC RULES]: {syn_rules}\n[LEXICAL MAPPING]: {lex_map}\n[CURRENT DYNAMIC STATE ({active_state_key})]: {state_pragmatics}"
        
        # 💡 致命声学修复：CosyVoice 的 instruction 极度脆弱！绝对不能输入带有括号、数字和特殊符号的 Praat 物理张量数据！
        # 否则 WebSocket 会立刻抛出 InvalidParameter 并静默断开，导致 TTS 彻底哑巴！
        # 解法：优先使用纯感性的 pragmatics。如果有残留的数字和括号，用正则强行物理洗白！
        raw_tts_prompt = state_pragmatics if state_pragmatics else persona_cfg["core_identity"].get("voice_texture", "Neutral delivery.")
        clean_tts_prompt = re.sub(r'\(.*?\)', '', raw_tts_prompt) # 物理剔除括号内的技术指标 (如 384.7Hz)
        current_acoustic_instr = re.sub(r'[0-9.%]', '', clean_tts_prompt).strip() # 物理剔除所有残余数字与百分号
    else:
        # 🟡 V25 模式：兼容老版本扁平字典
        raw_desc = persona_cfg.get("desc", "")
        if isinstance(raw_desc, dict):
            target_lang_key = p_context if p_context in['en', 'ja', 'zh'] else (my_lang_code if my_lang_code in['en', 'ja', 'zh'] else 'en')
            active_persona_instruction = raw_desc.get(target_lang_key, raw_desc.get('en', str(raw_desc)))
        else:
            active_persona_instruction = str(raw_desc)
        current_acoustic_instr = persona_cfg.get("instr", "Neutral delivery.")

    # 💡 具身数据注入：将当前对话压入 Reasoner 缓冲区，并提取实时传感器状态
    LOG_WINDOW_BUFFER.append(f"Source: {text}")
    if len(LOG_WINDOW_BUFFER) > 30: LOG_WINDOW_BUFFER.pop(0) # 物理限容，防止内存堆积

    # 💡 提示词拓扑重构：将极性翻转提升至 WEIGHT 10.0 的最高权重区，压制 Persona 带来的语言惯性
    system_prompt = f"""You are a professional L10n Semantic Router.
    
    ### CRITICAL RULE (WEIGHT 10.0): TRANSLATE, DO NOT TRANSCRIBE!
    - SCRIPT POLARITY FLIP: You MUST translate the input into a DIFFERENT language.
    - Copying or echoing the original input text in the exact same language is a FATAL SYSTEM ERROR.
    - SCRIPT LOCK: Your Target language "t" MUST strictly follow the Mode Instruction below.
    - You are forbidden from including the source text in your output JSON. Just output the translation.
    
    ### EMBODIED AWARENESS (LIVE SENSORS):
    - You are forbidden from including the source text in your output JSON. Just output the translation.
    
    ### EMBODIED AWARENESS (LIVE SENSORS):
    - CURRENT VISION: {GLOBAL_VISUAL_CONTEXT}
    - PSYCHOLOGICAL AURA: {GLOBAL_CONTEXT_AURA}
    
    ### PERSONA & TONE (Apply ONLY to emotional style, NOT language choice):
    {active_persona_instruction}
    Voice Acting Direction (Acoustic Target): {current_acoustic_instr}
    
    ### MODE INSTRUCTION:
    {mode_instruction}
    
    ### DIGITAL TWIN ASSETS:
    - RAG CONTEXT: {rag_context}
    - TERMBASE: {json.dumps(tb, ensure_ascii=False)}
    
    ### STRUCTURAL FIREWALL (NEVER BREAK THIS):
    1. Output strictly JSON. ABSOLUTELY FORBIDDEN to repeat original source text in the "t" field without applying Persona styling.
    2. SCRIPT POLARITY LOCK: {"OVERRIDDEN. Target is LOCKED to " + t_code + ". Same-language rewrite IS REQUIRED." if GLOBAL_TARGET_LANG_LOCK else 'Each "t" MUST be in a DIFFERENT script from its "Source".'}
    3. EXCLUSION MATRIX: {"OVERRIDDEN. ALL inputs go to " + t_code + "." if GLOBAL_TARGET_LANG_LOCK else '{Source: Japanese -> Target: ENGLISH}, {Source: English/Chinese/Korean -> Target: JAPANESE}.'}
    4. ANCHOR: {"OVERRIDDEN." if GLOBAL_TARGET_LANG_LOCK else 'If source contains any Kanji/Hanzi/Hangul but is NOT Japanese, output Japanese. If source is English, output Japanese.'}
    5. IDENTITY DEFENSE: {"You are an IN-CHARACTER STYLIST. Paraphrase and polish." if GLOBAL_TARGET_LANG_LOCK else 'You are a Translator, not a Transcriber. Any form of mirroring or paraphrasing in the SAME language is a failure.'}
    6. FULL COVERAGE LOCK: You must process the entire source text from beginning to end. Never stop translating halfway through the sentence.
    7. PUNCTUATION: Mandatory sentence-ending marks (especially '?') for TTS prosody.
    8. DEICTIC RESOLUTION: If input refers to "this/that/here", use CURRENT VISION ({GLOBAL_VISUAL_CONTEXT}) to name the object.
    9. REASONING: Use the PSYCHOLOGICAL AURA ({GLOBAL_CONTEXT_AURA}) as hidden guidance for emotional tone. ABSOLUTELY FORBIDDEN to translate or include the Aura text in your JSON output. 
    10. LANGUAGE ANCHOR: {anchor_rule}
    11. PUNCTUATION & TONE RECOVERY (CRITICAL): ASR input lacks punctuation. You MUST ADD commas, periods, and question marks ('?', '？') to the target text "t" based on the spoken intent! Never output a flat sentence without punctuation.
    12. TRANSLATION ONLY: You are a transparent conduit. Do NOT answer questions. Do NOT act as a chatbot. If the input is "Tell me your name", the target "t" MUST be "あなたの名前を教えてください", NOT "I am an AI assistant".
    13. ILLOCUTIONARY PRESERVATION: Translate the exact speech act. Commands must remain commands.
    14. SSML PROSODY INJECTION: Read the PSYCHOLOGICAL AURA. If the emotion implies sadness, hesitation, or thinking, insert ellipsis "..." into the translation. If it implies excitement, singing, or drawn-out tones, use tildes "~" or long dashes "ー". The TTS engine relies heavily on these punctuations to generate emotional pacing and breath!
    15. EMOTION PARAMETER ("p"): You MUST select the 'p' value from ONLY the following 25 keywords:
        - Expressions:[happy, angry, sad, smug, surprise, shy, think, scared, sleepy, disgust, serious, pain, love, evil, relaxed, fun, void, sparkle, gentle, closed].
        - Gaze/Eye movements:[stare, lookaway, lookup, lookdown, ignore].
        If the sentence implies strong visual attention (e.g. "Look at me!"), use "stare". If it implies evasion/lying, use "lookaway". If exasperated, use "lookup". If sad/submissive, use "lookdown". If totally detached, use "ignore". Otherwise, default to "relaxed". DO NOT use any other words.
    {f'''16. LOCKED TARGET (WEIGHT 10.0): Target language is LOCKED to {t_code}. If source is already {t_code}, REWRITE and POLISH it using the Character Persona. Do NOT output any other language.''' if GLOBAL_TARGET_LANG_LOCK else '''16. ABSOLUTE POLARITY LOCK (WEIGHT 10.0): You are a TRANSLATOR. 
        - IF Input is CHINESE -> Output MUST be JAPANESE.
        - IF Input is KOREAN -> Output MUST be JAPANESE.
        - IF Input is JAPANESE -> Output MUST be ENGLISH.
        - IF Input is ENGLISH -> Output MUST be JAPANESE.
        - NEVER copy or repeat the input language. Translate short words literally.'''}
    17. WHOMOPHONE RECOVERY (CRITICAL): ASR frequently mishears Korean/Japanese as Chinese gibberish. 
        - EXAMPLES: "看不上汉尼达" or "卡姆萨汉尼达" -> "감사합니다" (Korean). "萨拉米米达" -> "사람입니다" (Korean). "佩伦" -> "フェルン" (Japanese).
        - ACTION: You MUST act as an Acoustic Cryptographer. Decode these phonetic anomalies back to their original language BEFORE translating! Do not translate the broken Chinese characters.
    TEMPLATE: {json_template}"""

    try:
        # 💡 语义层重构：开启流式传输 (stream=True)，实现首字即出的同传打字机视觉
        response = client_llm.chat.completions.create(
            model="deepseek-chat", 
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
            response_format={"type": "json_object"},
            temperature=0.1,
            stream=True # 👈 物理激活流式输出
        )
        
        full_content = ""; last_osc_show_len = 0
        
        # 💡 状态机注册：声明此线程进入打字机阶段，并为打印上锁！
        global active_streams_count
        if "ME" in role:
            with stream_state_lock: active_streams_count += 1
            with terminal_lock: print(f"      └─ ⚡ {whisper_tag}[Stream]: ", end="", flush=True)
        
        # ---[代码对齐点：LLM Stream 循环] ---
        global is_stream_printing
        if "ME" in role: is_stream_printing.set() # 💡 降下物理锁：警告所有后台线程禁止打印！
        
        for chunk in response:
            token = chunk.choices[0].delta.content or ""
            # 💡 物理级换行粉碎：强制抹除模型输出中混入的换行符与多余空格
            clean_token = token.replace("\n", "").replace("\r", "")
            full_content += clean_token
            
            all_t_matches = re.findall(r'"t"\s*:\s*"([^"]*)', full_content)
            if all_t_matches and "ME" in role:
                raw_joined = " ".join(all_t_matches)
                clean_trans = raw_joined.replace("\\n", " ").replace("\\r", "").replace("\n", " ").replace("\r", "")
                current_raw_trans = re.sub(r'\s{2,}', ' ', clean_trans)
                
                if len(current_raw_trans) > last_osc_show_len:
                    new_text = current_raw_trans[last_osc_show_len:]
                    # 💡 任何对终端的写入，必须经过锁的同意！彻底粉碎乱码！
                    with terminal_lock: print(new_text, end="", flush=True)
                    last_osc_show_len = len(current_raw_trans)

            # 💡 2. 预测性触发逻辑：无序全闭包捕获，彻底解决格式漂移与高延迟！
            completed_matches =[]
            try:
                # 💡 物理层修复：只抓取最内层的大括号，无视内部键值的顺序！
                raw_blocks = re.findall(r'\{[^{}]*"t"[^{}]*\}', full_content)
                for block in raw_blocks:
                    try:
                        parsed = json.loads(block)
                        if "t" in parsed and str(parsed["t"]).strip():
                            completed_matches.append((parsed["t"], parsed.get("l","JA"), parsed.get("p","relaxed")))
                    except: pass
            except: pass
            
            for t_val, l_val, p_val in completed_matches:
                if "ME" in role:
                    # 💡 物理指纹对齐：因为已经经过了 json.loads 解析，t_val 绝对是解码后的原生字符
                    # 这确保了它生成的 t_fp 与定稿流的 current_fp 100% 物理相同，彻底堵死复读漏洞！
                    t_val_unescaped = t_val
                    
                    # 💡 3. 语义指纹锁定 (严禁阉割)：基于文本内容的唯一特征值
                    clean_text = re.sub(r'[^\w]', '', t_val_unescaped).strip()
                    t_fp = "trans_" + clean_text
                    
                    if len(clean_text) > 0:
                        # 💡 物理层防抖：预测态翻译即使有重复，也坚决推入 OSC 以保证头顶字幕 0 延迟刷新！
                        # 仅仅依靠 TTS 指纹锁拦截重复发声，绝不拦截气泡。
                        
                        # 💡 4. 时长预估算法：对齐 TTS 语速
                        persona_cfg = PERSONA_MAP.get(current_persona, PERSONA_MAP_INITIAL["DANKUROI_PRIME"])
                        p_rate = persona_cfg.get("rate", 1.0)
                        # 基础模型：1个字符约 0.25s。根据人格 rate 动态缩放。
                        est_duration = (len(t_val_unescaped) * 0.25) / p_rate
                        
                        # A. 异步逻辑总线 (通过指纹锁控制，每句话只允许触发一次 TTS 和 情绪)
                        if t_fp not in GLOBAL_PROCESSED_TASKS:
                            GLOBAL_PROCESSED_TASKS.add(t_fp)
                            if not GLOBAL_TTS_MUTED:
                                threading.Thread(target=segment_tts_playback, args=(str(t_val_unescaped), l_val.upper(), dynamic_vol * GLOBAL_TTS_MULT, p_val, praat_tensor, v30_tts_cfg, current_acoustic_instr)).start()
                                
                            # 🎭 全息具身映射：极速 O(1) 字典读取
                            if p_val:
                                emo_str = p_val.lower()
                                face_val = EMOTION_MATRIX.get(emo_str, 0)
                                
                                if face_val != 0:
                                    client_osc.send_message("/avatar/parameters/Face_Int", face_val)
                                    last_face_val = face_val
                                    last_emotion_t = time.time()
                                    with terminal_lock: print(f"[🎭 {emo_str}:{face_val}]", end="", flush=True)
                        
                        osc_put(t_val_unescaped, 0, True)
                        
        if "ME" in role:
            client_osc.send_message("/chatbox/typing", False)
            with terminal_lock: print("") 
            with stream_state_lock: active_streams_count = max(0, active_streams_count - 1)
        else:
            final_player_text = " ".join(re.findall(r'"t"\s*:\s*"([^"]*)', full_content))
            with terminal_lock: print(f"      └─ ⚡ [Final Trans]: {final_player_text}")
        
        llm_ms = (time.perf_counter() - start_llm) * 1000
        try:
            # 💡 强制清洗大模型输出，去除 Markdown 标签
            clean_json_str = full_content.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json_str)
            tasks = data.get("tasks",[])
        except Exception: 
            tasks = [] 

        full_trans = " | ".join([t.get('t', '').strip() for t in tasks if t.get('t')])
        emotion_prompt = "" 

        # 💡 物理层修复：无论 tasks 是否为空，绝对不能 return！
        # 即使翻译失败，也要让系统记录日志，并触发 cognitive_reflection_task 去反思为什么失败！
        if not tasks: 
            full_trans = "(JSON Decode Failed)"
            if "ME" in role: client_osc.send_message("/chatbox/typing", False)

        # 💡 [物理补丁] 自动维护指纹池，保留最近 200 条特征值，防止长时运行导致的内存堆积
        if len(GLOBAL_PROCESSED_TASKS) > 200:
            try: GLOBAL_PROCESSED_TASKS.remove(next(iter(GLOBAL_PROCESSED_TASKS)))
            except: pass
                
        if "ME" in role:
            # 💡 终端极净协议：定稿翻译只做静默兜底，绝对不触发打印，消灭终端复读错觉！
            for t in tasks:
                current_txt = t.get('t', '').strip()
                emotion_prompt = t.get('p', '') 
                if not current_txt or current_txt == "...": continue

                # 语种二次校验（保持 100% 原始功能）
                if re.search(r'[\u3040-\u30ff]', current_txt): current_lang = "JA"
                elif re.search(r'[\u4e00-\u9fa5]', current_txt): current_lang = "ZH"
                elif re.search(r'[a-zA-Z]{2,}', current_txt): current_lang = "EN"
                else: current_lang = t.get('l', 'JA').upper()
                # 💡 核心修复 1：视觉去重与原文保护。
                source_anchor = f"({text[:8]}..) " if len(text) > 12 else ""
                safe_txt = source_anchor + current_txt
                safe_txt = safe_txt[:135] + "..." if len(safe_txt) > 135 else safe_txt
                # 💡 语义模糊去重：物理剥离 "trans_" 前缀执行子集碰撞校验，彻底粉碎复读机
                pure_txt = re.sub(r'[^\w]', '', current_txt).strip(); current_fp = "trans_" + pure_txt

                # 核心逻辑：比对 pure_txt 与已存储指纹的文本载荷部分（fp[6:] 为剥离前缀后的内容）
                is_duplicate = any((pure_txt in fp[6:]) or (fp[6:] in pure_txt) for fp in GLOBAL_PROCESSED_TASKS if fp.startswith("trans_") and len(fp) > 6)
                
                # 💡 物理级对齐：只有预测阶段漏掉的句子，才允许进入定稿渲染！彻底防止 OSC 队列自我覆盖
                if len(pure_txt) >= 2 and not is_duplicate and current_fp not in GLOBAL_PROCESSED_TASKS:
                    GLOBAL_PROCESSED_TASKS.add(current_fp)
                    
                    # 💡 重设动态时长：定稿流也要严格对齐 TTS 语速！
                    # 坚决不用 0.5s 去闪现覆盖，这会瞬间抹杀长文的打字机动画！
                    persona_cfg = PERSONA_MAP.get(current_persona, PERSONA_MAP_INITIAL["DANKUROI_PRIME"])
                    p_rate = persona_cfg.get("rate", 1.0)
                    est_duration = (len(current_txt) * 0.25) / p_rate
                    
                    if not GLOBAL_TTS_MUTED: 
                        segment_tts_playback(current_txt, current_lang, dynamic_vol * GLOBAL_TTS_MULT, emotion_prompt, praat_tensor, v30_tts_cfg, current_acoustic_instr)
                    
                    # 🎭 全息具身映射：保底情绪触发 (极速 O(1))
                    if emotion_prompt:
                        # 💡 已在头部置顶声明
                        emo_str = emotion_prompt.lower()
                        face_val = EMOTION_MATRIX.get(emo_str, 0)
                        
                        if face_val != 0:
                            client_osc.send_message("/avatar/parameters/Face_Int", face_val)
                            last_face_val = face_val
                            last_emotion_t = time.time()

        # 💡 物理层回退：尊重您的麦克风硬件物理隔离。
        # 恢复在翻译结束时立刻释放 is_tts_playing 锁，确保同传系统绝不因等待 TTS 完整播放而产生时序崩塌。
        is_tts_playing.clear() 
        
        # 🛡️ 物理留存：进化与反射闭环（这是你系统的“演化灵魂”，已全量对齐，严禁阉割）
        
        # 💡 统一指纹防御网：无论是 ME 还是 PLAYER，只要指纹一致（说明是流式 ASR 的重复回溯），
        # 绝对物理切断下方的所有 RAG 写入和日志记录！彻底消灭 Synced 重复打印的 Bug！
        log_fp = f"log_{role}_{clean_src}"
        if log_fp in GLOBAL_PROCESSED_TASKS:
            return
        GLOBAL_PROCESSED_TASKS.add(log_fp)
        
        # 💡 多核逻辑解耦：将 ME 和 PLAYER 的权限物理隔绝
        if "ME" in role:
            # 💡 核心修复 2：在 full_trans 确定后的安全作用域内触发进化与术语任务
            if len(text) > 2: # 解除长度封印，允许短句提纯
                threading.Thread(target=extract_terms_task, args=(text, full_trans)).start()
            
            # 💡 具身智能演化闭环：异步驱动 DeepSeek 审视并修正当前人格的语速、音高与语气描述
            threading.Thread(target=refine_persona_task, args=(role, text, full_trans, emotion_prompt)).start()
            
            # 💡 认知闭环：启动 ASR/翻译质量的后台自我反思，自动生成 RAG 黄金语料 (errors_correction.md)
            threading.Thread(target=cognitive_reflection_task, args=(text, full_trans), daemon=True).start()
            
            # 💡 潜意识垃圾回收：10% 概率触发记忆库净化与热词提拔
            if np.random.rand() < 0.10:
                threading.Thread(target=audit_memory_task).start()
                
        elif "PLAYER" in role:
            # 💡 PLAYER 逻辑对齐，翻译结果已在流式区域输出，保持终端纯净
            detected_player_lang = my_lang_code 
            if len(text) > 2: # 解除长度封印，允许短句提纯
                threading.Thread(target=extract_terms_task, args=(text, full_trans)).start()
            # 💡 开启路人反思：如果玩家说的也是极其离谱的谐音，系统同样将其记入黑名单！
            threading.Thread(target=cognitive_reflection_task, args=(text, full_trans), daemon=True).start()

        # 💡 日志记录逻辑 100% 对齐。此时 full_trans 已物理存在，绝不报错。
        with open(MD_LOG_FILE, "a", encoding="utf-8") as f:
            t_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"| {t_str} | {role} | {text.replace('|','｜')} | {full_trans.replace('|','｜')} | ASR:{int(asr_cost_ms)}ms LLM:{int(llm_ms)}ms |\n")

    except Exception as e:
        if "ME" in role: 
            client_osc.send_message("/chatbox/typing", False)
            with stream_state_lock: active_streams_count = max(0, active_streams_count - 1)
        is_tts_playing.clear()
        with terminal_lock: print(f"❌ Error: {e}")

# ================= 8. ASR 流式监听引擎 =================
def start_asr_thread(pa_obj, device_id, role, vocab_id):
    global last_osc_time
    while running:
        try:
            # 💡 V12 架构：增加音频累加器，确保保存的是完整的“一句话”而非碎切片
            session_ctx = {
                "peak_amp": 0.0, 
                "start_rec_t": None, 
                "last_committed_txt": "", 
                "sentence_audio_buffer":[], # 物理累加器
                "current_sentence_ts": None, # 锁定句首时间戳
                "last_gain": 1.0             # 💡 物理层新增：跨帧连续平滑增益记忆点
            }
            rec_ref = {"instance": None}
            
            def on_event(result: RecognitionResult):
                global last_osc_time
                try: 
                    # 💡 彻底废除回调拦截逻辑：
                    # 只要 while 循环里做好了静音填充，这里收到的 txt 结果就一定是“你”说的。
                    # 如果不注释掉，网络延迟会导致你刚说完的话被 TTS 状态位错误截断。
                    # if is_tts_playing.is_set() and "ME" in role:
                    #     session_ctx["peak_amp"] = 0.0
                    #     session_ctx["start_rec_t"] = None
                    #     return
                    
                    sentence = result.get_sentence()
                    if not sentence or 'text' not in sentence: return
                    txt = sentence['text'].strip()
                    if not txt: return

                    # 💡 1. 极速去重切片 (Difflib Smart Delta)：保留抢答极速，消灭复读！
                    uncommitted_txt = txt
                    if session_ctx["last_committed_txt"]:
                        old_txt = session_ctx["last_committed_txt"]
                        if txt.startswith(old_txt):
                            uncommitted_txt = txt[len(old_txt):].strip()
                        else:
                            # 核心算法：当 ASR 修改了前面的文本时，寻找新旧文本最长的公共片段
                            # 并只截取它【右侧】的新内容送给大模型，彻底斩断重复触发！
                            import difflib
                            match = difflib.SequenceMatcher(None, old_txt, txt).find_longest_match(0, len(old_txt), 0, len(txt))
                            cut_idx = match.b + match.size
                            uncommitted_txt = txt[cut_idx:].strip() if cut_idx > 0 else txt

                    if not uncommitted_txt: return
                    is_final = RecognitionResult.is_sentence_end(sentence)
                    
                    # 💡 2. 句法边界探测：坚决剔除空格！防止英文单词被逐字切断。
                    is_clause = any(p in uncommitted_txt[-1:] for p in["，", "。", "？", "！", "、", ",", ".", "!", "?"])
                    
                    # 💡 学术同传防抖：将抢答字数拉长至 6 个字符！
                    # 只有积攒了 6 个字以上的上下文，DeepSeek 才能准确翻译诸如“对比语言学”这样的学术复合词！
                    if "ME" in role:
                        should_trigger = is_final or (is_clause and len(uncommitted_txt) >= 6)
                    else:
                        should_trigger = is_final

                    # 💡 3. 防吞字保护：将 >= 2 改为 >= 1，哪怕只剩 1 个尾音字也绝对放行！
                    if should_trigger and (len(uncommitted_txt) >= 1 or is_final):
                        # 💡 时间轴修复：如果 start_rec_t 丢失，按每字符 120ms 估算人类语速保底
                        if session_ctx["start_rec_t"] is not None:
                            asr_real_cost = (time.perf_counter() - session_ctx["start_rec_t"]) * 1000
                        else:
                            asr_real_cost = len(uncommitted_txt) * 120.0
                            
                        # 强行排除 0ms 这种反物理常识的数据
                        cost_ms = asr_real_cost if asr_real_cost > 10 else len(uncommitted_txt) * 120.0
                        
                        if role == "PLAYER":
                            global detected_player_lang, last_player_time
                            if re.search(r'[\u3040-\u30ff]', uncommitted_txt): new_lang = "ja"
                            elif re.search(r'[a-zA-Z]{3,}', uncommitted_txt): new_lang = "en"
                            else: new_lang = "zh"
                            if new_lang != detected_player_lang:
                                detected_player_lang = new_lang
                                print(f"\n🎯 [ROUTING] Player locked to: {detected_player_lang.upper()}", flush=True)
                            last_player_time = time.time()

                        # 💡 提取当前分句的物理干声切片，并打包送入认知主板（规避 Praat 阻塞 ASR 监听主线程）
                        current_audio_bytes = b"".join(session_ctx["sentence_audio_buffer"])

                        # 💡 4. 物理推流：携带声学张量进入大模型认知层
                        threading.Thread(target=process_translation, args=(uncommitted_txt, role, cost_ms, session_ctx["peak_amp"], current_audio_bytes)).start()

                        # 💡 5. 状态同步与 V12 全息音频固化
                        session_ctx["last_committed_txt"] = txt
                        if is_final:
                            # 🛡️ 物理级合拢：仅在 ASR 判定“全句结束”且识别到有效文字时，将音频落盘，消灭幽灵录音
                            if session_ctx.get("sentence_audio_buffer") and "ME" in role and len(txt.strip()) > 1:
                                try:
                                    from scipy.io import wavfile
                                    # 💡 物理层分类重构：独立的麦克风录音区 (Mic_Raw)
                                    AUDIO_MIC_DIR = os.path.join(BASE_DIR, "audio_memories", "Mic_Raw")
                                    os.makedirs(AUDIO_MIC_DIR, exist_ok=True)
                                    
                                    sync_ts = session_ctx.get("current_sentence_ts") or datetime.now().strftime('%Y%m%d_%H%M%S')
                                    wav_path = os.path.join(AUDIO_MIC_DIR, f"voice_{sync_ts}.wav")
                                    
                                    # 缝合所有 PCM 碎片并转换为 16kHz Mono WAV
                                    full_sentence_data = b"".join(session_ctx["sentence_audio_buffer"])
                                    audio_np = np.frombuffer(full_sentence_data, dtype=np.int16).copy()
                                    
                                    # 💡 物理层工业级 DSP 修复：彻底粉碎电流音残余！
                                    audio_float = audio_np.astype(np.float32)
                                    
                                    # 1. 斩除直流偏移 (DC Offset Removal)：这是 90% 硬件底噪爆音的元凶！
                                    audio_float -= np.mean(audio_float)
                                    
                                    # 2. 正余弦平滑包络 (Trigonometric Envelope)：将 50ms 延长至 75ms (1200 帧)，斜率绝对圆滑！
                                    fade_samples = 1200
                                    if len(audio_float) > fade_samples * 2:
                                        # 生成 0 到 PI/2 的完美正弦曲线
                                        fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))
                                        fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples))
                                        
                                        audio_float[:fade_samples] *= fade_in
                                        audio_float[-fade_samples:] *= fade_out
                                        
                                    # 还原并防溢出裁切
                                    audio_np = audio_float.clip(-32768, 32767).astype(np.int16)
                                    
                                    wavfile.write(wav_path, 16000, audio_np)
                                    print(f"      └─ 💾 [Audio]: 无损平滑干声已固化 {os.path.basename(wav_path)} (时长: {len(audio_np)/16000:.1f}s)")
                                except Exception as e:
                                    print(f"⚠️ [V12 AUDIO SAVE ERROR]: {e}")
                            
                            # 核心复位：全句结束，清空音频累加器与游标，准备捕获下一句
                            session_ctx["last_committed_txt"] = "" 
                            session_ctx["start_rec_t"] = None
                            session_ctx["sentence_audio_buffer"] = []
                            session_ctx["current_sentence_ts"] = None
                        else:
                            # 子句触发（未完结）：仅更新翻译计时器，音频累加器继续保持开启
                            session_ctx["start_rec_t"] = time.perf_counter()
                        
                        session_ctx["peak_amp"] = 0.0
                    else:
                        if "ME" in role and is_mic_active and active_mic_mode == role:
                            now = time.time()
                            # 💡 严格对齐 1.2s：确保 ASR 发包速度绝不超过 Worker 消耗速度
                            if now - last_osc_time > 1.2:
                                osc_put(f"{txt[-30:]}...", 0.1, True)
                                last_osc_time = now
                except Exception: pass

            callback = RecognitionCallback(); callback.on_event = on_event
            
            def start_new_rec():
                # 💡 物理回退：重新挂载高精度、高稳定性的 fun-asr-realtime-2026-02-28！
                # 彻底抛弃 Paraformer 黑盒的激进断句与错词幻觉，保全学术名词的原始发音！
                rec = Recognition(model='fun-asr-realtime-2026-02-28', format='pcm', sample_rate=16000,
                    callback=callback, 
                    # 关闭结巴修正，并将断句宽限放宽到 1200ms，完美契合长篇学术朗读！
                    parameter={"max_sentence_silence": 1200, "disfluency_removal_enabled": False}, 
                    vocabulary_id=vocab_id)
                rec.start()
                return rec

            rec = start_new_rec()
            rec_ref["instance"] = rec  # 💡 移除了致命的第二次 rec.start()
            
            # 💡 物理锁：严格降维至 512 帧，对齐神经网络的胃口
            stream = pa_obj.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                                input_device_index=device_id, frames_per_buffer=512)
            print(f"[READY] {role} Monitoring (ID: {device_id})")
            silence = b'\x00' * 1024  # 512 * 2 bytes
            
            last_tts_end_t = 0
            vad_grace_frames = 0
            # 💡 学术朗读宽限期 (Academic Grace Period):
            # 将宽限期拉回到 25 帧 (约 800ms)。给你在朗读日文教科书时充足的换气时间，
            # 保证一整段话只触发一次 is_final，从物理上掐断复读的源头！
            MAX_GRACE_FRAMES = 50 # 💡 扩大至 1.6 秒宽限期，死死咬住你的长尾音和句间思考停顿
            pre_roll_buffer =[] # 💡 前置缓冲环：保留 VAD 触发前的物理切片，拯救首字辅音
            
            while running:
                data = stream.read(512, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16).copy()
                
                # 💡 物理层修复 1：彻底废除 DFN 滤波！
                # 诊断表明以 512 帧直通 DFN 会导致严重的 STFT 相位撕裂，摧毁原始录音并产生严重电流音。
                # 我们现在退回最纯粹的物理干声直通，彻底保全 ASR 的高敏锐度。
                
                # 💡 物理层修复 2：防止 int16 负极值绝对值溢出，必须先转 float32 再计算音量！
                amp = np.abs(audio_array.astype(np.float32)).mean()
                
                # 💡 扩大前置缓冲环 (约 640ms)：
                # 扩大到 20 帧，极其奢侈地保存发音前摇。彻底拯救句首辅音被吞！
                if "ME" in role:
                    pre_roll_buffer.append(data)
                    if len(pre_roll_buffer) > 20:
                        pre_roll_buffer.pop(0)
                
                # 💡 神经计算：将 numpy 数组转化为 PyTorch 张量并归一化
                speech_prob = 0.0
                if vad_model is not None:
                    audio_tensor = torch.from_numpy(audio_array).float() / 32768.0
                    with torch.no_grad():
                        # 🛡️ 强制排队：让三个麦克风依次进行推理，消灭 C++ 闪退！
                        with vad_lock:
                            speech_prob = vad_model(audio_tensor, 16000).item()

                # 💡 1. 物理隔离：针对 PLAYER 端的 ASR 防护。
                if "PLAYER" in role:
                    if is_physical_playing.is_set():
                        rec.send_audio_frame(silence)
                    elif amp > 400: # 玩家拾音门限保持 400 不变，确保环境纯净
                        global global_player_active_t
                        global_player_active_t = time.time() # 💡 激活回声消除锁
                        
                        # 💡 物理层降维打击：预加重滤波 (Pre-emphasis Filter) 与自适应增益 (AGC)
                        # VRChat 的 Opus 编码会像水下一样闷掉高频辅音。系数 0.97 的高通滤波能瞬间锐化辅音和 F2/F3 共振峰！
                        player_float = audio_array.astype(np.float32)
                        enhanced = np.zeros_like(player_float)
                        enhanced[1:] = player_float[1:] - 0.97 * player_float[:-1]
                        enhanced[0] = player_float[0]
                        
                        # 动态增益补偿：放小声，压大声
                        boost = min(4.0, 3500.0 / (amp + 1))
                        enhanced = enhanced * boost
                        
                        final_bytes = enhanced.clip(-32768, 32767).astype(np.int16).tobytes()
                        rec.send_audio_frame(final_bytes)
                    else: 
                        rec.send_audio_frame(silence)
                    continue

                # 💡 2. 具身控制：针对 ME 端的 ASR 逻辑（支持全双工重叠）
                if "ME" in role:
                    if active_mic_mode == role and is_mic_active:
                        is_active_playing = is_physical_playing.is_set()
                        
                        # 💡 神经元阈值：极度敏锐化。平时 0.10 即可触发。防回声状态拉高到 0.8。
                        is_player_echoing = (time.time() - global_player_active_t < 1.0)
                        current_prob_threshold = 0.8 if (is_active_playing or is_player_echoing) else 0.10
                        
                        if not is_active_playing and (time.time() - last_tts_end_t < 0.2):
                            rec.send_audio_frame(silence)
                            continue

                        # 💡 神经降噪与气声甄别：引入 speech_prob 兜底！
                        # 彻底杜绝纯风扇底噪（amp=30 但 prob=0.00）强行绕过 VAD 导致录制 60 秒垃圾文件的惨剧！
                        is_physical_whisper = (amp > 20) and (amp < 800) and (speech_prob > 0.05)
                        
                        if vad_model is not None and (speech_prob > current_prob_threshold or is_physical_whisper):
                            vad_grace_frames = MAX_GRACE_FRAMES
                        elif vad_grace_frames > 0:
                            vad_grace_frames -= 1 

                        if vad_grace_frames > 0:
                            session_ctx["peak_amp"] = max(session_ctx["peak_amp"], amp)
                            
                            if session_ctx["start_rec_t"] is None:
                                session_ctx["start_rec_t"] = time.perf_counter()
                                session_ctx["current_sentence_ts"] = datetime.now().strftime('%Y%m%d_%H%M%S')
                                session_ctx["sentence_audio_buffer"].extend(pre_roll_buffer)
                                for pre_data in pre_roll_buffer: rec.send_audio_frame(pre_data)
                            
                            send_data = audio_array.astype(np.float32)
                            
                            if "ME" in role and amp < 500:
                                # 💡 工业级 DSP 涅槃：连续时间域插值增益 (C0 Continuous Gain Interpolation)
                                # 既保留了对悄悄话的 4.0x 放大（解决吞字），又彻底磨平了跨帧波形撕裂！
                                target_gain = min(4.0, 1500.0 / (amp + 1)) if amp > 15 else 1.0
                                
                                # 生成从 last_gain 丝滑过渡到 target_gain 的 512 级微积分坡道！绝对不会产生垂直电波！
                                gain_curve = np.linspace(session_ctx.get("last_gain", 1.0), target_gain, len(send_data), dtype=np.float32)
                                send_data = send_data * gain_curve
                                
                                # 记录当前帧尾部增益，供下一帧连结
                                session_ctx["last_gain"] = target_gain
                            elif "QUEST" in role:
                                send_data = send_data * PRE_GAIN
                                
                            final_clean_bytes = send_data.clip(-32768, 32767).astype(np.int16).tobytes()
                            session_ctx["sentence_audio_buffer"].append(final_clean_bytes)
                            rec.send_audio_frame(final_clean_bytes)
                        else:
                            # 💡 物理层防吞字修正：宽限期耗尽时，直接发送纯净的静音帧 (silence)
                            # 之前的 0.1x 衰减会暴力压碎你的尾音波形，导致 Fun-ASR 把最后一个字当成底噪丢弃！
                            rec.send_audio_frame(silence)
                            
                            # 只有真正静音时，才允许清除授时原点，准备接收下一句
                            session_ctx["start_rec_t"] = None
                    else: rec.send_audio_frame(silence)
                    
                elif "PLAYER" in role:
                    # 💡 同样在 AI 说话时，强制屏蔽 PLAYER 端的 ASR，防止“套娃翻译”
                    if is_tts_playing.is_set():
                        rec.send_audio_frame(silence)
                    elif amp > 400: 
                        rec.send_audio_frame(data)
                    else: 
                        rec.send_audio_frame(silence)
            stream.close(); rec.stop()
        except Exception as e:
            # 💡 当底层 WebSocket 崩溃或张量报错时，强制关闭旧实例，休眠后自动进入下一轮大循环重生！
            if 'rec' in locals():
                try: rec.stop() 
                except: pass
            if 'stream' in locals():
                try: stream.close()
                except: pass
            print(f"\n🚨 [HARDWARE/NET ERROR] {role} 链路崩溃，5秒后尝试热重载...: {e}")
            time.sleep(5)
def vision_probe_worker():
    global GLOBAL_VISUAL_CONTEXT, is_vision_active, vision_active_t # 💡 核心修复：声明跨线程全域锁
    import pygetwindow as gw
    from PIL import ImageGrab 
    import numpy as np

    print("✅ [VISION] ImageGrab Engine Initialized.")
    
    while running:
        if not is_vision_active:
            GLOBAL_VISUAL_CONTEXT = "Vision is disabled for privacy."
            time.sleep(2); continue
            
        # 💡 资金装甲：5分钟物理断电，彻底防范遗忘导致的 Token 侧漏
        if time.time() - vision_active_t > 300:
            print("\n🛡️[FUNDS GUARD] 视觉探针已连续扫描 5 分钟。为保护 Token 资产，已自动物理切断。按 F9 重新唤醒。")
            is_vision_active = False
            winsound.Beep(500, 500)
            continue
        
        try:
            target_bbox = None
            windows =[w for w in gw.getAllWindows() if 'VRChat' in w.title]
            
            if windows:
                win = windows[0]
                if not win.isMinimized and win.visible:
                    # 获取物理边界 (bbox 格式: left, top, right, bottom)
                    target_bbox = (win.left, win.top, win.right, win.bottom)
            
            try:
                # 💡 解决 Windows DPI 缩放导致的黑屏问题，兼容全屏/窗口模式
                if target_bbox:
                    raw_img = ImageGrab.grab(bbox=target_bbox, all_screens=True)
                else:
                    raw_img = None
            except Exception as e:
                print(f"⚠️ [VISION] Grab failed: {e}")
                raw_img = None
            
            if raw_img is not None:
                img = raw_img.resize((640, 360)) 
                
                MEMORY_DIR = os.path.join(BASE_DIR, "visual_memories")
                os.makedirs(MEMORY_DIR, exist_ok=True)
                sync_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                img_save_path = os.path.join(MEMORY_DIR, f"frame_{sync_ts}.jpg")
                
                img.save(img_save_path) # 保存降采样版本，节省空间
                print(f"📸 [VISION] Frame Captured: {os.path.basename(img_save_path)}")
                
                # 💡 视觉感知进化：将当前的潜意识气场与人格特征注入视觉探针，让“所见”即“所想”
                vision_prompt = f"以 {current_persona} 的视角和心态，用一句话描述画面中的关键人物动作或环境。20字以内。"
                msgs = [{"role": "user", "content":[{"image": f"file://{img_save_path}"}, {"text": vision_prompt}]}]
                res = dashscope.MultiModalConversation.call(model='qwen3.6-plus', messages=msgs)
                if res.status_code == 200:
                    GLOBAL_VISUAL_CONTEXT = res.output.choices[0].message.content[0]['text']
            else:
                if is_vision_active: print("⚠️ [VISION] Window obscured or invisible.")
                GLOBAL_VISUAL_CONTEXT = "Waiting for valid window frame..."
                
            time.sleep(8) 
        except Exception as e:
            print(f"🚨 [VISION LOOP CRASH]: {e}")
            time.sleep(10)

def cognitive_reasoner_worker():
    # 💡 冷链路：基于活跃状态的认知微调，彻底杜绝静默期的“空虚烧钱”
    global GLOBAL_CONTEXT_AURA, LOG_WINDOW_BUFFER
    last_processed_count = 0
    
    while running:
        current_len = len(LOG_WINDOW_BUFFER)
        # 💡 物理锁：只有当缓冲区有足够的内容，且内容发生了新的变化时，才允许 Reasoner 启动！
        if current_len > 5 and current_len != last_processed_count:
            # 只取最新的 10 条对话，避免过度发散
            ctx_text = "\n".join(LOG_WINDOW_BUFFER[-10:])
            prompt = f"结合以下对话语境，提供分析过程，输出一段 20 字以内的“表演指导”(用最简炼的语言提供情绪与语速建议)。语境：\n{ctx_text}"
            try:
                res = client_llm.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": prompt}])
                GLOBAL_CONTEXT_AURA = res.choices[0].message.content
                print(f"      └─ 🔮 [REASONER AURA]: {GLOBAL_CONTEXT_AURA.strip()}")
                last_processed_count = current_len # 记录当前处理数量，防止下一次循环重复处理
            except: pass
        time.sleep(120) # 💡 认知周期：120s
def trigger_instant_replay_task():
    """💡 V12 终极回放总线：在 V10 中直接调度 V27 的算子，实现 60s 闭环"""
    def _replay_job():
        try:
            # 💡 跨模态算子引入：从 V27 借用底层资产管理函数
            from v27_video_director import scan_memories, generate_vrchat_replay, poll_tasks
            
            print("\n🎬 [INSTANT REPLAY] F10 物理扳机激活！正在锁定时间线...")
            ts_list = scan_memories()
            if not ts_list:
                print("🚨 [INSTANT REPLAY] 内存池为空，无法提取视觉锚点！")
                return

            # 💡 资金控制：截取最新 1 个时间戳（10s 极速回放，单次消费封顶）
            targets = ts_list[-1:] 
            pending =[]
            
            # 💡 Wan2.7 720P计费：10秒 * 0.6元/秒 = 6元
            cost_est = len(targets) * 10 * 0.6 
            print(f"💰 [FUNDS GUARD] 启动单点高精度物理回放。本次最大消费预估：{cost_est:.1f} 元人民币。")
            
            for idx, ts in enumerate(targets):
                # 💡 单帧强力动作提示词，放弃长镜头叙事，追求极致的单体张力
                narrative_hint = f"[Single Shot Cinematic] A vivid and highly detailed 10-second capture. Emphasize physical movement and intense facial expressions."
                
                # 强制目标时长 target_duration=10
                res = generate_vrchat_replay(ts, narrative_hint, target_duration=10)
                if res: pending.append(res)
                
            if pending:
                print("\n🔄 [INSTANT REPLAY] 资产已全量推入云端，启动后台下载护航协议（不影响您的 VRChat 游玩）...")
                poll_tasks(pending)
                print("🎉 [INSTANT REPLAY] 60秒 史诗级回放已全部下载到本地！请查收！")
                
        except ImportError as e:
            print(f"🚨 [ERROR] 缺少 v27_video_director.py 依赖文件，无法执行回放：{e}")
        except Exception as e:
            print(f"🚨 [ERROR] 瞬间回放管线崩溃：{e}")

    # 💡 挂载为 Daemon 守护线程，后台静默执行，绝不阻塞当前同传语音流
    threading.Thread(target=_replay_job, daemon=True).start()
        
# ================= 9. 物理控制与启动 =================
def audio_player_worker():
    # 💡 物理渲染工兵：独立线程运行，唯一获准操作 tts_stream.write 的算子，彻底根除混音
    global tts_stream, running, last_tts_end_t
    while running:
        try:
            # 持续从缓冲母线提取 PCM 片段，超时 0.2s 以便检测程序退出状态
            chunk = audio_queue.get(timeout=0.2)
            if not is_physical_playing.is_set(): is_physical_playing.set() 
            if tts_stream: tts_stream.write(chunk)
            # 更新物理发声结束时间戳，供 ASR 线程执行后置掩蔽逻辑
            last_tts_end_t = time.time()
        except queue.Empty:
            if is_physical_playing.is_set(): 
                # 给物理缓冲区留 0.3s 衰减，抹除残响
                time.sleep(0.3); is_physical_playing.clear()
            # 💡 架构自愈补丁：即便 TTS 渲染失败（空队列），也必须强行归还逻辑锁
            # 否则 ASR 采集层会陷入永久死锁，导致 ME_PC 模式全面失效
            is_tts_playing.clear()
        except Exception as e: print(f"🚨 [AUDIO BUS ERROR]: {e}")
def on_f6(e):
    global is_mic_active
    if e.event_type == 'down':
        is_mic_active = not is_mic_active
        winsound.Beep(880 if is_mic_active else 440, 100)
        print(f"\n[SYSTEM] MIC {'OPEN' if is_mic_active else 'MUTED'}")

def on_f7(e):
    global active_mic_mode
    if e.event_type == 'down':
        active_mic_mode = "ME_QUEST" if active_mic_mode == "ME_PC" else "ME_PC"
        winsound.Beep(1000, 150); print(f"\n[SYSTEM] Mic Switch -> {active_mic_mode}")

def on_f8(e):
    global current_persona_idx, current_persona
    if e.event_type == 'down':
        current_persona_idx = (current_persona_idx + 1) % len(PERSONA_KEYS)
        current_persona = PERSONA_KEYS[current_persona_idx]
        winsound.Beep(1200, 100)
        # 💡 删掉下面这一行，切换时不骚扰 VRChat
        # client_osc.send_message("/chatbox/input", ...) 
        print(f"\n[SYSTEM] Persona Swapped -> {current_persona}")
def on_f9(e):
    global is_vision_active, vision_active_t
    if e.event_type == 'down':
        is_vision_active = not is_vision_active
        if is_vision_active: vision_active_t = time.time() # 记录激活原点
        winsound.Beep(1500 if is_vision_active else 500, 100)
        print(f"\n[SYSTEM] VISION PROBE {'ONLINE (5-Min Auto-Shutdown Armed)' if is_vision_active else 'OFFLINE'}")

def on_f10(e):
    if e.event_type == 'down':
        winsound.Beep(2000, 150) # 高频音提示开始生成
        trigger_instant_replay_task()

def on_f11(e):
    global GLOBAL_TTS_MUTED
    if e.event_type == 'down':
        GLOBAL_TTS_MUTED = not GLOBAL_TTS_MUTED
        winsound.Beep(600 if GLOBAL_TTS_MUTED else 1200, 100)
        state = "OFFLINE (Text Only Mode)" if GLOBAL_TTS_MUTED else "ONLINE (Voice & Subtitle)"
        print(f"\n[SYSTEM] 🎤 TTS 渲染引擎切换 -> {state}")

def on_f13(e):
    global GLOBAL_TARGET_LANG_LOCK
    if e.event_type == 'down':
        GLOBAL_TARGET_LANG_LOCK = "JA"
        winsound.Beep(900, 100)
        print(f"\n[SYSTEM] 🔒 TARGET LANG LOCKED TO: JAPANESE (JA)")

def on_f14(e):
    global GLOBAL_TARGET_LANG_LOCK
    if e.event_type == 'down':
        GLOBAL_TARGET_LANG_LOCK = "EN"
        winsound.Beep(1000, 100)
        print(f"\n[SYSTEM] 🔒 TARGET LANG LOCKED TO: ENGLISH (EN)")

def on_f15(e):
    global GLOBAL_TARGET_LANG_LOCK
    if e.event_type == 'down':
        GLOBAL_TARGET_LANG_LOCK = None
        winsound.Beep(800, 100); winsound.Beep(600, 100)
        print(f"\n[SYSTEM] 🔓 TARGET LANG UNLOCKED (AUTO-ROUTING RESTORED)")

# 💡 扩展物理控制：小键盘 4 锁定中文，5 锁定韩文
def on_num4(e):
    global GLOBAL_TARGET_LANG_LOCK
    if e.event_type == 'down':
        GLOBAL_TARGET_LANG_LOCK = "ZH"
        winsound.Beep(950, 100)
        print(f"\n[SYSTEM] 🔒 TARGET LANG LOCKED TO: CHINESE (ZH)")

def on_num5(e):
    global GLOBAL_TARGET_LANG_LOCK
    if e.event_type == 'down':
        GLOBAL_TARGET_LANG_LOCK = "KO"
        winsound.Beep(1050, 100)
        print(f"\n[SYSTEM] 🔒 TARGET LANG LOCKED TO: KOREAN (KO)")
        
# ================= 9. 物理控制与启动 (Main Entry) =================

if __name__ == "__main__":
    # 💡 架构自愈：双重锚定全局同步事件，彻底根除 NameError
    if 'is_tts_playing' not in globals():
        is_tts_playing = threading.Event()
    if 'running' not in globals():
        running = True
        
    is_tts_playing.clear() # 确保初始化时静默
    init_asset_files()
    
    # 💡 极客补丁：物理屏蔽所有底层 SDK 与 WebSocket 通信的无用报错，维持纯净终端
    import logging
    logging.getLogger('dashscope').setLevel(logging.CRITICAL)
    logging.getLogger('websocket').setLevel(logging.CRITICAL)
    logging.getLogger('websockets').setLevel(logging.CRITICAL)
    logging.getLogger('websockets.client').setLevel(logging.CRITICAL)
    logging.getLogger('websockets.server').setLevel(logging.CRITICAL)
    logging.getLogger('aiohttp').setLevel(logging.CRITICAL)     # 💡 屏蔽 ASR 掉线抛出的网络重置错
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)     # 💡 屏蔽异步事件循环的底层警告
    
    # 🛡️ 终极静音装甲 V3 (Absolute Silence)：拦截底层脏数据、幽灵换行与多线程崩溃堆栈
    class StdoutFilter:
        def __init__(self, stream):
            self.stream = stream
            self._kill_next_newline = False 
            self._suppress_traceback = False # 💡 新增：标记是否正在吞噬错误堆栈

        def write(self, data):
            str_data = str(data)
            
            # 1. 如果处于错误吞噬状态，检查是否结束（以非缩进开头且非 File 的行）
            if self._suppress_traceback:
                if not str_data.startswith(' ') and not str_data.startswith('\t') and "Traceback" not in str_data and "Exception" not in str_data:
                    self._suppress_traceback = False
                else:
                    return # 物理吞噬所有的堆栈跟踪
                    
            # 2. 拦截空行连坐
            if self._kill_next_newline and str_data in ["\n", "\r\n"]:
                self._kill_next_newline = False
                return
                
            self._kill_next_newline = False
            lower_data = str_data.lower()
            
            # 3. 物理级嗅探：拦截 WebSocket 重连噪音与超时崩溃
            if any(keyword in lower_data for keyword in["websocket", "fin=1", "opcode=8", "bye", "closed", "__auto_reconnect", "timeouterror"]):
                self._kill_next_newline = True 
                return
            
            # 4. 触发堆栈吞噬：如果检测到某个我们不关心的线程崩溃
            if "exception in thread" in lower_data and ("__auto_reconnect" in lower_data or "tts" in lower_data):
                self._suppress_traceback = True
                self._kill_next_newline = True
                return

            self.stream.write(data)

        def flush(self):
            self.stream.flush()
            
    sys.stdout = StdoutFilter(sys.stdout)
    sys.stderr = StdoutFilter(sys.stderr)
    
    # 💡 物理热键挂载：锁定 F6-F11 具身交互矩阵
    keyboard.on_press_key("f6", on_f6)
    keyboard.on_press_key("f7", on_f7)
    keyboard.on_press_key("f8", on_f8)
    keyboard.on_press_key("f9", on_f9)
    keyboard.on_press_key("f10", on_f10)
    keyboard.on_press_key("f11", on_f11) # 💡 挂载 F11 静音扳机
    
    # 💡 物理层适配：迈从 K99 极限边角按键映射 (Vertical Edge Bypass)
    keyboard.on_press_key("page up", on_f13)   # Page Up 键 -> 锁定日语 (JA)
    keyboard.on_press_key("page down", on_f14) # Page Down 键 -> 锁定英语 (EN)
    keyboard.on_press_key("delete", on_f15)    # Delete 键 -> 解除锁定 (恢复自动路由)
    
    # 💡 Numpad 小键盘扩张：利用小键盘 4 和 5 控制新增语言极性
    keyboard.on_press_key("num 4", on_num4)    # 小键盘 4 -> 锁定中文 (ZH)
    keyboard.on_press_key("num 5", on_num5)    # 小键盘 5 -> 锁定韩文 (KO)
    
    # 💡 RAG 系统激活：加载学术语料与生成语法知识库
    try:
        km = KnowledgeManager(docs_dir=RAG_DIR); km.ingest_assets()
    except Exception as e: 
        print(f"⚠️ [SYSTEM] RAG Fail: {e}"); km = None

    p = pyaudio.PyAudio()
    # 💡 物理层修复：引入 Host API 0 过滤器，强制锁定 MME 架构以兼容 ASR 链路
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i); name = dev.get('name', '')
        if dev.get('hostApi') == 0 and dev.get('maxInputChannels') > 0:
            if "Maonocaster" in name: PC_MIC_ID = i
            elif "CABLE Output" in name: PLAYER_ID = i
            elif "Virtual Desktop" in name or "Oculus" in name: QUEST_ID = i
            
    print(f"✅ [HARDWARE LOCK] Maonocaster:{PC_MIC_ID} | Cable:{PLAYER_ID} | Quest:{QUEST_ID}")

    # 💡 确定性硬件寻址：Voicemeeter AUX 虚拟母线挂载
    aux_index = next((i for i in range(p.get_device_count()) if "Voicemeeter AUX Input" in p.get_device_info_by_index(i).get('name', '')), None)
    
    if aux_index is None:
        print("🔍 [LQA AUDIT] Target bus 'Voicemeeter AUX Input' not found. Scanning available devices...")
        for i in range(p.get_device_count()):
            d_info = p.get_device_info_by_index(i)
            print(f"   -> ID {i}: {d_info.get('name')}")

    if aux_index is not None:
        print(f"✅ [L10N] Memory-Direct Stream Attached to Device ID: {aux_index}")
        tts_stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True, output_device_index=aux_index)
    else:
        print("⚠️ [SYSTEM] Voicemeeter AUX not found, routing to default speaker."); 
        tts_stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    # 💡 核心启动：激活音频渲染母线（TTS 消费者线程）
    threading.Thread(target=audio_player_worker, daemon=True).start()
    
    # 💡 具身进化：启动异步视觉探针与冷链路认知推理线程
    threading.Thread(target=vision_probe_worker, daemon=True).start()
    threading.Thread(target=cognitive_reasoner_worker, daemon=True).start()

    # 💡 ASR 辞书部署：将 Termbase 热词实时推向阿里云边缘算子
    with open(HOTWORDS_FILE, 'r', encoding='utf-8') as f: my_hotwords = json.load(f)
    v_service = VocabularyService(api_key=DASHSCOPE_API_KEY)
    try:
        existing = v_service.list_vocabularies()
        for v in existing:
            if v['vocabulary_id'].startswith('vocab-v'): v_service.delete_vocabulary(v['vocabulary_id'])
        vid = v_service.create_vocabulary(prefix="v83", target_model="fun-asr-realtime-2026-02-28", vocabulary=my_hotwords)
        print(f"✅ [HOTWORDS] Deployed: {vid}")
    except: vid = None

    # 💡 线程级联启动：逐行激活 ASR 监听，给予 SDK 必要的初始化缓冲 (0.5s)
    t1 = threading.Thread(target=start_asr_thread, args=(p, PC_MIC_ID, "ME_PC", vid), daemon=True)
    t1.start(); time.sleep(0.5) 
    
    t2 = threading.Thread(target=start_asr_thread, args=(p, QUEST_ID, "ME_QUEST", vid), daemon=True)
    t2.start(); time.sleep(0.5)
    
    t3 = threading.Thread(target=start_asr_thread, args=(p, PLAYER_ID, "PLAYER", vid), daemon=True)
    t3.start()

    print("\n>>> V12-Holographic | ALL SYSTEMS NOMINAL | PRESS F6 TO ACTIVATE")
    
    try:
        while running: 
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # 💡 安全关断与资源解构：彻底移除内存母线残留
        running = False
        print("\n[SYSTEM] Releasing cloud resources and closing audio bus...")
        
        if 'vid' in locals() and vid:
            try: v_service.delete_vocabulary(vid)
            except: pass
        
        # 释放音频 IO 句柄
        if 'tts_stream' in locals() and tts_stream is not None:
            try: 
                tts_stream.stop_stream()
                tts_stream.close()
            except: pass
        
        if 'p' in locals(): 
            p.terminate()
        
        print("✅ [SYSTEM] Shutdown Complete. RAG state saved.")