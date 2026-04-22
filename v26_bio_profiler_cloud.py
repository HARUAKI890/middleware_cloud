import os, json, time, re
from openai import OpenAI
import dashscope 
# 💡 挂载离线声学工业级引擎
import numpy as np
import parselmouth
from parselmouth.praat import call
from pydub import AudioSegment

# 💡 物理网络隔离装甲 (Proxy Firewall)：
# 强制屏蔽系统级 VPN 代理对 Python 底层 requests 的劫持，彻底消灭 SSL 握手失败！
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['no_proxy'] = '*'

# 💡 物理路径绝对锚定 (拒绝环境漂移)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "vrchat_duplex_log.md")
RAG_DIR = os.path.join(BASE_DIR, "rag_docs")
MEMOIR_IMG_DIR = os.path.join(BASE_DIR, "memoir_images")
PERSONA_FILE = os.path.join(BASE_DIR, "v25_persona_config.json")
os.makedirs(RAG_DIR, exist_ok=True); os.makedirs(MEMOIR_IMG_DIR, exist_ok=True)

REFINED_OUTPUT = os.path.join(RAG_DIR, "refined_history_gold.md")
MEMOIR_OUTPUT = os.path.join(RAG_DIR, "childhood_memoirs.md")
THOUGHT_DIR = os.path.join(BASE_DIR, "raw_thoughts")
THOUGHT_OUTPUT = os.path.join(RAG_DIR, "refined_thoughts_gold.md")
os.makedirs(THOUGHT_DIR, exist_ok=True)
# 💡 1. 物理隔离：定义 API 矩阵 (解耦国内外网络链路)
DEEPSEEK_API_KEY = "sk" # 深度求索推理大脑
DASHSCOPE_API_KEY = "sk" # 阿里灵积多模态前哨

# 💡 2. 挂载 Client (确保 DashScope 直连，压降延迟)
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
dashscope.api_key = DASHSCOPE_API_KEY


# ================= MODE 1: VRChat 历史语料洗练算子 =================
def refine_log_chunk(content, attempt=1):
    prompt = f"""You are a Linguistics Expert analyzing Dankuroi's VRChat translation logs.
1. IGNORE ASR errors, noise, and meaningless repetitive greetings.
2. EXTRACT 3-5 high-quality [Source -> Translation] pairs representing his INTP-A/Femboy style.
3. OUTPUT strictly in clean Markdown bullet points. Example format:
- **Source**:[text]
  **Trans**: [text]
  **Persona/Style**: [Analysis]

If the chunk is total garbage, output exactly: "NO VALUABLE DATA."
LOG CHUNK:
{content}"""
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.1)
        return res.choices[0].message.content
    except Exception as e:
        print(f"\n  🚨 [API ERROR] Attempt {attempt} failed: {e}")
        if attempt <= 3: time.sleep(3 * attempt); return refine_log_chunk(content, attempt + 1)
        return "ERROR: API failed after 3 retries."
    
def refine_thought_chunk(content, attempt=1):
    # 💡 核心指令：针对非结构化随笔与琐碎发言，提取逻辑锚点与语言癖好
    prompt = f"""You are a Computational Linguist. Analyze these raw personal notes/chats.
1. IDENTIFY speakers: treat unnamed blocks as Dankuroi's inner thoughts. In named chats (e.g., WeChat), focus exclusively on Dankuroi's logic.
2. EXTRACT recurring thought patterns, logic-jump styles, and specific idiosyncratic vocabulary.
3. DISCARD noise (weather talk, generic greetings, meaningless filler).
4. OUTPUT a "Cognitive Style Profile" in Markdown. 
- **Logical Flow**: [How he connects abstract concepts]
- **Syntactic Fingerprint**: [Unique phrasing or sentence structures]
- **Core Philosophy**: [The underlying value system in this segment]

RAW TEXT:
{content}"""
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.1)
        return res.choices[0].message.content
    except Exception as e:
        if attempt <= 3: time.sleep(3 * attempt); return refine_thought_chunk(content, attempt + 1)
        return "ERROR."

# ================= MODE 2: Qwen3.6-PLUS 视觉转录与灵魂提取 =================
def ocr_memoir_task():
    # 💡 断点续传前置逻辑：扫描并提取已成功落盘的页面指纹
    processed_pages = set()
    if os.path.exists(MEMOIR_OUTPUT):
        with open(MEMOIR_OUTPUT, 'r', encoding='utf-8') as f:
            processed_pages = set(re.findall(r'## Page: (.*?\.(?:jpg|png|jpeg|webp))', f.read(), re.IGNORECASE))
            
    # 💡 物理层降维：改用 'a' (Append) 追加模式，誓死捍卫已存在的 25 页数据
    with open(MEMOIR_OUTPUT, 'a', encoding='utf-8') as f:
        for img_name in sorted(os.listdir(MEMOIR_IMG_DIR)):
            valid_extensions = ('.jpg', '.png', '.jpeg', '.webp')
            if img_name.lower().endswith(valid_extensions):
                # 💡 幂等性校验：如果该页已经存在于 MD 文件中，直接跳过，0 Token 损耗
                if img_name in processed_pages:
                    print(f"⏭️ [SKIP]: {img_name} already processed. Bypassing...")
                    continue
                    
                print(f"🚀 [QWEN3.6-PLUS]: Analyzing {img_name}...")
                img_path = os.path.abspath(os.path.join(MEMOIR_IMG_DIR, img_name))
                
                msgs = [{"role": "user", "content":[
                    {"image": f"file://{img_path}"},
                    {"text": "你是一位心理学家与档案专家。请精准转录这张手写信件。注意：这是连载传记的一部分，请保持文字的原始流动感，不要在结尾加总结。模糊字用[?]。直接输出 Markdown 文本内容。"}
                ]}]
                
                # 💡 强力防波堤：针对 SSL 断流引入 3 次指数重试机制
                for attempt in range(3):
                    try:
                        response = dashscope.MultiModalConversation.call(
                            model='qwen3.6-plus', 
                            messages=msgs,
                            vl_high_resolution_images=True 
                        )
                        
                        if response.status_code == 200:
                            text_content = response.output.choices[0].message.content[0]['text']
                            f.write(f"\n\n## Page: {img_name}\n{text_content}\n")
                            f.flush(); os.fsync(f.fileno()) # 💡 再次确保存档安全
                            break # 识别成功，跳出重试循环
                        else:
                            print(f"🚨 [DASH ERROR]: {response.code} - {response.message}")
                            break # 属于 API 业务报错（非网络问题），不重试
                    except Exception as e:
                        print(f"  ⚠️[NET ERROR] SSL/Timeout on attempt {attempt+1}: {e}")
                        if attempt < 2:
                            time.sleep(5) # 💡 被掐断后休眠 5 秒，给 WAF 喘息时间
                        else:
                            print(f"❌ [FAILED] {img_name} skipped due to persistent network crashes.")
    return MEMOIR_OUTPUT

def inject_soul_task(memoir_path):
    with open(memoir_path, 'r', encoding='utf-8') as f: content = f.read()
    
    # 💡 终极最优解读提示词：心理学解构 + 语言学指纹提取
    sys_prompt = """You are a master of Psychoanalysis and a Character Architect for high-fidelity Digital Twins.
Analyze the provided childhood memoirs as a "Linguistic and Psychological Fossil."
1. Identify the core intellectual escape mechanisms and logical defense structures.
2. Trace the origins of alienation from irrational external systems (family/tradition).
3. Extract specific syntactic patterns or "mantras" that define his worldview.
OUTPUT: A single, high-density English 3rd-person paragraph (max 150 words). 
Focus on the "Cerebral Ascetic" and "Rationalist Fortress" traits. 
This will be the SOUL-KERNEL of his Digital Twin. NO summary, ONLY the distilled essence of his cognitive architecture."""

    print("🧠 [REASONING]: Utilizing DeepSeek-Reasoner for deep soul extraction...")
    # 💡 关键：切换模型为 deepseek-reasoner
    res = client.chat.completions.create(
        model="deepseek-reasoner", # 👈 物理切换至 R1 推理模型
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": content}]
    )
    soul_desc = res.choices[0].message.content
    
    with open(PERSONA_FILE, 'r', encoding='utf-8') as f: p_data = json.load(f)
    p_data["DANKUROI_PRIME"]["desc"] = soul_desc # 💡 热挂载至真身节点
    with open(PERSONA_FILE, 'w', encoding='utf-8') as f: json.dump(p_data, f, ensure_ascii=False, indent=2)
    print(f"✨ [DIGITAL TWIN]: Persona 'desc' hot-reloaded.\n\n--- SOUL FRAGMENT ---\n{soul_desc}\n---------------------")
# ================= MODE 4: 多模态角色文献提纯 (PDF/Manga -> JSON Persona) =================
LORE_DIR = os.path.join(BASE_DIR, "doc_assets", "character_lore")
os.makedirs(LORE_DIR, exist_ok=True)

def extract_lore_to_persona_task():
    print(f"\n📚 [LORE PROBE] 正在扫描 {LORE_DIR} 下的角色物理隔离区...")
    
    # 💡 架构重构：扫描子文件夹，将文件夹名作为独立的角色 ID
    character_dirs = [d for d in os.listdir(LORE_DIR) if os.path.isdir(os.path.join(LORE_DIR, d))]
    
    if not character_dirs:
        print("🚨 架构未就绪：请在 character_lore 文件夹下以角色 ID 创建子文件夹（例如 WITTGENSTEIN 或 FRIEREN）。")
        return

    # 💡 自动化批处理：遍历每一个角色文件夹，实现无人值守提纯
    for char_id in character_dirs:
        char_id = char_id.upper().strip()
        print("\n" + "="*40)
        print(f"🧬 [FORGE TARGET]: 开始锻造角色 -> {char_id}")
        print("="*40)
        
        char_dir_path = os.path.join(LORE_DIR, char_id)
        # 强制排序，确保漫画与书籍的连贯性
        lore_files = sorted([f for f in os.listdir(char_dir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        if not lore_files:
            print(f"⚠️ 跳过 {char_id}: 文件夹为空，未发现视觉锚点。")
            continue

        # 1. 影像转录与语义提纯 (Qwen3.6-PLUS)
        accumulated_text = ""
        for img_name in lore_files:
            print(f"👁️ 正在解析 {char_id} 的视网膜图像: {img_name}...")
            img_path = os.path.join(char_dir_path, img_name)
            
            # 针对漫画与纯文本的通用识别 Prompt
            msgs = [{"role": "user", "content":[
                {"image": f"file://{img_path}"},
                {"text": "你是一位跨语言文学分析师。请提取图片中的所有文本（包括日文或英文）。如果是漫画，请描述角色的表情特征和语气习惯。忽略不重要的背景噪音。"}
            ]}]
            
            try:
                res = dashscope.MultiModalConversation.call(model='qwen3.6-plus', messages=msgs)
                if res.status_code == 200:
                    accumulated_text += f"\n--- Source: {img_name} ---\n{res.output.choices[0].message.content[0]['text']}"
                else:
                    print(f"🚨 [DASH ERROR]: {res.code} - {res.message}")
            except Exception as e:
                print(f"⚠️ 跳过 {img_name}: {e}")

        if len(accumulated_text) < 10:
            print(f"⚠️ 提取失败: {char_id} 的有效文本不足，跳过推理。")
            continue

        # 2. 降维打击：DeepSeek-Reasoner 将散乱文本铸造为 V25 人格格式
        print(f"\n🧠 [REASONER] 启动高维逻辑融合，正在为 {char_id} 锻造 Persona JSON 碎片...")
        
        prompt = f"""[Cross-Modal Persona Forge Task - V30 Dynamic Architecture]
        Analyze the literature/manga extraction of the character '{char_id}':
        {accumulated_text}
        
        Task: Create a V30 Dynamic Persona JSON block. This acts as a Translation Skill State Machine.
        CRITICAL V30 STRUCTURE:
        1. 'core_identity': 'desc' MUST be nested dict ('en', 'ja', 'zh'). 'voice_texture' objectively describes their acoustic timbre.
        2. 'translation_skills': 'lexical_mapping' maps custom vocabulary. 'syntactic_rules' defines sentence flow and particles.
        3. 'dynamic_states': MUST include "STATE_CASUAL" and "STATE_COMBAT" (or "STATE_INTENSE").
           - 'tts_override': MUST contain 'vol_scale', 'rate', 'pitch' as floats. (e.g., Frieren casual rate=0.85).
        
        Output ONLY valid JSON matching this exact structure:
        {{
          "{char_id}": {{
            "core_identity": {{
              "desc": {{"en": "...", "ja": "...", "zh": "..."}},
              "voice_texture": "..."
            }},
            "translation_skills": {{
              "lexical_mapping": {{"keyword":["trans1", "trans2"]}},
              "syntactic_rules": "..."
            }},
            "dynamic_states": {{
              "STATE_CASUAL": {{
                "trigger_condition": "Normal conversational context",
                "tts_override": {{"vol_scale": 1.0, "rate": 1.0, "pitch": 1.0}},
                "pragmatics": "..."
              }},
              "STATE_COMBAT": {{
                "trigger_condition": "High stress or intense emotion",
                "tts_override": {{"vol_scale": 1.2, "rate": 1.1, "pitch": 1.05}},
                "pragmatics": "..."
              }}
            }}
          }}
        }}"""

        try:
            ds_res = client.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": prompt}])
            raw_json = ds_res.choices[0].message.content
            match = re.search(r'\{.*\}', raw_json, re.DOTALL)
            persona_data = json.loads(match.group()) if match else json.loads(raw_json)
            
            # 3. 资产单独落盘：生成独立的碎片卡牌
            out_path = os.path.join(BASE_DIR, f"v25_persona_{char_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(persona_data, f, ensure_ascii=False, indent=2)
            print(f"✨ [ASSET FORGED] {char_id} 角色指纹已成功落盘: {out_path}。")
            
            # 为了防止并发请求过快导致阿里/DeepSeek API 熔断，物理休眠 3 秒
            time.sleep(3)
            
        except Exception as e:
            print(f"🚨 {char_id} 碎片锻造失败: {e}")

# ================= MODE 5: RAG 长文本小说知识蒸馏 (Novel/Lore Distillation) =================
def distill_novel_memory_task():
    print(f"\n📚[MEMORY REFINERY] 启动长文本知识蒸馏工厂 (EPUB -> RAG Crystal)...")
    # 💡 物理路由重构：输入是从 doc_assets 获取原石，输出是压入 rag_docs 中枢
    raw_dir = os.path.join(BASE_DIR, "doc_assets", "character_lore")
    rag_target_dir = os.path.join(BASE_DIR, "rag_docs", "character_memories")
    
    # 扫描所有的 MD 文件
    md_files =[]
    for root, _, files in os.walk(raw_dir): # 👈 改为扫描 raw_dir
        for file in files:
            if file.endswith(".md") and "distilled" not in file and file != "dynamic_log.md":
                md_files.append(os.path.join(root, file))
                
    if not md_files:
        print("🚨 未在 character_memories 中发现可蒸馏的原始 MD 文献。")
        return
        
    print("📁 发现以下可蒸馏的 RAG 原石：")
    for idx, path in enumerate(md_files):
        rel_path = os.path.relpath(path, raw_dir)
        print(f"  [{idx+1}] {rel_path}")
        
    try:
        choice = int(input("\n请输入要蒸馏的文件序号: ")) - 1
        target_file = md_files[choice]
    except:
        print("❌ 选择无效。")
        return
        
    char_name = input("请输入该文献所属的角色 ID (如 FRIEREN): ").strip().upper()
    
    # 💡 物理分流：定义蒸馏通道矩阵
    doc_type = input("\n请选择蒸馏提纯的语料范式:\n[1] ACGN 小说/剧情 (提取世界观、源石技艺、人际关系)\n[2] 纯学术随笔/书信 (提取哲学公理、认知逻辑、语用学脾气)\nChoice [1/2]: ").strip()
    is_academic = (doc_type == '2')
    
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 💡 物理防爆锁：按双换行符切分，组合成大约 3000 字的 Chunk 送给 LLM
    paragraphs = content.split('\n\n')
    chunks =[]
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) > 3000:
            chunks.append(current_chunk)
            current_chunk = p + "\n\n"
        else:
            current_chunk += p + "\n\n"
    if current_chunk: chunks.append(current_chunk)

    # 💡 自动映射存放路径：将提纯后的晶体存入对应的 rag_docs 子文件夹
    final_out_dir = os.path.join(rag_target_dir, char_name)
    os.makedirs(final_out_dir, exist_ok=True)
    out_file = os.path.join(final_out_dir, f"{char_name}_distilled_{int(time.time())}.md")
    
    print(f"🚀 [REASONER] 原石已切分为 {len(chunks)} 块，启动高维语义萃取...")

    with open(out_file, 'w', encoding='utf-8') as out_f:
        out_f.write(f"# Distilled Memory Matrix: {char_name}\n\n")
        
        for i, chunk in enumerate(chunks):
            print(f"⏳ 正在萃取切片 [{i+1}/{len(chunks)}]...", end="", flush=True)
            
            # 💡 架构师级解耦：针对纯学术/书信与二次元小说的双轨 Prompt
            if is_academic:
                prompt = f"""You are a RAG Database Curator & Philosophical Profiler.
Analyze this raw excerpt from the academic notes/letters of '{char_name}'.
1. EXTRACT CORE AXIOMS: Summarize the primary philosophical or logical propositions mentioned in this chunk.
2. EXTRACT COGNITIVE STYLE & TONE (CRITICAL): How does {char_name} argue? Identify their tone (e.g., arrogant, cold, axiomatic, dismissive of stupidity, desperate for logical clarity). 
3. IDIOSYNCRATIC VOCABULARY: Record any specific terminology used to define their world boundary (e.g., "State of affairs", "Logical multiplicity").
DISCARD generic greetings, weather complaints, or irrelevant daily filler.
OUTPUT format: A concise Markdown list of highly dense facts and psychological observations. Do not wrap in ```markdown blocks, just pure text.

RAW TEXT:
{chunk}"""
            else:
                prompt = f"""You are a RAG Database Curator & Lore Expert.
Analyze this raw excerpt from an ACGN EPUB/script. 
1. If the target '{char_name}' is a SPECIFIC CHARACTER: Extract their history, combat skills/magic (e.g., Originium Arts), personality, and relationships.
2. If the target '{char_name}' is 'WORLD' or 'LORE': Extract geopolitical facts, Factions, Catastrophes, and Technology.
DISCARD meaningless environmental descriptions and filler words. Retain EXACT terminology.
OUTPUT format: A concise Markdown list of highly dense facts. Do not wrap in ```markdown blocks, just pure text.

RAW TEXT:
{chunk}"""

            try:
                # 物理调用 Reasoner
                res = client.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": prompt}])
                distilled_text = res.choices[0].message.content
                out_f.write(f"## Memory Block {i+1}\n{distilled_text}\n\n")
                out_f.flush()
                print(" ✅ 完成")
                time.sleep(2) # 防熔断冷却
            except Exception as e:
                print(f" 🚨 失败: {e}")
                
    print(f"✨ [DONE] 高维记忆晶体已落盘: {out_file}")
    print("💡 建议：去文件夹中删掉原版的 MD 文件，让 KnowledgeManager 下次只读取这个提纯版！")

# ================= MODE 6: 视觉百科全书压制 (Guidebook OCR -> RAG MD) =================
def ocr_guidebook_to_rag_task():
    print(f"\n📚 [VISION TO RAG] 启动视觉文献提纯工厂...")
    guide_dir = os.path.join(BASE_DIR, "doc_assets", "character_lore")
    
    char_dirs =[d for d in os.listdir(guide_dir) if os.path.isdir(os.path.join(guide_dir, d))]
    if not char_dirs: return print("🚨 未找到角色文献目录。")
        
    print("📁 发现以下物理隔离区：")
    for idx, d in enumerate(char_dirs): print(f"  [{idx+1}] {d}")
    try:
        choice = int(input("\n请选择要进行 OCR 的百科全书目录: ")) - 1
        target_dir_name = char_dirs[choice]
    except: return print("❌ 无效选择。")

    target_dir_path = os.path.join(guide_dir, target_dir_name)
    
    # 💡 物理层降维：废除递归 os.walk。只扫描当前选中的第一级目录。
    # 彻底杜绝《ARTS1》和《WORLD》被强行混合缝合成一个巨大 MD 的污染现象！
    img_files =[]
    for f in os.listdir(target_dir_path):
        full_path = os.path.join(target_dir_path, f)
        if os.path.isfile(full_path) and f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
            img_files.append(full_path)
            
    # 💡 物理层修复：植入 Natural Sort (自然排序) 引擎。
    # 彻底解决 10 排在 2 前面、100 排在 11 前面的字典序灾难。
    # 它会将 'index-10_1.jpg' 切分为 ['index-', 10, '_', 1, '.jpg'] 进行绝对数值对比。
    def natural_sort_key(s):
        import re
        return[int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        
    img_files = sorted(img_files, key=natural_sort_key)
    
    if not img_files: 
        return print(f"🚨 目录 [{target_dir_name}] 下没有直接存放图片！如果你有子文件夹，请在 doc_assets 把它提出来！")

    print(f"\n🎯 扫描到 {len(img_files)} 张图像实体。")
    target_persona = input("请输入这些记忆要注入的 RAG 角色 ID (例如 FRIEREN_EARLY): ").strip().upper()
    if not target_persona: return print("❌ 角色 ID 不能为空。")
    
    # 💡 物理分流：隔离二次元设定集与纯学术书籍的 OCR 逻辑
    doc_type = input("\n请选择文献类型:\n[1] ACG 设定集/画集 (提取美术特征与游戏世界观)\n[2] 纯学术文献/扫描版书籍 (高精度排版转录，抗幻觉)\nChoice [1/2]: ").strip()
    is_academic = (doc_type == '2')

    rag_target_dir = os.path.join(RAG_DIR, "character_memories", target_persona)
    os.makedirs(rag_target_dir, exist_ok=True)
    
    suffix = "_Academic" if is_academic else "_Guidebook"
    out_md_path = os.path.join(rag_target_dir, f"{target_dir_name}{suffix}.md")

    # 💡 补救护甲 1：断点续传状态嗅探 (State Recovery)
    processed_records = set()
    if os.path.exists(out_md_path):
        with open(out_md_path, "r", encoding="utf-8") as f:
            # 使用正则抓取所有已经写入的图片记录头
            processed_records = set(re.findall(r'## Record: (.*?\.jpg|.*?\.png|.*?\.jpeg|.*?\.webp)', f.read(), re.IGNORECASE))
        print(f"♻️ [STATE RECOVERY] 检测到已存在 {len(processed_records)} 页记忆晶体，启动断点续传。")
        file_mode = "a" # 改为追加模式
    else:
        file_mode = "w" # 全新写入模式

    print(f"🚀[QWEN3.6-PLUS] 正在对 {len(img_files)} 页档案执行深度跨模态 OCR...")
    print(f"⚠️ 提示：具备断点续传与防坠网重试机制。随时可按 Ctrl+C 中止。")
    
    with open(out_md_path, file_mode, encoding="utf-8") as f:
        # 如果是新文件，写入总标题
        if file_mode == "w":
            f.write(f"# Encyclopedia Matrix: {target_persona} - {target_dir_name}\n\n")
        
        for idx, img_path in enumerate(img_files):
            img_name = os.path.basename(img_path)
            
            # 💡 补救护甲 2：无损跳跃已解析资源
            if img_name in processed_records:
                print(f"⏭️[{idx+1}/{len(img_files)}] 缓存命中，跳过转录: {img_name}")
                continue
                
            print(f"👁️ [{idx+1}/{len(img_files)}] 正在转录: {img_name}...", end="", flush=True)
            
            # 💡 动态视觉降维 (In-Memory Compression)：
            # 彻底解决 2.0x 爆破图片产生的 5MB+ Base64 导致阿里云 WAF 强行切断 SSL 的问题！
            import base64
            from io import BytesIO
            from PIL import Image
            try:
                with Image.open(img_path) as img:
                    # 视觉特征对齐：Qwen-VL 最优识别长边为 1568。若超过则降采样，不超则保留。
                    max_dim = 1568
                    if max(img.size) > max_dim:
                        ratio = max_dim / max(img.size)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # 将 RGB 图像写入内存缓冲区，压制为质量 85 的 JPEG（体积通常小于 500KB）
                    buffer = BytesIO()
                    img.convert('RGB').save(buffer, format="JPEG", quality=85)
                    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                b64_image = f"data:image/jpeg;base64,{encoded_string}"
            except Exception as e:
                print(f" ⚠️ 图像降维失败，跳过: {e}")
                continue
            
            # 💡 动态提示词路由：彻底粉碎大模型在面对空白页时的脑补幻觉
            if is_academic:
                prompt_text = """你是一个严谨的学术文献 OCR 引擎。请高精度转录此扫描页。
1. 【精准转录】：一字不落地转录图像中的所有文本。保留原语言（如英文、德文），不要随意翻译。
2. 【排版还原】：保留原书的段落结构、章节编号、脚注与斜体。
3. 【抗幻觉】(CRITICAL)：如果页面是空白的，或者只有微小的扫描噪点，请直接输出 "[BLANK PAGE]"。绝不要臆造任何游戏、小说或机构的专有名词！
4. 【RAG 锚点】：提取本页核心主旨，在开头用 `# ` 作为一级标题。直接输出纯正的 Markdown。"""
            else:
                prompt_text = """你是一个无情的游戏设定集提取器（Lore Extractor）。请深度解析这张画集/设定集图片：
1. 【文本提取】：精准转录/翻译图中的设定文本。保留专有名词的原语（如 源石 Originium, 罗德岛 Rhodes Island）。
2. 【视觉降维】：提取图中的核心美术情报（如：武器构造细节、阵营标志/Logo、角色的服装材质与表情差分），用有条理的中文描述。
3. 【RAG 锚点注入】(CRITICAL)：必须使用 `# 一级标题` 标明页面的核心实体，使用 `## 二级标题` 标明子模块属性。
4. 【抗幻觉与去噪】：如果页面过曝或全白，直接输出 "[BLANK PAGE]"。绝不要输出“这是一张图片”之类的废话。直接输出纯正的 Markdown。"""

            msgs =[{"role": "user", "content":[
                {"image": b64_image}, # 👈 物理直传，无需走云端中转
                {"text": prompt_text}
            ]}]
            
            # 💡 补救护甲 3：微重试循环 (Micro-Retry Loop)
            success = False
            for attempt in range(3):
                try:
                    res = dashscope.MultiModalConversation.call(model='qwen3.6-plus', messages=msgs)
                    if res.status_code == 200:
                        text_content = res.output.choices[0].message.content[0]['text']
                        f.write(f"## Record: {img_name}\n{text_content}\n\n")
                        f.flush() 
                        print(" ✅ 完成")
                        success = True
                        time.sleep(2) 
                        break
                    else: 
                        # 💡 审查护盾：如果明确被官方绿网拦截，立刻跳出重试循环，不再做无效挣扎！
                        if "DataInspectionFailed" in str(res.code) or "DataInspectionFailed" in str(res.message):
                            print(f" 🛑[WAF BLOCKED] 触发阿里云绿网审查 (涉暴/涉黄/猎奇)。已强行跳过。")
                            f.write(f"## Record: {img_name}\n[SYSTEM LOG: Image skipped due to Cloud WAF Content Inspection.]\n\n")
                            f.flush()
                            success = True # 伪装成功以继续下一张图
                            break
                        else:
                            print(f" 🚨 API拒绝 ({res.code})，准备重试...")
                            time.sleep(5)
                except Exception as e:
                    print(f" ⚠️ 崩溃 (第{attempt+1}次尝试): {e}")
                    time.sleep(5 * (attempt + 1)) 
            
            if not success:
                print(f" ❌ 彻底失败，已跳过 {img_name}。")
                f.write(f"## Record: {img_name}\n[SYSTEM LOG: Image skipped due to persistent network failure.]\n\n")
                f.flush()
                
    print(f"\n✨[DONE] 百科全书已被压制为 RAG 记忆晶体: {out_md_path}")
    print("💡 接下来你可以启动 KnowledgeManager，将这些文本化作向量注入 ChromaDB！")

# ================= MODE 8: 离线声意多模态锻造 (Praat Objective + Qwen Subjective + DeepSeek R1) =================
def acoustic_semantic_forge_task():
    print(f"\n🎵 [ACOUSTIC-SEMANTIC FORGE] 启动全自动声学缝合与语用学融合引擎...")
    char_id = input("👉 请输入目标角色 ID (如 KROOS_EARLY): ").strip().upper()
    if not char_id: return
    
    # 💡 宏大叙事扩容：允许注入世界观 ID (如 ARKNIGHTS)
    universe_id = input("👉 请输入该角色所属的世界观 ID (可选，如 ARKNIGHTS，直接回车跳过): ").strip().upper()
    
    # 💡 路径自动化路由
    lore_dir = os.path.join(BASE_DIR, "doc_assets", "character_lore", char_id)
    
    # 💡 物理扩展：允许强制重定向干声文件夹扫描路径
    custom_wav_dir = input(f"👉 请输入干声碎片的绝对路径 (直接回车则默认扫描 {lore_dir}):\n> ").strip().strip('"').strip("'")
    audio_scan_dir = custom_wav_dir if custom_wav_dir and os.path.exists(custom_wav_dir) else lore_dir
    
    # 指向 RAG 中枢里的晶体文件 (如 WITT_EARLY_distilled.md 或 官方设定 MD)
    md_path = os.path.join(RAG_DIR, "character_memories", char_id, f"{char_id}.md") 
    
    if not os.path.exists(audio_scan_dir): return print(f"🚨 找不到原声素材目录: {audio_scan_dir}")
    if not os.path.exists(md_path): print(f"⚠️ 未找到专属 MD ({md_path})，将仅依赖音频进行部分推理。")

    # =================================================================
    # 💡 1. 物理层：Pydub 自动化音频矩阵缝合 (无需手动使用 Audacity)
    # =================================================================
    print("✂️[PYDUB SPLICER] 正在扫描并缝合零散干声...")
    # 💡 物理排异护甲：绝对屏蔽已经生成的 Matrix 和 temp 缓存文件，彻底粉碎套娃融合导致的文件无限膨胀！
    wav_files =[f for f in os.listdir(audio_scan_dir) if f.lower().endswith(('.wav', '.mp3', '.ogg')) and "Acoustic_Matrix" not in f and "omni_temp" not in f]
    if not wav_files: return print(f"🚨 目录 [{audio_scan_dir}] 下没有发现音频文件！")
    
    combined_audio = AudioSegment.empty()
    for f in wav_files:
        try:
            # 读取音频，强制转为 16kHz 单声道 (工业识别标准)，并在每句话后面加 0.5 秒静音防粘连
            seg = AudioSegment.from_file(os.path.join(audio_scan_dir, f)).set_frame_rate(16000).set_channels(1)
            combined_audio += seg + AudioSegment.silent(duration=500)
        except Exception as e: print(f" ⚠️ 跳过损坏的音频 {f}: {e}")
        
    matrix_wav_path = os.path.join(audio_scan_dir, f"{char_id}_Acoustic_Matrix.wav")
    combined_audio.export(matrix_wav_path, format="wav")
    print(f"✅[PYDUB] 成功缝合 {len(wav_files)} 段干声，生成 {len(combined_audio)/1000:.1f} 秒的声学载体: {matrix_wav_path}")

    # =================================================================
    # 💡 2. 客观张量提取层：Praat (parselmouth) 语言学深度剖析与可视化
    # =================================================================
    print("🔬[PRAAT ENGINE] 正在提取客观声带张量 (F0 & Jitter) 并渲染声谱图...")
    praat_report = "Praat Acoustic Tensor:[Data Extraction Failed]"
    try:
        snd = parselmouth.Sound(matrix_wav_path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        valid_pitch = pitch_values[pitch_values > 0]
        
        if len(valid_pitch) > 0:
            pitch_mean = valid_pitch.mean()
            pitch_std = valid_pitch.std() # 标准差代表情绪波动率
            point_process = call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)
            jitter = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100
            shimmer = call([snd, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6) * 100
            
            # 💡 调音语音学视觉装甲 (Articulatory Phonetics Visualizer)
            # 将看不见的物理波形，转化为包含频域、时域与 F0 轮廓的 4K 级声谱底片！
            try:
                import matplotlib.pyplot as plt
                spectrogram = snd.to_spectrogram()
                plt.figure(figsize=(16, 6)) # 工业级宽屏比例
                
                # 绘制宽带声谱底片 (动态范围锁定 70dB 过滤底噪)
                X, Y = spectrogram.x_grid(), spectrogram.y_grid()
                sg_db = 10 * np.log10(spectrogram.values)
                plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - 70, cmap='magma')
                
                # 叠加 F0 基频轨迹 (Cyan 色，规避底片暖色调)
                pitch_times = pitch.xs()
                pitch_freqs = pitch.selected_array['frequency']
                pitch_freqs[pitch_freqs == 0] = np.nan # 物理屏蔽无声段，防止连线坠底
                plt.plot(pitch_times, pitch_freqs, 'o', markersize=2, color='cyan', label='F0 (Pitch Contour)')
                
                plt.ylim([0, spectrogram.ymax])
                plt.ylabel("Frequency[Hz]")
                plt.xlabel("Time [s]")
                plt.title(f"Acoustic Spectrogram & Pitch Contour: {char_id}")
                plt.legend(loc='upper right')
                
                # 物理落盘
                plot_path = matrix_wav_path.replace(".wav", "_Spectrogram.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📊 [VISUALIZER] 调音学声谱图已刻录至: {plot_path}")
            except ImportError:
                print("⚠️ [VISUALIZER] 缺少 matplotlib，跳过声谱图渲染。(请运行 pip install matplotlib)")
            except Exception as plot_e:
                print(f"⚠️[VISUALIZER] 声谱渲染崩溃: {plot_e}")

            praat_report = f"""
- Pitch (F0) Mean: {pitch_mean:.1f} Hz (Base Voice Tone)
- Pitch (F0) StdDev: {pitch_std:.1f} Hz (Emotional Dynamic Range)
- Jitter (Roughness): {jitter:.2f}% (Vocal Cord Tension/Breathiness)
- Shimmer: {shimmer:.2f}% (Amplitude Instability)
*Analysis*: F0 Mean determines age/gender. High F0 StdDev indicates extreme emotional expression or energetic combat shouts. High Jitter indicates breathiness, sleepiness, or physical exhaustion.
"""
            print(f"✅ [PRAAT] 提取完成: 均频 {pitch_mean:.1f}Hz, 波动方差 {pitch_std:.1f}Hz, 粗糙度 {jitter:.2f}%")
    except Exception as e: print(f"⚠️ Praat 分析失败: {e}")

    # =================================================================
    # 💡 3. 主观感知层：Qwen Omni Captioner (感性听觉微观抽样)
    # =================================================================
    print("🔊 [OMNI CAPTIONER] 正在向阿里云投递 30s 声学切片以提取主观表现力...")
    audio_caption = ""
    try:
        # 强行截取前 30 秒用于感性听感测绘
        omni_sample = combined_audio[:30000] if len(combined_audio) > 30000 else combined_audio
        # 💡 获取绝对物理路径
        omni_temp_path = os.path.abspath(os.path.join(lore_dir, "omni_temp_slice.wav"))
        omni_sample.export(omni_temp_path, format="wav")

        # 💡 阿里 SDK 底层 Bug 修复：SDK 会强制切除 "file://"。如果用标准的三斜杠，剩下的 "/C:/" 会导致 Windows 判定文件不存在！
        # 必须手动拼接为 "file://C:/..."，让它切完正好剩下完美的绝对路径！
        audio_url = f"file://{omni_temp_path.replace('\\', '/')}"
        
        # 💡 物理层致命破局：严格遵循 Omni-Captioner 官方底层协议！
        # 探针模型【仅接收音频】，携带任何 Text 都会触发 InvalidParameter 拦截！直接裸投！
        msgs =[{"role": "user", "content":[{"audio": audio_url}]}]
        
        res = dashscope.MultiModalConversation.call(model="qwen3-omni-30b-a3b-captioner", messages=msgs)
        
        if os.path.exists(omni_temp_path): os.remove(omni_temp_path) # 阅后即焚

        if res.status_code == 200:
            audio_caption = res.output.choices[0].message.content[0]['text']
            print(f"👂 [OMNI] 主观听感提取完成。")
        else: print(f"🚨 Omni 识别失败: {res.code}")
    except Exception as e: print(f"🚨 Omni 通信断裂: {e}")

    # =================================================================
    # 💡 4. 认知融合层：DeepSeek Reasoner (R1) 宏大叙事与声学融合
    # 彻底打通单体角色 (KROOS) 与 宏大世界观 (ARKNIGHTS) 的 RAG 知识壁垒！
    # =================================================================
    text_lore = ""
    
    # A. 物理贯穿：加载世界观公有库 (Universe Lore)
    if universe_id:
        uni_dir = os.path.join(RAG_DIR, "character_memories", universe_id)
        if os.path.exists(uni_dir):
            for f_name in os.listdir(uni_dir):
                if f_name.endswith(".md"):
                    with open(os.path.join(uni_dir, f_name), 'r', encoding='utf-8') as mf:
                        # 💡 截取前 15000 字符防止 Token 溢出
                        text_lore += f"\n---[UNIVERSE LORE: {f_name}] ---\n{mf.read()[:15000]}\n"
                        
    # B. 物理贯穿：加载角色私有档案 (Micro Character Profile)
    if os.path.exists(md_path):
        with open(md_path, 'r', encoding='utf-8') as f: 
            text_lore += f"\n---[CHARACTER LORE: {char_id}] ---\n{f.read()}\n"

    if not text_lore.strip():
        print(f"⚠️ 未找到专属 MD ({md_path}) 或世界观设定，将仅依赖音频进行部分推理。")

    print(f"\n🧠 [REASONER] 启动高维交叉推理，已注满 {len(text_lore)} 字符的设定张量...")
    print(f"🧠 正在无缝缝合客观张量 (Praat)、主观听感 (Omni) 与宏大叙事...")
    # 💡 物理层降维打击：R1 提示词终极进化，融合宏观世界观萃取与时空相位锁！
    prompt = f"""[Acoustic-Semantic Skill Forge Task - V30 Dynamic Architecture]
    You are a Master Computational Linguist, L10n Architect, and Psychological Profiler.
    Your task is to merge Objective Physics (Praat), Subjective Audio (Omni), and Textual/Philosophical Lore (Wiki/Books) into a Digital Twin State Machine. 
    This Twin acts as a "Simultaneous Translation Filter". It translates the user's raw input into the target language using the exact psychological logic, vocal texture, and syntax of the target entity.
    
    Target Entity: {char_id}
    [1. Objective Praat Data]: {praat_report}[2. Subjective Audio Profile]: {audio_caption}
    [3. Macro-World & Micro-Character Lore]: {text_lore[:12000]}
    
    Task: Output a V30 Dynamic Persona JSON block.
    CRITICAL V30 STRUCTURE:
    1. 'core_identity': 
       - 'desc' (nested dict en/ja/zh): Describe their psychology, worldview, and epistemology. CRITICAL: Acknowledge their specific TEMPORAL PHASE (e.g., Early naive vs. Late era) based on the entity's ID and lore.
       - 'voice_texture': Objectively describe their acoustic timbre based on Praat/Omni (e.g., "Crystal clear, cold resonance, low Jitter, authoritative").
    2. 'translation_skills' (MACRO-MICRO LORE FUSION & CORE SKILL DISTILLATION): 
       - 'lexical_mapping': EXTRACT MACRO-WORLD TERMINOLOGY (e.g., "Originium" for Arknights, or "Logical Atomism" for Philosophy) AND micro-character catchphrases. Map them strictly.
       - 'syntactic_rules': Define their absolute grammatical laws. (e.g., "Use axiomatic, declarative short sentences. Avoid emotional adjectives. Emphasize logical absolutes. Translate into formal Japanese 'である/だ' instead of 'です/ます'").
    3. 'dynamic_states': Map their psychological duality. 
       - "STATE_CASUAL" (or STATE_OBSERVATION): Normal analytical/relaxed state.
       - "STATE_COMBAT" (or STATE_AXIOMATIC): When enforcing absolute truth, expressing anger, or facing intense scenarios. 
       - 'tts_override': MUST contain 'vol_scale', 'rate', 'pitch' as floats. Align these with the Praat Data (e.g., if F0 is extremely stable/low stddev, reflect that coldness in the rate/pitch).
       - 'pragmatics': How the translation style shifts in this state.
    
    Output ONLY valid JSON matching this exact structure:
    {{
      "{char_id}": {{
        "core_identity": {{
          "desc": {{"en": "...", "ja": "...", "zh": "..."}},
          "voice_texture": "..."
        }},
        "translation_skills": {{
          "lexical_mapping": {{"keyword": ["trans1", "trans2"]}},
          "syntactic_rules": "..."
        }},
        "dynamic_states": {{
          "STATE_CASUAL": {{
            "trigger_condition": "Normal context or passive observation",
            "tts_override": {{"vol_scale": 1.0, "rate": 1.0, "pitch": 1.0}},
            "pragmatics": "..."
          }},
          "STATE_COMBAT": {{
            "trigger_condition": "Enforcing absolute logic, intense emotion, or combat",
            "tts_override": {{"vol_scale": 1.1, "rate": 0.95, "pitch": 0.9}},
            "pragmatics": "..."
          }}
        }}
      }}
    }}"""

    try:
        ds_res = client.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": prompt}])
        raw_json = ds_res.choices[0].message.content
        match = re.search(r'\{.*\}', raw_json, re.DOTALL)
        persona_data = json.loads(match.group()) if match else json.loads(raw_json)
        
        out_path = os.path.join(BASE_DIR, f"v25_persona_{char_id}.json")
        with open(out_path, "w", encoding="utf-8") as f: json.dump(persona_data, f, ensure_ascii=False, indent=2)
        print(f"✨[ASSET FORGED] 工业级声意对齐完毕！人格字典已落盘: {out_path}")
    except Exception as e: print(f"🚨 R1 锻造失败: {e}")

# ================= MODE 9: 专辑级声学与叙事蒸馏 (Vocaloid Album -> RAG MD) =================
def vocaloid_album_distillation_task():
    print(f"\n💿 [ALBUM DISTILLERY] 启动 Vocaloid/P主 专辑级声意降维工厂...")
    album_id = input("👉 请输入专辑/P主 ID (如 KIKUO_MIKU_6): ").strip().upper()
    if not album_id: return
    
    album_dir = os.path.join(BASE_DIR, "doc_assets", "vocaloid_tracks", album_id)
    if not os.path.exists(album_dir): return print(f"🚨 找不到专辑物理目录: {album_dir}")
    
    tracks = sorted([f for f in os.listdir(album_dir) if f.lower().endswith(('.wav', '.mp3'))])
    if not tracks: return print("🚨 专辑内未发现音频轨道！")

    out_dir = os.path.join(RAG_DIR, "character_memories", album_id)
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"{album_id}_Album_Lore.md")

    print(f"🚀 发现 {len(tracks)} 首核心曲目。开始逐轨解析并铸造 MD 记忆晶体...")
    
    import base64
    with open(md_path, 'w', encoding='utf-8') as md:
        md.write(f"# Album Narrative & Acoustic Matrix: {album_id}\n\n")
        
        for idx, track_file in enumerate(tracks):
            base_name = os.path.splitext(track_file)[0]
            audio_path = os.path.join(album_dir, track_file)
            lyrics_path = os.path.join(album_dir, f"{base_name}.txt")
            
            lyrics_content = "No lyrics provided."
            if os.path.exists(lyrics_path):
                with open(lyrics_path, 'r', encoding='utf-8') as lf: lyrics_content = lf.read()

            print(f"\n🎵 [Track {idx+1}/{len(tracks)}] {base_name}")
            
            # 1. 物理切片：提取高潮 (假设高潮在 1 分钟左右)，截取 30 秒
            try:
                seg = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)
                slice_start = 60000 if len(seg) > 90000 else 0
                omni_sample = seg[slice_start:slice_start + 30000]
                
                # Praat 提取全曲宏观张量
                snd = parselmouth.Sound(np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0, sampling_frequency=16000)
                pitch = snd.to_pitch()
                pitch_values = pitch.selected_array['frequency']
                valid_pitch = pitch_values[pitch_values > 0]
                pitch_mean = valid_pitch.mean() if len(valid_pitch) > 0 else 0
                pitch_std = valid_pitch.std() if len(valid_pitch) > 0 else 0
                
                # 💡 官方物理直传：绕过 Base64，保障稳定性
                omni_temp = os.path.abspath(os.path.join(album_dir, "temp_omni.wav"))
                omni_sample.export(omni_temp, format="wav")
                
                # 💡 同理修复 SDK 的 Windows 路径解析 Bug
                audio_url = f"file://{omni_temp.replace('\\', '/')}"
                
                # 💡 物理层降维：同步剔除非法 Prompt
                msgs =[{"role": "user", "content":[{"audio": audio_url}]}]
                res = dashscope.MultiModalConversation.call(model="qwen3-omni-30b-a3b-captioner", messages=msgs)
                if os.path.exists(omni_temp): os.remove(omni_temp)
                audio_caption = res.output.choices[0].message.content[0]['text'] if res.status_code == 200 else "Omni extraction failed."
                
                # 写入 MD 切片
                md.write(f"## Track: {base_name}\n")
                md.write(f"### Objective Acoustic Tensor\n- F0 Mean: {pitch_mean:.1f}Hz\n- F0 StdDev: {pitch_std:.1f}Hz\n\n")
                md.write(f"### Subjective Musicality\n{audio_caption}\n\n")
                md.write(f"### Thematic Lyrics\n```text\n{lyrics_content}\n```\n\n")
                md.flush()
                print(" ✅ RAG 锚点已刻录")
                time.sleep(2) # 防熔断
            except Exception as e: print(f" ⚠️ 轨道解析失败: {e}")

    print(f"\n✨[DONE] 专辑全维记忆晶体已封存至: {md_path}")
    print("💡 接下来，请使用 Mode 7 (Universe-to-Persona Forge)，将这个 MD 转化为 P主/专辑 的终极人格 JSON！")

if __name__ == "__main__":
    # 💡 任务路由：多轨并发逻辑入口
    print("\n" + "="*40 + "\n[DANKUROI DATA HUB - V26]\n" + "="*40)
    mode = input("""Select Mode: 
[1] Refine VRChat Logs (Shard-based)
[2] OCR Memoirs & Inject Soul (Vision-based)
[3] Refine Daily Thoughts & Notes (File-based)[4] Extract Lore to Persona (Literature/Manga-based)
[5] Distill RAG Memory (Novel/Script text to Facts)
[6] OCR Guidebook to RAG Memory (Vision -> MD)[7] Universe-to-Persona Forge (World Lore + Profile -> JSON)
[8] Acoustic-Semantic Forge (Audio Wav + Wiki MD -> JSON Persona)
[9] Vocaloid Album Distillation (Audio+Lyrics -> RAG MD)
Choice: """).strip()

    if mode == "1":
        # 🛡️ 核心对齐：100% 保留 150 行分片与流量整形逻辑
        if not os.path.exists(LOG_FILE): print(f"🚨 致命错误: 找不到日志文件 {LOG_FILE}"); exit(1)
        print("🚀 [REFINERY] Reading log file...")
        with open(LOG_FILE, 'r', encoding='utf-8') as f: lines = f.readlines()
        chunks = []; current_chunk = ""
        for line in lines:
            current_chunk += line
            if current_chunk.count('\n') >= 150: chunks.append(current_chunk); current_chunk = ""
        if current_chunk: chunks.append(current_chunk)
        print(f"🚀 [REFINERY] Log split into {len(chunks)} shards. Starting refinement...")
        with open(REFINED_OUTPUT, 'w', encoding='utf-8') as out:
            for i, chunk in enumerate(chunks):
                print(f"⏳ Processing Shard {i+1}/{len(chunks)}...", end="", flush=True)
                start_t = time.perf_counter(); refined = refine_log_chunk(chunk); cost_t = time.perf_counter() - start_t
                if "NO VALUABLE DATA" not in refined.upper() and "ERROR:" not in refined:
                    out.write(f"\n### Shard {i+1} Refined Logic\n" + refined + "\n")
                    out.flush(); os.fsync(out.fileno())
                print(f" ✅ Done in {cost_t:.1f}s"); time.sleep(1.5)
        print(f"\n✨ [DONE] Your refined 'Gold Corpus' is safely stored.")

    elif mode == "2":
        # 🛡️ 核心对齐：物理检查截图目录是否存在
        if not os.path.exists(MEMOIR_IMG_DIR) or not os.listdir(MEMOIR_IMG_DIR):
            print(f"🚨 致命错误: 请确保 '{MEMOIR_IMG_DIR}' 文件夹中存有截图！"); exit(1)
            
        # 💡 幂等性防御：检测定稿 MD 是否已存在，防止误触导致的数据覆灭
        if os.path.exists(MEMOIR_OUTPUT):
            print(f"📁 [SAFE-GUARD]: '{MEMOIR_OUTPUT}' already exists.")
            sub_choice = input("Select Action: [A] Append/Resume OCR  [S] Only Re-extract Soul  [R] Full Reset: ").upper()
            if sub_choice == 'A': 
                memoir_path = ocr_memoir_task() # 继承了断点续传逻辑
            elif sub_choice == 'S': 
                memoir_path = MEMOIR_OUTPUT # 直接进入灵魂提取
            else: 
                # 物理重置：彻底擦除现有 MD 并重启 OCR
                confirm = input("⚠️ WARNING: This will DELETE existing memoirs. Confirm? (y/n): ")
                if confirm.lower() == 'y':
                    open(MEMOIR_OUTPUT, 'w', encoding='utf-8').close()
                    memoir_path = ocr_memoir_task()
                else: exit()
        else:
            print("🚀 [BIO-PROFILER] Starting Qwen3.6-Plus Optical Transmission...")
            memoir_path = ocr_memoir_task()

        # 无论哪种路径，最终执行灵魂热挂载
        inject_soul_task(memoir_path)
        print(f"✨ [DONE] Digital Twin 'DANKUROI_PRIME' calibrated.")

    elif mode == "3":
        # 🛡️ 核心扩展：非结构化日常数据提纯
        raw_files = [f for f in os.listdir(THOUGHT_DIR) if f.endswith(('.txt', '.md'))]
        if not raw_files: print(f"🚨 请先将个人随笔/笔记放入 '{THOUGHT_DIR}' 文件夹！"); exit(1)
        print(f"🚀 [THOUGHT-REFINERY] Found {len(raw_files)} entries. Starting distillation...")
        with open(THOUGHT_OUTPUT, 'a', encoding='utf-8') as out:
            for file_name in raw_files:
                file_path = os.path.join(THOUGHT_DIR, file_name)
                with open(file_path, 'r', encoding='utf-8') as rf:
                    content = rf.read()
                    if len(content.strip()) < 10: continue
                    print(f"⏳ Distilling style from: {file_name}...", end="", flush=True)
                    start_t = time.perf_counter()
                    refined = refine_thought_chunk(content)
                    out.write(f"\n\n### Source Entity: {file_name}\n{refined}\n")
                    out.flush(); os.fsync(out.fileno()) # 💡 关键物理锁，防崩溃
                    print(f" ✅ Done ({time.perf_counter()-start_t:.1f}s)")
        print(f"\n✨ [DONE] Your daily cognitive fingerprint saved to {THOUGHT_OUTPUT}")
    elif mode == "4":
        extract_lore_to_persona_task()
    elif mode == "5":
        distill_novel_memory_task()
    elif mode == "6":
        ocr_guidebook_to_rag_task()
    elif mode == "7":
        print("\n🌌 [UNIVERSE FORGE] 启动跨文档人格锻造引擎...")
        universe_id = input("👉 请输入世界观 ID (如 ARKNIGHTS): ").strip().upper()
        char_id = input("👉 请输入目标角色 ID (如 KROOS): ").strip().upper()
        
        # 1. 挂载世界观公有库
        universe_dir = os.path.join(RAG_DIR, "character_memories", universe_id)
        # 2. 挂载角色私有档案 (假设你把官方档案命名为 KROOS_Profile.md 放在对应文件夹)
        char_dir = os.path.join(RAG_DIR, "character_memories", char_id)
        
        context_payload = ""
        print(f"🔍 正在扫描 {universe_id} 与 {char_id} 的高维记忆晶体...")
        
        # 物理贯穿：读取世界观
        if os.path.exists(universe_dir):
            for f in os.listdir(universe_dir):
                if f.endswith(".md"):
                    with open(os.path.join(universe_dir, f), 'r', encoding='utf-8') as md:
                        # 💡 截取前 15000 字符防止 Token 溢出，确保核心设定被加载
                        context_payload += f"\n--- [UNIVERSE LORE: {f}] ---\n{md.read()[:15000]}\n"
        
        # 物理贯穿：读取角色私有档案
        if os.path.exists(char_dir):
            for f in os.listdir(char_dir):
                if f.endswith(".md"):
                    with open(os.path.join(char_dir, f), 'r', encoding='utf-8') as md:
                        context_payload += f"\n---[CHARACTER PROFILE: {f}] ---\n{md.read()}\n"
                        
        if not context_payload.strip():
            print("🚨 未找到任何相关世界观或角色文献！")
            exit()
            
        print(f"🚀 [REASONER] 已注满 {len(context_payload)} 字符的设定张量。启动人格锻造...")
        
        prompt = f"""[Universe-to-Persona Forge Task - V30 Dynamic Architecture]
        You are a Master Character Architect and L10n Skill Distiller. 
        Read the Universe Lore ({universe_id}) and Character Profile ('{char_id}').
        
        Context Payload:
        {context_payload}
        
        Task: Create a V30 Dynamic Persona JSON block. This is NOT just a chat persona; it is a "Translation Skill Filter".
        CRITICAL V30 STRUCTURE:
        1. 'core_identity': 
           - 'desc' MUST be a nested dict ('en', 'ja', 'zh') outlining their psychology and worldview. CRITICAL: Acknowledge their specific TEMPORAL PHASE (e.g., Early naive vs. Late traumatized) based on the entity's ID and profile.
           - 'voice_texture' describes their acoustic timbre.
        2. 'translation_skills' (MACRO-MICRO LORE FUSION): 
           - 'lexical_mapping': You MUST extract Macro-World Terminology from {universe_id} (e.g., Factions, Catastrophes, Magic Systems, Philosophical Concepts) AND the character's micro-habits. Formulate a dictionary mapping common words to their Universe-specific equivalents.
           - 'syntactic_rules': Define strict grammar/sentence-ending habits based on their lore.
        3. 'dynamic_states': MUST include "STATE_CASUAL" (default relaxed/normal/observational) and "STATE_COMBAT" (high tension, anger, ideological clash).
           - 'tts_override': MUST contain 'vol_scale', 'rate', 'pitch' as floats. Adjust these based on the state's emotional intensity!
        
        Output ONLY valid JSON matching this exact structure:
        {{
          "{char_id}": {{
            "core_identity": {{
              "desc": {{"en": "...", "ja": "...", "zh": "..."}},
              "voice_texture": "..."
            }},
            "translation_skills": {{
              "lexical_mapping": {{"hello": ["...", "..."]}},
              "syntactic_rules": "..."
            }},
            "dynamic_states": {{
              "STATE_CASUAL": {{
                "trigger_condition": "Normal conversational context",
                "tts_override": {{"vol_scale": 1.0, "rate": 1.0, "pitch": 1.0}},
                "pragmatics": "..."
              }},
              "STATE_COMBAT": {{
                "trigger_condition": "High stress, combat, or anger",
                "tts_override": {{"vol_scale": 1.2, "rate": 1.1, "pitch": 1.05}},
                "pragmatics": "..."
              }}
            }}
          }}
        }}"""

        try:
            ds_res = client.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": prompt}])
            raw_json = ds_res.choices[0].message.content
            match = re.search(r'\{.*\}', raw_json, re.DOTALL)
            persona_data = json.loads(match.group()) if match else json.loads(raw_json)
            
            out_path = os.path.join(BASE_DIR, f"v25_persona_{char_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(persona_data, f, ensure_ascii=False, indent=2)
            print(f"✨ [ASSET FORGED] {char_id} 的干员精神矩阵已落盘: {out_path}。")
        except Exception as e:
            print(f"🚨 锻造失败: {e}")

    elif mode == "8":
        acoustic_semantic_forge_task()