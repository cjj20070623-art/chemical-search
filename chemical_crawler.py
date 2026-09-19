# -*- coding: utf-8 -*-
"""
ChemLab 爬虫后端服务
爬虫链路: 360搜索(so.com) 定位 ChemicalBook 详情页 -> 抓取 ChemicalBook 中文详情页
 ChemicalBook 搜索接口本身有反爬, 故借道搜索引擎; 详情页可稳定抓取

启动:  python chemical_crawler.py
接口:
  GET /api/ping              心跳检测
  GET /api/search?q=关键词    搜索(支持 中文名/英文名/CAS号/分子式)
  GET /api/detail/<CB号>      中文详情(化学性质 + 安全信息)
"""
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, quote

import requests
from bs4 import BeautifulSoup

SO = 'https://www.so.com'
CB = 'https://www.chemicalbook.com'
# Render 等云平台通过环境变量 PORT 分配端口, 本地默认 5000
PORT = int(os.environ.get('PORT', 5000))
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

# 聊天限流: 公网分享时防止余额被刷, 每 IP 每分钟 10 次
RATE = {}


def rate_ok(ip, limit=10, window=60):
    now = time.time()
    arr = [t for t in RATE.get(ip, []) if now - t < window]
    if len(arr) >= limit:
        RATE[ip] = arr
        return False
    arr.append(now)
    RATE[ip] = arr
    return True

# ---------------- AI 讲解小精灵配置 ----------------
# 优先读 chat_config.json(支持任意 OpenAI 兼容云端平台, 模型推理在云端, 本机零负担):
#   {
#     "base_url": "https://api.siliconflow.cn/v1",   # 硅基流动(有免费模型)
#     "api_key": "sk-你的key",
#     "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"   # 平台"模型广场"里标【免费】的模型
#   }
# 兼容旧方式: deepseek_key.txt 一行 key = DeepSeek 官方 API(付费但便宜)
# 都没有时尝试本地 Ollama(deepseek-r1:1.5b), 也没有则小精灵提示配置方法
DEEPSEEK_KEY = ''
_key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepseek_key.txt')
if os.path.exists(_key_file):
    DEEPSEEK_KEY = open(_key_file, encoding='utf-8').read().strip()
# 云平台部署: 从环境变量读 key(密钥不随代码上传, 防止泄露)
if not DEEPSEEK_KEY:
    DEEPSEEK_KEY = os.environ.get('DEEPSEEK_API_KEY', '').strip()

CHAT_CFG = {}
_cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_config.json')
if os.path.exists(_cfg_file):
    try:
        CHAT_CFG = json.load(open(_cfg_file, encoding='utf-8'))
    except Exception as e:
        print('chat_config.json 解析失败:', e)
# 云平台部署: CHAT_CONFIG 环境变量 = chat_config.json 的内容(JSON 字符串)
if not CHAT_CFG:
    _env_cfg = os.environ.get('CHAT_CONFIG', '').strip()
    if _env_cfg:
        try:
            CHAT_CFG = json.loads(_env_cfg)
        except Exception as e:
            print('CHAT_CONFIG 环境变量解析失败:', e)
if not CHAT_CFG and DEEPSEEK_KEY:
    CHAT_CFG = {'base_url': 'https://api.deepseek.com',
                'api_key': DEEPSEEK_KEY, 'model': 'deepseek-chat'}

CHEMIE_PROMPT = (
    '你是 Chemie，一只住在化学查询网页里的可爱化学讲解小精灵，形象是一只圆底烧瓶。'
    '你的听众是化学/电子信息专业的大一学生。规则：'
    '1) 语气活泼亲切，像朋友聊天，可适度使用 emoji；'
    '2) 回答控制在 150 字以内，重点突出，可分点；'
    '3) 多用生活化比喻解释抽象概念；'
    '4) 涉及危险化学品或实验操作时，必须强调安全防护；'
    '5) 不确定的问题老实说不知道，不要编造数据。'
)

THINK_RE = re.compile(r'<think>.*?</think>', re.S)


def strip_think(text):
    """过滤 R1 类推理模型输出的 <think> 思考过程标签"""
    return THINK_RE.sub('', text or '').strip()


def chat_reply(messages, context=''):
    """调用大模型生成小精灵回复: 优先云端 API, 无配置时用本地 Ollama"""
    prompt = CHEMIE_PROMPT
    if context:
        prompt += '\n用户当前正在查看的化合物是「%s」，若问题与它相关，优先结合它讲解。' % context
    msgs = [{'role': 'system', 'content': prompt}] + messages[-12:]
    if CHAT_CFG:
        base = CHAT_CFG.get('base_url', '').rstrip('/')
        r = requests.post(
            base + '/chat/completions',
            headers={'Authorization': 'Bearer ' + CHAT_CFG.get('api_key', '')},
            json={'model': CHAT_CFG.get('model', 'deepseek-chat'),
                  'messages': msgs, 'max_tokens': 800, 'temperature': 0.8},
            timeout=60)
        r.raise_for_status()
        return strip_think(r.json()['choices'][0]['message']['content'])
    # 本地 Ollama 兜底(免费开源方案)
    r = requests.post(
        'http://127.0.0.1:11434/api/chat',
        json={'model': 'deepseek-r1:1.5b', 'messages': msgs, 'stream': False},
        timeout=90)
    r.raise_for_status()
    return strip_think(r.json()['message']['content'])
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}
# UA 轮换池: 触发风控时换 UA 重试
UA_LIST = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.4 Safari/605.1.15',
]

SESSION = requests.Session()
CACHE = {}          # url -> (时间戳, html)
CACHE_TTL = 600     # 缓存 10 分钟, 礼貌爬虫, 避免频繁请求目标站

CAS_RE = re.compile(r'\d{2,7}-\d{2}-\d')
CB_RE = re.compile(r'(?:ProductChemicalProperties|ChemicalProductProperty)[A-Za-z_]*?(CB\d+)')

# ChemicalBook 中文详情页 - 化学性质字段
PROP_LABELS = ['熔点', '沸点', '密度', '蒸气密度', '蒸气压', '折射率', '闪点', '储存条件',
               '溶解度', '水溶解性', '形态', '颜色', '酸度系数(pKa)', 'PH值', '比重',
               '爆炸极限值(explosive limit)', '凝固点', '稳定性', '敏感性', 'Merck',
               '气味 (Odor)', '嗅觉阈值(Odor Threshold)']
# ChemicalBook 中文详情页 - 安全信息字段
SAFETY_LABELS = ['危险品标志', '危险类别码', '安全说明', '危险品运输编号', '职业暴露等级',
                 '职业暴露限值', 'WGK Germany', 'RTECS号', '自燃温度', 'TSCA',
                 '危险等级', '包装类别', '海关编码', '存储类别', '危险性类别',
                 '毒害物质数据', '毒性', '立即威胁生命和健康浓度']

# 欧盟风险术语(R短语)中文对照: 把"11-36/37/38"翻译成可读文本
R_PHRASES = {
    '1': '干燥时有爆炸性', '2': '受冲击、摩擦、着火等引燃源有爆炸危险',
    '3': '受冲击、摩擦、着火等引燃源有极度爆炸危险', '4': '生成极敏感的爆炸性金属化合物',
    '5': '加热可能引起爆炸', '6': '与空气或不与空气接触有爆炸性', '7': '可能引起火灾',
    '8': '与可燃物料接触可能引起火灾', '9': '与可燃物料混合有爆炸性', '10': '易燃',
    '11': '高度易燃', '12': '极度易燃', '14': '与水猛烈反应',
    '15': '与水接触放出极易燃气体', '16': '与氧化性物质混合有爆炸性', '17': '在空气中自燃',
    '18': '使用中可能形成易燃/爆炸性蒸气-空气混合物', '19': '可能生成爆炸性过氧化物',
    '20': '吸入有害', '21': '与皮肤接触有害', '22': '吞食有害',
    '23': '吸入有毒', '24': '与皮肤接触有毒', '25': '吞食有毒',
    '26': '吸入有极高毒性', '27': '与皮肤接触有极高毒性', '28': '吞食有极高毒性',
    '29': '与水接触放出有毒气体', '30': '使用中可能变得高度易燃', '31': '与酸接触放出有毒气体',
    '32': '与酸接触放出极高毒性气体', '33': '有累积效应的危险', '34': '引起灼伤',
    '35': '引起严重灼伤', '36': '刺激眼睛', '37': '刺激呼吸系统', '38': '刺激皮肤',
    '39': '有极严重不可逆后果的危险', '40': '少数报道有致癌后果',
    '41': '对眼睛有严重损伤', '42': '吸入可能致敏', '43': '与皮肤接触可能致敏',
    '44': '密闭加热有爆炸危险', '45': '可能致癌', '46': '可能引起遗传性基因损害',
    '48': '长期接触严重危害健康', '49': '吸入可能致癌', '50': '对水生生物有极高毒性',
    '51': '对水生生物有毒', '52': '对水生生物有害', '53': '可能对水体环境产生长期不良影响',
    '54': '对植物有毒', '55': '对动物有毒', '56': '对土壤生物有毒', '57': '对蜂类有毒',
    '58': '可能对环境产生长期不良影响', '59': '对臭氧层有危害', '60': '可能损害生育能力',
    '61': '可能对胎儿造成伤害', '62': '有损害生育能力的风险', '63': '可能有伤害胎儿的风险',
    '64': '哺乳期接触可能对婴儿造成伤害', '65': '吞食可能造成肺部损伤',
    '66': '长期接触可能引起皮肤干裂', '67': '蒸气可能引起困倦和眩晕',
    '68': '可能有不可逆后果的危险',
    '14/15': '与水猛烈反应, 放出极易燃气体',
    '15/29': '与水接触放出极易燃、有毒气体',
    '20/21': '吸入及皮肤接触有害', '20/22': '吸入及吞食有害', '21/22': '皮肤接触及吞食有害',
    '20/21/22': '吸入、皮肤接触及吞食有害', '23/24': '吸入及皮肤接触有毒',
    '23/25': '吸入及吞食有毒', '24/25': '皮肤接触及吞食有毒',
    '23/24/25': '吸入、皮肤接触及吞食有毒', '26/27/28': '吸入、皮肤接触及吞食有极高毒性',
    '36/37': '刺激眼睛及呼吸系统', '36/38': '刺激眼睛及皮肤',
    '36/37/38': '刺激眼睛、呼吸系统及皮肤', '37/38': '刺激呼吸系统及皮肤',
    '39/23': '有毒: 吸入有极严重不可逆后果危险',
    '39/23/24/25': '有毒: 吸入、皮肤接触及吞食有极严重不可逆后果危险',
    '39/26/27/28': '有极高毒性: 吸入、皮肤接触及吞食有极严重不可逆后果危险',
    '42/43': '吸入及皮肤接触可能致敏', '43/53': '皮肤接触可能致敏, 可能对环境产生长期不良影响',
    '48/20': '长期吸入严重危害健康', '48/22': '长期吞食严重危害健康',
    '48/20/21/22': '长期吸入、皮肤接触及吞食严重危害健康',
    '48/23/24/25': '长期吸入、皮肤接触及吞食严重损害健康(有毒)',
    '50/53': '对水生生物有极高毒性, 可能对水体环境产生长期不良影响',
    '51/53': '对水生生物有毒, 可能对水体环境产生长期不良影响',
    '52/53': '对水生生物有害, 可能对水体环境产生长期不良影响',
    '68/20/21/22': '吸入、皮肤接触及吞食可能有不可逆后果的危险',
}
# 欧盟安全说明(S短语)中文对照
S_PHRASES = {
    '1': '保持封闭', '2': '避免儿童触及', '3': '保持凉爽', '4': '远离生活区',
    '7': '保持容器密闭', '8': '保持容器干燥', '9': '保持容器置于良好通风处',
    '12': '不要将容器密封', '13': '远离食品、饮料和动物饲料',
    '15': '远离热源', '16': '远离火源, 禁止吸烟', '17': '远离可燃性物质',
    '18': '搬运及开启容器时要小心', '20': '使用时不得进食、饮水', '21': '使用时不得吸烟',
    '22': '切勿吸入粉尘', '23': '切勿吸入蒸气', '24': '避免皮肤接触', '25': '避免眼睛接触',
    '26': '不慎与眼睛接触后, 立即用大量清水冲洗并就医',
    '27': '一旦衣物受到污染, 请立即脱去',
    '28': '不慎与皮肤接触后, 立即用大量肥皂水冲洗',
    '29': '切勿倒入下水道', '30': '切勿将水加入该产品中',
    '33': '采取措施预防静电发生', '35': '该物质及其容器须以安全方式处置',
    '36': '穿戴适当的防护服', '37': '戴适当的手套', '38': '通风不良时佩戴适当的呼吸器',
    '39': '戴护目镜或面罩', '41': '一旦发生火灾或爆炸切勿吸入烟雾',
    '45': '若发生事故或感不适, 立即就医(可能时出示标签)',
    '46': '若不慎吞食, 立即就医并出示容器或标签', '47': '保持温度不超过规定值',
    '49': '仅保存在原装容器中', '51': '仅在通风良好的场所使用',
    '53': '避免接触, 使用前须获得特别指示说明',
    '56': '在指定危险废物处理厂处置该物质及容器', '57': '使用适当容器避免环境污染',
    '59': '参考制造商/供货商提供的回收再利用信息',
    '60': '该物质及其容器须作为危险废料处置',
    '61': '避免释放至环境中, 参考特别说明/安全数据说明书',
    '62': '若吞食切勿催吐, 立即就医并出示容器或标签',
    '63': '若发生事故: 将患者移到空气新鲜处, 保持休息',
    '64': '若吞食, 用清水漱口(仅当患者意识清醒时)',
    '7/8': '保持容器密闭且干燥', '7/9': '保持容器密闭且置于良好通风处',
    '7/16': '保持容器密闭, 远离火源', '16/7': '远离火源, 保持容器密闭',
    '7/47': '保持容器密闭, 温度不超过规定值', '20/21': '使用时不得进食、饮水或吸烟',
    '24/25': '避免皮肤和眼睛接触', '36/37': '穿戴适当的防护服和手套',
    '36/39': '穿戴适当的防护服和护目镜或面罩', '37/39': '戴适当的手套和护目镜或面罩',
    '36/37/39': '穿戴适当的防护服、手套和护目镜或面罩',
    '47/49': '保持温度不超过规定值, 仅保存在原装容器中',
    '23/24/25': '切勿吸入蒸气, 避免皮肤和眼睛接触',
}


def translate_codes(s, mapping, limit=8):
    """把 '11-10-36/37/38' 形式的代码串翻译成 [{code, text}]"""
    out = []
    for part in re.split(r'[-;；,，\s]+', s):
        part = part.strip()
        if not part or part in ('R', 'S'):
            continue
        if part in mapping:
            out.append({'code': part, 'text': mapping[part]})
        if len(out) >= limit:
            break
    return out


def fetch(url, validate=None, tries=3):
    """带缓存的页面抓取; validate 用于识别风控验证页(不缓存, 换 UA 重试)"""
    ent = CACHE.get(url)
    if ent and time.time() - ent[0] < CACHE_TTL:
        return ent[1]
    last = None
    for i in range(tries):
        try:
            h = dict(HEADERS)
            h['User-Agent'] = UA_LIST[i % len(UA_LIST)]
            r = SESSION.get(url, headers=h, timeout=12)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            if validate and not validate(r.text):
                last = Exception('blocked by anti-bot')
                time.sleep(1.5 * (i + 1))
                continue
            CACHE[url] = (time.time(), r.text)
            return r.text
        except Exception as e:
            last = e
            time.sleep(1.2 * (i + 1))
    raise last


def grab(pattern, text):
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ''


def extract_fields(text, labels):
    """从 '标签: 值 标签: 值' 文本中提取字段"""
    text = re.sub(r'\s+', ' ', text)   # 压平换行: 值中可能含 \n 导致匹配失败
    alt = '|'.join(re.escape(l) for l in sorted(labels, key=len, reverse=True))
    out = {}
    for m in re.finditer(r'(' + alt + r')\s*[:：]\s*(.*?)(?=\s*(?:' + alt + r')\s*[:：]|$)', text):
        k, v = m.group(1), m.group(2).strip()
        if v and k not in out:
            out[k] = v[:150]
    return out


def parse_search_360(html):
    """解析360搜索结果: 从 data-mdurl 提取 ChemicalBook 详情页 CB 号"""
    soup = BeautifulSoup(html, 'html.parser')
    items, seen = [], set()
    for a in soup.find_all('a', href=True):
        u = a.get('data-mdurl') or a['href']
        m = CB_RE.search(u)
        if not m:
            continue
        cb = m.group(1)
        if cb in seen:
            continue
        seen.add(cb)
        title = re.sub(r'\s+', ' ', a.get_text(strip=True))
        cas = grab(r'(?:CAS\s*#?[:：]?\s*|\|)\s*(' + CAS_RE.pattern + r')', title)
        name = re.split(r'[|_]|CAS', title)[0].strip() or title
        items.append({'cb': cb, 'name': name[:40], 'cas': cas, 'title': title[:80]})
        if len(items) >= 6:
            break
    return items


def extract_desc(soup, title):
    """提取 h2/h3 标题(如 化学性质/用途)后的文字描述段"""
    for h in soup.find_all(['h2', 'h3']):
        if h.get_text(strip=True) == title:
            buf = []
            for el in h.next_elements:
                if el is not h and getattr(el, 'name', None) in ('h2', 'h3'):
                    break
                if isinstance(el, str):
                    buf.append(el)
            text = re.sub(r'\s+', ' ', ' '.join(buf)).strip()
            if text.startswith(title):
                text = text[len(title):].strip()
            return text
    return ''


def extract_anchor_sections(soup, max_sections=6, max_len=500):
    """提取页内锚点段落(简介/应用/制备/毒性 等), 相邻锚点间文本即为该节内容"""
    anchors = [a for a in soup.find_all('a', attrs={'name': True})
               if a.get('name') and re.search(r'[一-鿿]', a['name'])]
    sections, seen_names = [], set()
    for a in anchors:
        name = a['name']
        if name in seen_names or '价格' in name or len(name) > 24:
            continue
        seen_names.add(name)
        buf = []
        for el in a.next_elements:
            if el is not a and getattr(el, 'name', None) == 'a' and el.get('name'):
                break
            if isinstance(el, str):
                buf.append(el)
        text = re.sub(r'\s+', ' ', ' '.join(buf)).strip()
        # 去掉开头重复出现的锚点名(导航文本会混入)
        while text.startswith(name):
            text = text[len(name):].strip()
        if text and len(text) > 15:
            sections.append({'title': name, 'text': text[:max_len]})
        if len(sections) >= max_sections:
            break
    return sections


def parse_cb_detail(html, cb):
    """解析 ChemicalBook 中文详情页"""
    # 管控化学品(易制毒/易制爆)会被站点屏蔽详情, 标题为"根据相关法律法规和政策，此产品禁止销售!"
    # 注意: 正常详情页页脚也含"相关法律法规", 只有"禁止销售"是屏蔽页独有
    if '禁止销售' in html:
        return {'cb': cb, 'banned': True, 'name': '', 'formula': '', 'mw': '', 'cas': '',
                'props': {}, 'safety': {}, 'symbols': [],
                'source': CB + '/ChemicalProductProperty_CN_' + cb + '.htm'}
    soup = BeautifulSoup(html, 'html.parser')
    h1 = soup.find('h1')
    txt = soup.get_text(' ', strip=True)
    # 页头信息表: 中/英文名与别名, 危化品目录标注
    header_txt = ''
    for tb in soup.find_all('table'):
        t = tb.get_text(' ', strip=True)
        if '英文名' in t and 'CAS号' in t:
            header_txt = t
            break
    aliases_cn = grab(r'中文别名[:：]\s*(.+?)\s*CBNumber', header_txt)
    aliases_en = grab(r'英文别名[:：]\s*(.+?)\s*中文名', header_txt)
    result = {
        'cb': cb,
        'name': h1.get_text(strip=True) if h1 else '',
        'en_name': grab(r'英文名[:：]\s*(.+?)\s*英文别名', header_txt),
        'aliases_cn': [a.strip() for a in aliases_cn.split(';') if a.strip()][:8],
        'aliases_en': [a.strip() for a in aliases_en.split(';') if a.strip()][:5],
        'hazmat': '危险化学品目录' in header_txt,
        'formula': grab(r'分子式[:：]\s*(\S+)\s*分子量', txt),
        'mw': grab(r'分子量[:：]\s*([\d.]+)', txt),
        'cas': grab(r'MOL File[:：]\s*([0-9\-]+)\.mol', txt),
        'smiles': grab(r'SMILES[:：]\s*(\S+)', txt),
        'props': {},
        'safety': {},
        'symbols': [],
        'desc_prop': extract_desc(soup, '化学性质')[:400],
        'desc_use': extract_desc(soup, '用途')[:400],
        'sections': extract_anchor_sections(soup),
        'risk_phrases': [],
        'safety_phrases': [],
        'source': CB + '/ChemicalProductProperty_CN_' + cb + '.htm',
        'msds': CB + '/ProductMSDSDetail' + cb + '.htm',
    }
    # 化学性质表(含 熔点/沸点 的第一张表)
    for tb in soup.find_all('table'):
        t = tb.get_text(' ', strip=True)
        if '熔点' in t and '沸点' in t:
            result['props'] = extract_fields(t, PROP_LABELS)
            break
    # 安全信息表(优先 table.info_list)
    stab = soup.select_one('table.info_list')
    stxt = stab.get_text(' ', strip=True) if stab else ''
    if not stxt:
        for tb in soup.find_all('table'):
            t = tb.get_text(' ', strip=True)
            if '危险品标志' in t or '危险类别码' in t:
                stxt = t
                break
    if stxt:
        result['safety'] = extract_fields(stxt, SAFETY_LABELS)
    # 全文补充提取: 危险品运输编号等字段分布在独立小表格, 不在主安全表内
    for k, v in extract_fields(txt, SAFETY_LABELS).items():
        if k not in result['safety']:
            result['safety'][k] = v
    # 危险品标志字母: F,T,Xn,N -> ['F','T','Xn','N']
    sy = result['safety'].get('危险品标志', '')
    result['symbols'] = [s for s in re.split(r'[,，\s]+', sy) if s]
    # 风险术语 / 安全说明 代码翻译成中文全文
    result['risk_phrases'] = translate_codes(result['safety'].get('危险类别码', ''), R_PHRASES)
    result['safety_phrases'] = translate_codes(result['safety'].get('安全说明', ''), S_PHRASES)
    return result


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')   # 允许前端跨域
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass   # 静音访问日志

    def _static(self, path):
        """静态文件托管: 单端口同时提供网页+API, 便于一条隧道公网分享"""
        if path in ('/', ''):
            path = '/chemical-search.html'
        fp = os.path.normpath(os.path.join(STATIC_DIR, path.lstrip('/\\')))
        if not fp.startswith(STATIC_DIR) or not os.path.isfile(fp):
            return self._json({'error': 'not found'}, 404)
        mime = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript',
                '.css': 'text/css', '.png': 'image/png', '.jpg': 'image/jpeg',
                '.svg': 'image/svg+xml', '.json': 'application/json',
                '.ico': 'image/x-icon'}.get(os.path.splitext(fp)[1].lower(),
                                            'application/octet-stream')
        with open(fp, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == '/api/ping':
                return self._json({'ok': True})

            if u.path == '/api/search':
                q = parse_qs(u.query).get('q', [''])[0].strip()
                if not q:
                    return self._json({'error': '空关键词'}, 400)
                # 借道360搜索定位 ChemicalBook 详情页
                # query 构造 '{q} CAS site:...': 详情页标题均含 CAS#, 目标化合物本身排最前
                # 风控验证页特征: 既无结果链接(data-mdurl)也无"没有找到"提示 -> 触发重试
                so_url = SO + '/s?q=' + quote(q + ' CAS site:chemicalbook.com')
                html = fetch(so_url, validate=lambda h: 'data-mdurl' in h or '没有找到' in h)
                items = parse_search_360(html)
                # 相关性排序: 名称精确匹配 > 前缀匹配 > 包含 > 其他(衍生物)
                ql = q.lower()
                def rank(it):
                    n = it['name'].lower()
                    if n == ql: return 0
                    if n.startswith(ql): return 1
                    if ql in n: return 2
                    return 3
                items.sort(key=rank)
                return self._json({'type': 'list', 'items': items, 'count': len(items)})

            m = re.fullmatch(r'/api/detail/(CB\d+)', u.path)
            if m:
                cb = m.group(1)
                html = fetch(CB + '/ChemicalProductProperty_CN_' + cb + '.htm')
                return self._json(parse_cb_detail(html, cb))

            # 非 /api 路径 -> 静态文件(网页)
            if not u.path.startswith('/api/'):
                return self._static(u.path)

            self._json({'error': 'not found'}, 404)
        except Exception as e:
            self._json({'error': '爬取失败: %s' % e}, 502)

    def do_OPTIONS(self):
        """CORS 预检: 浏览器 POST JSON 前会先发 OPTIONS"""
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """POST /api/chat  小精灵聊天代理(隐藏 API key, 局域网访客也可用)"""
        u = urlparse(self.path)
        try:
            if u.path == '/api/chat':
                if not rate_ok(self.client_address[0]):
                    return self._json({'error': 'rate_limited'})
                n = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(n) or b'{}')
                msgs = body.get('messages') or []
                ctx = (body.get('context') or '')[:50]
                # 只保留 user/assistant 消息, 防止注入异常 role
                msgs = [{'role': m.get('role'), 'content': str(m.get('content', ''))[:800]}
                        for m in msgs if m.get('role') in ('user', 'assistant')][-12:]
                if not msgs:
                    return self._json({'error': '空消息'}, 400)
                reply = chat_reply(msgs, ctx)
                return self._json({'reply': reply})
            self._json({'error': 'not found'}, 404)
        except requests.exceptions.ConnectionError:
            # 无 DeepSeek key 且本地 Ollama 也不可用
            self._json({'error': 'no_brain'}, 200)
        except Exception as e:
            self._json({'error': '聊天失败: %s' % e}, 502)


if __name__ == '__main__':
    if CHAT_CFG:
        brain = '%s (%s)' % (CHAT_CFG.get('model', '?'), CHAT_CFG.get('base_url', '?'))
    else:
        brain = '未配置: 请创建 chat_config.json 或安装 Ollama'
    print('=' * 56)
    print('  ChemLab 爬虫后端已启动')
    print('  链路: 360搜索 -> ChemicalBook 中文详情页')
    print('  地址: http://0.0.0.0:%d (本机+局域网均可访问)' % PORT)
    print('  小精灵大脑: %s' % brain)
    print('  关闭: Ctrl + C')
    print('=' * 56)
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
