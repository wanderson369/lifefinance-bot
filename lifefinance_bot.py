# -*- coding: utf-8 -*-
"""
LifeFinance Tournament Bot v2.0
================================
Estrategia com Analise MultiTimeframe (MTF)

PROBABILIDADES:
  1 TF confluente          = 80%
  Forca tendencia 1 TF     = 85%
  M5+M30 ou M5+H1 confluentes = 90%
  M15+H4 confluentes       = 90%
  TODOS os TFs confluentes  = 95%

INDICADORES:
  EMA 9, EMA 21, SMA 50
  RSI 10
  Stoch 6,2,2 | 20,10,20 | 50,10,50
  MACD 20,36,10

SINAL ENVIADO:
  Par | Direcao | Forca | Probabilidade | Horario
"""

import os, time, requests, threading
from datetime import datetime, timezone, timedelta
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler

BRT = timezone(timedelta(hours=-3))
def agora_brt(): return datetime.now(BRT).strftime("%d/%m %H:%M")
def converter_hora(dt_str):
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).astimezone(BRT).strftime("%d/%m %H:%M")
    except: return dt_str

# ============================================================
# CONFIGURACOES
# ============================================================
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN_LF", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TWELVE_API_KEY   = os.environ.get("TWELVE_API_KEY", "")

TODOS_PARES = {
    "EUR/USD":"EUR/USD","GBP/USD":"GBP/USD","USD/JPY":"USD/JPY",
    "AUD/USD":"AUD/USD","USD/CHF":"USD/CHF","USD/CAD":"USD/CAD",
    "NZD/USD":"NZD/USD","GBP/CAD":"GBP/CAD","EUR/GBP":"EUR/GBP",
    "EUR/JPY":"EUR/JPY","GBP/JPY":"GBP/JPY","AUD/JPY":"AUD/JPY",
    "EUR/AUD":"EUR/AUD","GBP/AUD":"GBP/AUD","XAU/USD":"XAU/USD",
    "BTC/USDT":"BTC/USDT",
}

# Todos os TFs para analise MTF
TODOS_TFS = ["5min", "15min", "30min", "1h", "4h"]

# Combinacoes MTF que elevam para 90%
COMBOS_90 = [
    ["5min", "30min"],
    ["5min", "1h"],
    ["15min", "4h"],
]

CONFIG = {
    "pausado":        False,
    "pares_ativos":   list(TODOS_PARES.keys()),
    "prob_minima":    80,
    "filtro_pares":   [],
    "filtro_direcao": "",
    "filtro_prob":    80,
}

INTERVALOS_SCAN = {"5min":300,"15min":900,"30min":1800,"1h":3600,"4h":14400}

sinais_enviados    = {}
historico_sinais   = deque(maxlen=200)
ultima_verificacao = {}
ultimo_update_id   = 0
inicio             = datetime.now(BRT).strftime("%d/%m/%Y %H:%M")
total_sinais       = 0

# ============================================================
# API
# ============================================================
def buscar_candles(par, tf, qtd=100):
    try:
        r = requests.get("https://api.twelvedata.com/time_series", params={
            "symbol":par, "interval":tf, "outputsize":qtd,
            "apikey":TWELVE_API_KEY, "format":"JSON",
        }, timeout=8)
        data = r.json()
        if data.get("status") == "error": return []
        return [{"open":float(v["open"]), "high":float(v["high"]),
                 "low":float(v["low"]),   "close":float(v["close"]),
                 "datetime":v["datetime"]}
                for v in reversed(data.get("values",[]))]
    except: return []

# ============================================================
# INDICADORES
# ============================================================
def calc_ema(closes, p):
    if len(closes) < p: return []
    k = 2/(p+1); r = [sum(closes[:p])/p]
    for c in closes[p:]: r.append(c*k + r[-1]*(1-k))
    return r

def calc_sma(closes, p):
    if len(closes) < p: return []
    return [sum(closes[i:i+p])/p for i in range(len(closes)-p+1)]

def inclinacao(s, n=3):
    if len(s) < n: return 0
    m = abs(s[-1]) if s[-1] != 0 else 1
    return (s[-1] - s[-n]) / m * 100 * 10

def calc_rsi(closes, p=10):
    if len(closes) < p+1: return []
    d  = [closes[i+1]-closes[i] for i in range(len(closes)-1)]
    g  = [max(x,0) for x in d]
    pe = [abs(min(x,0)) for x in d]
    ag = sum(g[:p])/p; ap = sum(pe[:p])/p; rv = []
    for i in range(p, len(d)):
        ag = (ag*(p-1)+g[i])/p; ap = (ap*(p-1)+pe[i])/p
        rs = ag/ap if ap != 0 else 100
        rv.append(100-(100/(1+rs)))
    return rv

def calc_stoch(candles, kp, dp, sm=1):
    if len(candles) < kp: return [], []
    closes = [c["close"] for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    kv = []
    for i in range(kp-1, len(candles)):
        h = max(highs[i-kp+1:i+1]); l = min(lows[i-kp+1:i+1])
        kv.append((closes[i]-l)/(h-l)*100 if h != l else 50)
    if sm > 1:
        sv = []
        for i in range(sm-1, len(kv)): sv.append(sum(kv[i-sm+1:i+1])/sm)
        kv = sv
    dv = [sum(kv[i-dp+1:i+1])/dp for i in range(dp-1, len(kv))]
    return kv, dv

def calc_macd(closes, fast=20, slow=36, sig=10):
    """MACD configurado em 20,36,10 conforme estrategia"""
    if len(closes) < slow+sig: return [], [], []
    ef = calc_ema(closes, fast); es = calc_ema(closes, slow)
    d  = len(ef)-len(es); ef = ef[d:]
    ml = [f-s for f,s in zip(ef,es)]
    sl = calc_ema(ml, sig); d2 = len(ml)-len(sl); ml = ml[d2:]
    return ml, sl, [m-s for m,s in zip(ml,sl)]

def hist_lat(h, n=3):
    if len(h) < n: return False
    r = h[-n:]; v = max(r)-min(r)
    m = abs(sum(r)/n) if sum(r) != 0 else 0.00001
    return (v/m) < 0.3

def div_alt(closes, ind, n=5):
    if len(closes) < n*2 or len(ind) < n*2: return False
    return (min(closes[-n:]) <= min(closes[-n*2:-n]) and
            min(ind[-n:])    >  min(ind[-n*2:-n]))

def div_bx(closes, ind, n=5):
    if len(closes) < n*2 or len(ind) < n*2: return False
    return (max(closes[-n:]) >= max(closes[-n*2:-n]) and
            max(ind[-n:])    <  max(ind[-n*2:-n]))

# ============================================================
# ANALISE DE UM UNICO TF
# Retorna: direcao ("COMPRA"/"VENDA"/None), forca (bool), atencao (bool), dados
# ============================================================
def analisar_tf(par, tf):
    """
    Analisa um unico TF e retorna o resultado da confluencia dos indicadores.
    Retorna dict com: direcao, forca, atencao, dados dos indicadores
    """
    candles = buscar_candles(par, tf, 100)
    if len(candles) < 60: return None

    closes = [c["close"] for c in candles]
    at     = candles[-1]

    # Medias moveis
    e9s  = calc_ema(closes, 9)
    e21s = calc_ema(closes, 21)
    s50s = calc_sma(closes, 50)
    if not e9s or not e21s or not s50s: return None

    e9  = e9s[-1]; e21 = e21s[-1]; s50 = s50s[-1]
    a9  = inclinacao(e9s, 5); a21 = inclinacao(e21s, 5); a50 = inclinacao(s50s, 5)

    e9c  = a9 > 0.3; e21c = a21 > 0.3; s50c = a50 >  0.5
    e9b  = a9 < -0.3; e21b = a21 < -0.3; s50b = a50 < -0.5

    zm  = min(e9, e21); zx = max(e9, e21)
    pz  = zm*0.999 <= at["close"] <= zx*1.001

    # RSI 10
    rs = calc_rsi(closes, 10)
    if not rs: return None
    ra   = rs[-1]; rant = rs[-2] if len(rs) > 1 else ra
    rang = inclinacao(rs, 3)
    rac  = ra > 50; rab = ra < 50
    ric  = rang > 0.5; rib = rang < -0.5
    rsv  = rant < 30 and ra >= 30
    rsc  = rant > 70 and ra <= 70
    rxc  = ra > 80; rxb = ra < 20   # extremos - atencao

    # Stoch 6,2,2
    sk6, sd6 = calc_stoch(candles, 6, 2, 2)
    if len(sk6) < 2 or len(sd6) < 2: return None
    s6cc = sk6[-2] < sd6[-2] and sk6[-1] > sd6[-1]
    s6cb = sk6[-2] > sd6[-2] and sk6[-1] < sd6[-1]
    s6a  = sk6[-1] < 50; s6b = sk6[-1] > 50

    # Stoch 20,10,20
    sk20, sd20 = calc_stoch(candles, 20, 10, 20)
    if len(sk20) < 3: return None
    ak20 = inclinacao(sk20, 3)
    s20c = ak20 > 0.5; s20b = ak20 < -0.5
    s20cc = sk20[-2] < sd20[-2] and sk20[-1] > sd20[-1]
    s20cb = sk20[-2] > sd20[-2] and sk20[-1] < sd20[-1]

    # Stoch 50,10,50
    sk50, sd50 = calc_stoch(candles, 50, 10, 50)
    if len(sk50) < 3: return None
    ak50  = inclinacao(sk50, 3)
    s50c2 = ak50 > 0.3; s50b2 = ak50 < -0.3

    # MACD 20,36,10
    ml, sl_, hist = calc_macd(closes, 20, 36, 10)
    if len(hist) < 4: return None
    ha   = hist[-1]; hant = hist[-2]; ma = ml[-1]
    hac  = ha > 0; hab = ha < 0
    mac  = ma > 0; mab = ma < 0
    hcc  = hant < 0 and ha > 0; hcb = hant > 0 and ha < 0
    hic  = ha > hant; hib = ha < hant
    hl   = hist_lat(hist, 3)

    # Divergencias
    dac = div_alt(closes, rs);  das = div_alt(closes, sk6)
    dbc = div_bx(closes, rs);   dbs = div_bx(closes, sk6)

    # ============================================================
    # FORCA DA TENDENCIA
    # COMPRA: EMA9/21/SMA50 cima + Stoch20/50 cima + MACD hist cima
    #         + RSI inclinado cima ou acima 50
    # VENDA:  oposto
    # ============================================================
    forca_c = (e9c and e21c and s50c and s20c and s50c2 and hac and (ric or rac))
    forca_v = (e9b and e21b and s50b and s20b and s50b2 and hab and (rib or rab))
    atencao_c = forca_c and rxc   # RSI > 80 = possivel reversao
    atencao_v = forca_v and rxb   # RSI < 20 = possivel reversao

    # ============================================================
    # REGRAS DE CONFLUENCIA
    # Verifica se os indicadores se confluenciam em uma direcao
    # ============================================================
    pontos_c = 0; pontos_v = 0

    # Regra 1: SMA50 + zona EMA + RSI/Stoch + MACD
    if s50c and pz and (rac or s6cc or rsv) and (hcc or hl): pontos_c += 2
    if s50b and pz and (rab or s6cb or rsc) and (hcb or hl): pontos_v += 2

    # Regra 2: Stoch20 + Stoch6 + MACD
    if s20c and s6a and (s6cc or s20cc) and (mac or hac): pontos_c += 2
    if s20b and s6b and (s6cb or s20cb) and (mab or hab): pontos_v += 2

    # Regra 3: Stoch20+50 + MACD histograma
    if s20c and s50c2 and hac and hic and (s6cc or s20cc): pontos_c += 2
    if s20b and s50b2 and hab and hib and (s6cb or s20cb): pontos_v += 2

    # Regra 4: Divergencia
    if (dac or das) and (hac or hcc): pontos_c += 2
    if (dbc or dbs) and (hab or hcb): pontos_v += 2

    # Bonus forca tendencia
    if forca_c: pontos_c += 3
    if forca_v: pontos_v += 3

    # Determinar direcao vencedora (minimo 2 pontos = 1 regra)
    direcao = None
    if pontos_c >= 2 and pontos_c > pontos_v:
        direcao = "COMPRA"
        forca   = forca_c
        atencao = atencao_c
    elif pontos_v >= 2 and pontos_v > pontos_c:
        direcao = "VENDA"
        forca   = forca_v
        atencao = atencao_v
    else:
        return None  # sem confluencia clara

    return {
        "direcao": direcao,
        "forca":   forca,
        "atencao": atencao,
        "pontos":  pontos_c if direcao == "COMPRA" else pontos_v,
        "preco":   at["close"],
        "horario": at["datetime"],
        "ema9":    e9, "ema21": e21, "sma50": s50,
        "rsi":     ra, "stoch6": sk6[-1],
        "hist":    ha,
        "rsi_txt": f"RSI={ra:.0f}{'(>80 ATENCAO!)' if rxc else '(<20 ATENCAO!)' if rxb else ''}",
    }

# ============================================================
# ANALISE MULTITIMEFRAME
# ============================================================
def calcular_probabilidade(tfs_confluentes, forcas, direcao):
    """
    Calcula probabilidade baseada na confluencia MTF:
    1 TF                         = 80%
    1 TF com forca tendencia     = 85%
    M5+M30 ou M5+H1 ou M15+H4   = 90%
    TODOS os 5 TFs               = 95%
    """
    n_tfs = len(tfs_confluentes)
    n_forcas = sum(1 for f in forcas if f)

    if n_tfs == 0: return 0

    # Verifica combo 95% - todos os 5 TFs
    todos = set(TODOS_TFS)
    if todos.issubset(set(tfs_confluentes)):
        return 95

    # Verifica combos 90%
    for combo in COMBOS_90:
        if all(tf in tfs_confluentes for tf in combo):
            return 90

    # 1 TF com forca = 85%
    if n_tfs >= 1 and n_forcas >= 1:
        return 85

    # 1 TF sem forca = 80%
    return 80

def analisar_par_mtf(par):
    """
    Analisa o par em todos os TFs e calcula probabilidade MTF
    """
    resultados = {}

    for tf in TODOS_TFS:
        r = analisar_tf(par, tf)
        if r:
            resultados[tf] = r
        time.sleep(0.3)  # evitar rate limit

    if not resultados: return None

    # Contar TFs por direcao
    compras = {tf: r for tf, r in resultados.items() if r["direcao"] == "COMPRA"}
    vendas  = {tf: r for tf, r in resultados.items() if r["direcao"] == "VENDA"}

    # Direcao vencedora = mais TFs confluentes
    if len(compras) >= len(vendas) and len(compras) > 0:
        tfs_conf = list(compras.keys())
        direcao  = "COMPRA"
        dados_tf = compras
    elif len(vendas) > len(compras):
        tfs_conf = list(vendas.keys())
        direcao  = "VENDA"
        dados_tf = vendas
    else:
        return None  # empate - sem sinal claro

    if len(tfs_conf) == 0: return None

    forcas  = [dados_tf[tf]["forca"]   for tf in tfs_conf]
    atencao = any(dados_tf[tf]["atencao"] for tf in tfs_conf)
    prob    = calcular_probabilidade(tfs_conf, forcas, direcao)
    forca   = any(forcas)

    # Usar dados do TF mais alto confluente para preco/horario
    tf_ref = tfs_conf[-1]
    dados  = dados_tf[tf_ref]

    # Descricao dos TFs confluentes
    tfs_txt = []
    for tf in tfs_conf:
        d = dados_tf[tf]
        fc = "F" if d["forca"] else ""
        tfs_txt.append(f"{tf.upper()}{fc}")

    return {
        "par":        par,
        "direcao":    direcao,
        "prob":       prob,
        "forca":      forca,
        "atencao":    atencao,
        "tfs_conf":   tfs_conf,
        "tfs_txt":    " | ".join(tfs_txt),
        "n_tfs":      len(tfs_conf),
        "preco":      dados["preco"],
        "horario":    dados["horario"],
        "ema9":       dados["ema9"],
        "ema21":      dados["ema21"],
        "sma50":      dados["sma50"],
        "rsi":        dados["rsi"],
        "stoch6":     dados["stoch6"],
        "hist":       dados["hist"],
        "rsi_txt":    dados["rsi_txt"],
    }

# ============================================================
# FORMATACAO DO SINAL
# ============================================================
def barra(p):
    f = int(p/10); return "="*f + "-"*(10-f)

def formatar(s):
    d     = s["direcao"]
    pr    = s["preco"]
    prob  = s["prob"]
    conf  = "MAXIMA" if prob >= 95 else "MUITO ALTA" if prob >= 90 else "ALTA" if prob >= 85 else "BOA"
    forca = "TENDENCIA FORTE" if s["forca"] else "Tendencia moderada"
    at_tx = "ATENCAO: RSI EXTREMO - possivel reversao!" if s["atencao"] else ""

    # SL e TP
    atr = abs(s["ema9"] - s["ema21"]) * 2
    if atr < pr * 0.001: atr = pr * 0.002
    if d == "COMPRA":
        sl  = pr - atr; tp1 = pr + atr*1.5
        tp2 = pr + atr*3; tp3 = pr + atr*5
    else:
        sl  = pr + atr; tp1 = pr - atr*1.5
        tp2 = pr - atr*3; tp3 = pr - atr*5

    # MTF info
    if prob == 95:
        mtf_txt = "TODOS OS TFs CONFLUENTES!"
    elif prob == 90:
        mtf_txt = f"Combo MTF 90%: {s['tfs_txt']}"
    elif prob == 85:
        mtf_txt = f"Forca confirmada: {s['tfs_txt']}"
    else:
        mtf_txt = f"TFs: {s['tfs_txt']}"

    return (
        f"{d} LIFEFINANCE - {s['par']}\n"
        f"-----------------------\n"
        f"Par:      {s['par']}\n"
        f"Direcao:  {d}\n"
        f"Preco:    {pr:.5f}\n"
        f"-----------------------\n"
        f"Prob: {prob}%\n"
        f"[{barra(prob)}] {conf}\n"
        f"{forca}\n"
        f"{at_tx}\n"
        f"-----------------------\n"
        f"MTF ({s['n_tfs']} TFs):\n"
        f"{mtf_txt}\n"
        f"-----------------------\n"
        f"EMA9:   {s['ema9']:.5f}\n"
        f"EMA21:  {s['ema21']:.5f}\n"
        f"SMA50:  {s['sma50']:.5f}\n"
        f"{s['rsi_txt']}\n"
        f"Stoch6: {s['stoch6']:.1f}\n"
        f"MACD(20,36,10): {s['hist']:.6f}\n"
        f"-----------------------\n"
        f"SL:  {sl:.5f}\n"
        f"TP1: {tp1:.5f} (1:1.5)\n"
        f"TP2: {tp2:.5f} (1:3)\n"
        f"TP3: {tp3:.5f} (1:5)\n"
        f"-----------------------\n"
        f"Horario: {converter_hora(s['horario'])}"
    )

# ============================================================
# TELEGRAM
# ============================================================
def enviar(msg, chat_id=None):
    if not TELEGRAM_TOKEN: return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id or TELEGRAM_CHAT_ID, "text": msg,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=8)
    except Exception as e: print(f"Erro Telegram: {e}")

def buscar_updates():
    global ultimo_update_id
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={"offset": ultimo_update_id+1, "timeout": 1}, timeout=4)
        upds = r.json().get("result", [])
        if upds: ultimo_update_id = upds[-1]["update_id"]
        return upds
    except: return []


# CONTA DEMO - SISTEMA COMPLETO
# ============================================================

import json, math
from datetime import date

DEMO_FILE = "demo_historico.json"

SL_TP_PIPS = {
    "5min":  {"sl": 6,  "tp": 9,  "be": 6},
    "15min": {"sl": 12, "tp": 18, "be": 10},
    "30min": {"sl": 18, "tp": 27, "be": 15},
    "1h":    {"sl": 25, "tp": 40, "be": 20},
    "4h":    {"sl": 50, "tp": 80, "be": 40},
}

PIP_VALUE = {
    "XAU/USD": 0.10, "BTC/USDT": 1.0,
    "USD/JPY":  0.01, "EUR/JPY": 0.01,
    "GBP/JPY":  0.01, "AUD/JPY": 0.01,
}

def pip_size(par):
    return PIP_VALUE.get(par, 0.0001)

def pips_para_preco(par, pips):
    return pips * pip_size(par)

def carregar_demo():
    try:
        if os.path.exists(DEMO_FILE):
            return json.load(open(DEMO_FILE))
    except: pass
    return None

def salvar_demo(demo):
    try:
        json.dump(demo, open(DEMO_FILE, "w"), indent=2)
    except Exception as e:
        print(f"Erro salvar demo: {e}")

def criar_demo(saldo_inicial=50.0):
    demo = {
        "saldo_inicial": saldo_inicial,
        "saldo_atual":   saldo_inicial,
        "trades_abertos": {},
        "historico":      [],
        "total_trades":   0,
        "ganhos":         0,
        "perdas":         0,
        "breakevens":     0,
        "criado_em":      agora_brt(),
        "max_trades_dia": 3,
        "trades_hoje":    0,
        "data_hoje":      str(date.today()),
    }
    salvar_demo(demo)
    return demo

def resetar_trades_dia(demo):
    hoje = str(date.today())
    if demo.get("data_hoje") != hoje:
        demo["trades_hoje"] = 0
        demo["data_hoje"]   = hoje
    return demo

def abrir_trade_demo(demo, sinal):
    """Abre um trade demo baseado no sinal MTF"""
    demo = resetar_trades_dia(demo)

    if demo["trades_hoje"] >= demo["max_trades_dia"]:
        return demo, f"Limite de {demo['max_trades_dia']} trades por dia atingido!"

    par     = sinal["par"]
    tf      = sinal["tfs_conf"][0] if sinal["tfs_conf"] else "15min"
    direcao = sinal["direcao"]
    preco   = sinal["preco"]
    pips    = SL_TP_PIPS.get(tf, SL_TP_PIPS["15min"])
    ps      = pip_size(par)

    sl_preco = preco - pips["sl"]*ps if direcao=="COMPRA" else preco + pips["sl"]*ps
    tp_preco = preco + pips["tp"]*ps if direcao=="COMPRA" else preco - pips["tp"]*ps
    be_preco = preco + pips["be"]*ps if direcao=="COMPRA" else preco - pips["be"]*ps

    # Lote baseado no saldo (risco 2%)
    risco_usd = demo["saldo_atual"] * 0.02
    sl_pips   = pips["sl"]
    lote      = round(risco_usd / (sl_pips * ps * 100000), 2)
    lote      = max(0.01, min(lote, 1.0))

    trade_id = f"{par}_{tf}_{int(time.time())}"
    trade = {
        "id":          trade_id,
        "par":         par,
        "tf":          tf,
        "direcao":     direcao,
        "preco_entry": preco,
        "sl":          sl_preco,
        "tp":          tp_preco,
        "be_nivel":    be_preco,
        "be_movido":   False,
        "lote":        lote,
        "pips_sl":     pips["sl"],
        "pips_tp":     pips["tp"],
        "pips_be":     pips["be"],
        "prob":        sinal["prob"],
        "forca":       sinal["forca"],
        "horario":     agora_brt(),
        "data":        str(date.today()),
        "stat
