"""
bot_craft_world.py — Bot de sinais de trading para o CRAFT WORLD

Adaptado do bot de trading usado no Sunflower Land (bot_v2.py / bot_cw.py
enviado como referência). Diferença central: a API do Craft World só
devolve PREÇO (+ uma recomendação própria BUY/SELL/HOLD), não estoque nem
histórico de compras — então TUDO que dependia de posição/custódia foi
removido. Este bot não sabe (e não precisa saber) se "você" possui o item;
ele só analisa preço e gera sinais informativos.

O QUE FOI REMOVIDO em relação ao bot original (por depender de estoque):
  - Abas Inventario / TradeLog e toda leitura delas
  - calcular_fifo_hibrido, custo médio, calcular_dias_custodia
  - Distinção "origem" (possui vs não possui) — o bot sempre opera como se
    não possuísse nada, gerando sinais de COMPRA (zona de entrada) e VENDA
    (zona de saída) puramente estatísticos/preço
  - Stop-loss por prejuízo de posição (dependia de custo médio)
  - Abas Alocacao_Recursos, Portfolio_Correlacao, Debug_Decisao e
    Sinais_Hoje (dashboard) — só ficam as 4 abas pedidas:
    Sinais_IA, Cerebro_RL, Cerebro_RL_Item, XGB_Audit
  - Índice de correlação entre insumos (MAPA_RECEITA / Wood→Stone→Iron):
    deixado de fora por enquanto — ver seção "EXTENSÃO FUTURA" abaixo.

O QUE FOI ADICIONADO (a "Dados" traz uma recomendação BUY/SELL própria,
vinda da API, junto com o preço):
  - Recomendacao_Num: a recomendação da API vira feature do XGBoost
    (substitui Season_Num, que não existe mais aqui — Craft World não
    tem estações)
  - Var_72h: variação percentual do preço nas últimas 72h (3 dias) — o
    Craft World não tem estações, mas tem "masterpieces" que duram no
    máximo 3 dias e dão um empurrão pontual na demanda; essa feature dá
    ao XGBoost um sinal direto desse tipo de evento de curto prazo
  - Recomendacao_API / Concorda_API: gravadas em Sinais_IA, para comparar
    depois no relatório
  - Lucro_Desejado_%: meta de lucro agora é DINÂMICA por item (escala
    com a volatilidade relativa — Desvio_Macro/Media_Macro), em vez de
    um valor fixo igual pra todos. Calculada e congelada no momento em
    que cada sinal é gerado (ver calcular_lucro_desejado_dinamico), pra
    validação e recompensa usarem sempre o mesmo valor que gerou aquele
    sinal específico, mesmo que a volatilidade do item mude depois
  - Recompensa do Cérebro RL agora combina DOIS sinais de acerto/erro:
    (a) o preço realmente bateu a meta em N dias (igual ao bot original,
        mas normalizado pelo Lucro_Desejado_% daquele sinal, não mais
        por uma constante fixa)
    (b) o nosso sinal concordou com a recomendação da API no momento em
        que foi gerado
    ver PESO_RECOMPENSA_PRECO / PESO_RECOMPENSA_API

O QUE NÃO MUDOU (herdado sem alteração de lógica): validação de sinais
com janela de monitoramento, calibração Bayesiana dos mapas de risco,
Q-learning tabular (global + item-específico), veto RL adaptativo,
detector de regime (crash), margem de atenção, dedup híbrida (4h/3%),
cache incremental de indicadores, modelo XGBoost pooled (1 modelo pra
todos os itens), escrita atômica.

VALORES CONFIRMADOS (Craft World, não são mais suposição do SFL):
  - REGISTROS_POR_DIA = 24 — continua valendo mesmo com a coleta e o bot
    rodando a cada 15 min: normalizar_grade_horaria reduz a aba Dados a 1
    linha por item por hora (último preço da hora) ANTES de qualquer
    cálculo, então o histórico antigo (horário, ~2 meses) e os registros
    novos (15 min) ficam na mesma escala. A hora em andamento é regravada
    a cada ciclo, então o sinal é reavaliado a cada 15 min com o preço mais
    recente
  - TAXA_MARKETPLACE = 0.05 — 5% confirmado
  - DIAS_SWING reduzido de 12 (SFL) pra 5: o Craft World é mais dinâmico
    e o próprio evento de demanda (masterpiece) já se resolve em até 3
    dias — 5 dias dá uma folga além do próprio evento pro preço reverter,
    sem herdar a janela de 12 dias do jogo antigo. JANELA_EVENTO_H (72h /
    3 dias) amarra o holdout do XGBoost e a janela de auditoria a esse
    mesmo ciclo. As janelas de Média/RSI (JANELA_MACRO_DIAS/
    JANELA_MICRO_DIAS) também foram encurtadas — não é preciso usar os
    2 meses inteiros só pra ter uma referência de "preço médio".
  - Isso É uma mudança estrutural de arquitetura (SHIFT_ALVO, holdout e
    janelas de indicadores todos derivam de DIAS_SWING) — os números
    abaixo são um ponto de partida razoável, mas ajuste livremente
    depois de ver XGB_Audit/Sinais_IA reais.

⚠️ AINDA EM ABERTO — confirme antes de rodar em produção:
  - MAPA_API_NUM — só vi "BUY" no exemplo que você mandou; se a API
    também devolver "SELL"/"HOLD"/outra coisa, confirme os rótulos
    exatos (comparação é case-insensitive, mas o rótulo precisa bater)
  - calcular_lucro_desejado_dinamico usa um multiplicador de partida
    (1.5x a volatilidade relativa) — recalibrar depois de acumular
    sinais reais em Sinais_IA
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import gspread
import os
import math
import pickle
import requests
from datetime import datetime, timedelta, timezone
from oauth2client.service_account import ServiceAccountCredentials
from gspread_formatting import set_frozen, format_cell_range, CellFormat, TextFormat

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================

# --- Planilha única (Craft World não separa origem/V2 — tudo na mesma) ---
SPREADSHEET_ID = "1NjtxK3HgsnT2DiI7NEEHS1NCrwcJAedqB4c4YDiBrbU"   # CRAFT WORLD
ABA_DADOS = "Dados"    # equivalente ao RecLog do bot antigo — só leitura

REGISTROS_POR_DIA = 24         # confirmado: 1 registro/hora
TAXA_MARKETPLACE = 0.05        # confirmado: 5% — usado só pra calcular meta de venda acima da média, sem ligação com estoque

# --- Ciclo de referência: masterpieces duram no máximo 3 dias ---
JANELA_EVENTO_H = 3 * REGISTROS_POR_DIA   # 72 registros/horas — holdout do XGBoost e janela de auditoria usam esse ciclo

DIAS_SWING = 5                 # reduzido de 12 (SFL) — ver docstring "VALORES CONFIRMADOS"
SHIFT_ALVO = DIAS_SWING * REGISTROS_POR_DIA
DIAS_MAX_MONITORAMENTO = 12

# Janelas de Média/RSI — não precisam olhar os 2 meses inteiros, só servem
# de referência de "como o preço se comporta" pra banda comprar-abaixo/
# vender-acima-da-média. Derivadas de DIAS_SWING pra escalar junto se ele mudar.
JANELA_MACRO_DIAS = DIAS_SWING * 3   # 15 dias — referência da média/desvio da banda
JANELA_MICRO_DIAS = DIAS_SWING       # 5 dias — RSI mais responsivo que os 10 dias do SFL
MACRO_MIN_PERIODS = 60               # ~16% da janela macro (mesma proporção do bot original)
MICRO_MIN_PERIODS = 24               # ~20% da janela micro (mesma proporção do bot original)

LUCRO_DESEJADO_MINIMO = 0.03    # piso da meta de lucro dinâmica (itens de baixa volatilidade)
LUCRO_DESEJADO_MAXIMO = 0.20    # teto da meta de lucro dinâmica (itens de alta volatilidade / masterpiece)

HEADERS_AUDIT = [
    "Timestamp", "Item", "MAE", "RMSE",
    "Acerto_Direcao_%", "N_Treino", "N_Teste", "ROI_Previsto_%"
]

HEADERS_SINAIS_IA = [
    "ID_Sinal", "Tipo", "Data_Sinal", "Item", "Preco_Sinal", "Referencia_Fin",
    "Lucro_Desejado_%", "Recomendacao_API", "Concorda_API", "Monitoramento",
    "Status", "Dias_Passados", "Estado_RL", "Chave_Unica", "Dias_Ate_Alvo",
    "Preco_Final", "Estado_RL_Continuo"
]

# --- Recomendação da API (linha de "Dados") -> feature numérica ---
# TODO: confirmar todos os rótulos possíveis que a API devolve (só vimos
# "BUY" no exemplo). Rótulo desconhecido cai em 0 (neutro) via fillna.
MAPA_API_NUM = {"BUY": 1, "SELL": -1, "HOLD": 0, "WAIT": 0, "NEUTRO": 0, "NEUTRAL": 0}
MAPA_CONCORDA_NUM = {"SIM": 1.0, "NÃO": -1.0, "NEUTRO": 0.0}

# --- Pesos da recompensa combinada do Cérebro RL (preço real + API) ---
PESO_RECOMPENSA_PRECO = 0.7
PESO_RECOMPENSA_API = 0.3

# X_COLS_BASE: igual ao bot original, trocando Season_Num (não existe no
# Craft World) por Recomendacao_Num (a recomendação que vem da API) e
# adicionando Var_72h (variação nas últimas 72h, alinhada ao ciclo das
# masterpieces).
X_COLS_BASE = ['Price', 'Lag_1', 'Lag_2', 'Lag_3', 'RSI_Micro', 'Media_Macro',
               'Desvio_Macro', 'Recomendacao_Num', 'Var_72h', 'Hora_Dia', 'Dia_Semana']
# EXTENSÃO FUTURA: se/quando o Craft World tiver uma cadeia de produção
# conhecida (ex.: item A usa item B como insumo), dá pra reintroduzir um
# X_COLS_CORRELACAO = ['Indice_Insumo', 'Margem_Insumo'] igual ao bot do
# SFL (MAPA_RECEITA + construir_indices_insumos + adicionar_features_correlacao).
# Por ora, decidido deixar de fora.
X_COLS_POOLED = X_COLS_BASE

# --- Cache incremental (feature store leve) ---
CACHE_DIR = "cache_craft_world"
CACHE_INDICADORES_PATH = os.path.join(CACHE_DIR, "indicadores_incrementais.pkl")
MODELO_POOLED_PATH = os.path.join(CACHE_DIR, "modelo_pooled.json")
MODELO_POOLED_META_PATH = os.path.join(CACHE_DIR, "modelo_pooled_meta.pkl")

ALPHA_MINIMO = 0.05


# ==============================================================================
# FUNÇÕES HERDADAS (sem alteração de lógica — ver bot original)
# ==============================================================================

def limpar_numero(texto):
    if not texto:
        return 0.0
    s = str(texto).strip()
    if '.' in s and ',' in s:
        s = s.replace('.', '')
    s = s.replace(',', '.')
    if s.count('.') > 1:
        partes = s.split('.')
        s = "".join(partes[:-1]) + "." + partes[-1]
    try:
        return float(s)
    except Exception:
        return 0.0


def obter_agora_brt():
    """Datetime atual em Horário de Brasília (BRT), sem tzinfo (compat pandas)."""
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-3))).replace(tzinfo=None)


def enviar_telegram(acao, ativo, preco, id_sinal):
    """Alerta via Telegram."""
    # TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID: configuráveis via variável de
    # ambiente (GitHub Actions secret) — os valores fixos abaixo só entram
    # como fallback pra rodar localmente sem precisar configurar nada.
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "8715722660:AAGgJSl2w2RpM0kxZ9Y4MFK4lHpDZaLyQmw")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "873167793")

    preco_formatado = f"{preco:.4f}".replace('.', ',')

    if "COMPRAR" in acao.upper():
        emoji = "🟢"
    elif "STOP" in acao.upper():
        emoji = "⚠️"
    else:
        emoji = "🔴"

    hora_formatada = obter_agora_brt().strftime("%d/%m %H:%M:%S")

    mensagem = (
        f"{emoji} *ALERTA DE OPERAÇÃO* {emoji}\n\n"
        f"*Ação:* {acao.upper()}\n"
        f"*Ativo:* {ativo}\n"
        f"*Preço:* {preco_formatado}\n"
        f"*ID Sinal:* {id_sinal}\n"
        f"*Hora:* {hora_formatada}"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": mensagem, "parse_mode": "Markdown"}

    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"❌ Erro ao enviar mensagem para o Telegram: {e}")


def parse_data_sinal_robusta(texto):
    if not texto or str(texto).strip() == "":
        return pd.NaT
    texto = str(texto).strip()
    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y"):
        dt = pd.to_datetime(texto, format=fmt, errors='coerce')
        if pd.notnull(dt):
            return dt
    return pd.to_datetime(texto, dayfirst=True, errors='coerce')


def salvar_aba_atomica(sh, titulo, header, linhas, rows_est=None, cols_est=None, formatar_header=True):
    """Escrita atômica: grava numa aba temporária, só substitui a original após confirmar sucesso."""
    titulo_temp = f"{titulo}_tmp_{int(datetime.now().timestamp())}"
    n_linhas = len(linhas) + 1
    n_cols = len(header)
    rows_final = str(max(rows_est or 0, n_linhas + 100))
    cols_final = str(max(cols_est or 0, n_cols))

    ws_temp = None
    try:
        ws_temp = sh.add_worksheet(title=titulo_temp, rows=rows_final, cols=cols_final)
        ws_temp.update(values=[header] + linhas, range_name='A1', value_input_option='USER_ENTERED')
        if formatar_header:
            set_frozen(ws_temp, rows=1)
            ultima_col = chr(ord('A') + min(n_cols, 26) - 1)
            format_cell_range(ws_temp, f'A1:{ultima_col}1', CellFormat(textFormat=TextFormat(bold=True)))
    except Exception as e:
        print(f"❌ Falha ao escrever aba temporária para '{titulo}': {e} — aba original preservada.")
        if ws_temp is not None:
            try:
                sh.del_worksheet(ws_temp)
            except Exception:
                pass
        return False

    try:
        try:
            ws_antiga = sh.worksheet(titulo)
            sh.del_worksheet(ws_antiga)
        except Exception:
            pass
        ws_temp.update_title(titulo)
        return True
    except Exception as e:
        print(f"⚠️ Escrita em '{titulo}' concluída, mas troca de nome falhou: {e}. "
              f"Dados seguros na aba temporária '{titulo_temp}' — verifique manualmente.")
        return False


def calcular_veto_rl(estado_rl, q_table, n_obs_por_estado=None, q_compra_item=None, n_obs_item=0):
    if estado_rl not in q_table:
        q_compra_global = 0.0
    else:
        q_compra_global = q_table[estado_rl].get("COMPRA", 0.0)

    if q_compra_item is not None:
        peso_item = min(1.0, n_obs_item / 10.0)
        q_compra = peso_item * q_compra_item + (1 - peso_item) * q_compra_global
    else:
        q_compra = q_compra_global

    if estado_rl not in q_table and q_compra_item is None:
        return False, 0.0

    n_obs = (n_obs_por_estado or {}).get(estado_rl, 0)

    if n_obs < 5:
        threshold = -0.5
    elif n_obs < 20:
        threshold = -0.5 + (0.05 / 15) * (n_obs - 5)
    else:
        threshold = -0.45

    vetado = q_compra < threshold
    bonus_roi = 0.0
    if q_compra > 0.7:
        bonus_roi = 10.0
    elif q_compra > 0.5:
        bonus_roi = 5.0

    return vetado, bonus_roi


def contar_obs_por_estado(df_val):
    if df_val is None or df_val.empty:
        return {}
    concluidos = df_val[df_val['Status'].str.contains("🏆|❌|💰", na=False)]
    if concluidos.empty:
        return {}
    return (
        concluidos['Estado_RL'].astype(str).str.strip().str.replace("'", "")
        .value_counts().to_dict()
    )


def calcular_taxa_confianca(n_total, n_acertos, soma_recompensas,
                             prior_a=1.0, prior_b=1.0, n_minimo=8, confianca_alvo=0.80):
    if n_total == 0:
        return 0.5, False, 0.0

    a = prior_a + n_acertos
    b = prior_b + (n_total - n_acertos)

    taxa_efetiva = a / (a + b)
    ev_esperado = soma_recompensas / n_total if n_total > 0 else 0.0

    variancia = (a * b) / ((a + b) ** 2 * (a + b + 1))
    ic_inferior = max(0, taxa_efetiva - 1.96 * math.sqrt(variancia))
    confiante = (n_total >= n_minimo) and (ic_inferior > 0.35)

    return taxa_efetiva, confiante, ev_esperado


def calcular_prior_empirico(df_conc, forca_prior=4.0):
    if df_conc is None or df_conc.empty:
        return 1.0, 1.0
    taxa_global = df_conc['Pts'].apply(lambda x: x >= 0.5).mean()
    prior_a = max(forca_prior * taxa_global, 0.5)
    prior_b = max(forca_prior * (1 - taxa_global), 0.5)
    return prior_a, prior_b


def calibrar_por_taxa_confianca(df_val_mem, mapa_mult, mapa_roi, mapa_veto):
    if df_val_mem is None or df_val_mem.empty:
        return

    df_conc = df_val_mem[df_val_mem['Status'].str.contains("🏆|❌|💰", na=False)].copy()
    if df_conc.empty:
        return

    def pontos(st):
        if "🏆" in st:
            return 1.0
        if "💰" in st:
            return 0.5
        return 0.0

    df_conc['Pts'] = df_conc['Status'].apply(pontos)
    prior_a_global, prior_b_global = calcular_prior_empirico(df_conc)

    for item_c, grupo in df_conc.groupby('Item'):
        n = len(grupo)
        n_acertos = int(grupo['Pts'].apply(lambda x: x >= 0.5).sum())
        soma_pts = grupo['Pts'].sum()

        taxa_ef, confiante, ev = calcular_taxa_confianca(
            n, n_acertos, soma_pts, prior_a=prior_a_global, prior_b=prior_b_global
        )

        if not confiante:
            mapa_mult[item_c] = 2.2
            mapa_roi[item_c] = -2.0
            mapa_veto[item_c] = False
        elif ev >= 0.4:
            mapa_mult[item_c] = 2.0
            mapa_roi[item_c] = 0.0
            mapa_veto[item_c] = False
        elif ev >= 0.1:
            mapa_mult[item_c] = 2.4
            mapa_roi[item_c] = -3.0
            mapa_veto[item_c] = False
        elif ev >= -0.1:
            mapa_mult[item_c] = 2.4
            mapa_roi[item_c] = -5.0
            mapa_veto[item_c] = True
        else:
            mapa_mult[item_c] = 2.8
            mapa_roi[item_c] = -10.0
            mapa_veto[item_c] = True


def calcular_margem_atencao(alvo_in, alvo_out, fracao=0.15, minimo_abs=0.0001, maximo_pct=0.10):
    banda = max(0, alvo_out - alvo_in)
    margem = banda * fracao
    margem = max(margem, minimo_abs)
    if alvo_in > 0:
        margem = min(margem, alvo_in * maximo_pct)
    return round(margem, 6)


def detectar_regime(d_item, janela_crash_h=24, limiar_queda=-0.12):
    if d_item is None or len(d_item) < janela_crash_h:
        return False, 0.0

    preco_atual = d_item.iloc[-1]['Price']
    preco_24h = d_item.iloc[-janela_crash_h]['Price']

    if preco_24h == 0:
        return False, 0.0

    variacao_24h = (preco_atual - preco_24h) / preco_24h
    return variacao_24h < limiar_queda, round(variacao_24h * 100, 2)


def calcular_roi_real(tipo, preco_sinal, preco_final):
    if preco_sinal <= 0:
        return 0.0
    if tipo == "COMPRA":
        venda_liquida = preco_final * (1 - TAXA_MARKETPLACE)
        return ((venda_liquida - preco_sinal) / preco_sinal) * 100
    return ((preco_sinal - preco_final) / preco_sinal) * 100


# ==============================================================================
# NOVO (Craft World) — recomendação da API x nosso sinal / Estado_RL sem estoque
# ==============================================================================

def calcular_lucro_desejado_dinamico(desvio_macro, media_macro,
                                      minimo=LUCRO_DESEJADO_MINIMO,
                                      maximo=LUCRO_DESEJADO_MAXIMO,
                                      multiplicador=1.5):
    """
    Lucro desejado DINÂMICO por item, em vez de uma meta fixa (7% no bot
    do SFL) igual pra todo mundo. Escala com a volatilidade relativa do
    item (Desvio_Macro / Media_Macro): itens mais voláteis — inclusive
    por causa de masterpieces de até 3 dias que dão um empurrão pontual
    na demanda — comportam uma meta de lucro maior; itens mais estáveis
    usam uma meta mais conservadora.

    Calculado e GRAVADO em Sinais_IA no momento em que o sinal é gerado
    (coluna Lucro_Desejado_%) — validação e recompensa do RL reusam esse
    valor congelado, em vez de recalcular com a volatilidade atual (que
    pode já ter mudado quando o sinal é resolvido).

    `multiplicador` é um ponto de partida — vale recalibrar depois de
    acumular sinais reais em Sinais_IA / XGB_Audit.
    """
    if media_macro <= 0:
        return minimo
    volat_relativa = desvio_macro / media_macro
    lucro = volat_relativa * multiplicador
    return min(maximo, max(minimo, lucro))


def avaliar_concordancia_api(tipo_sinal, recomendacao_api_raw):
    """
    Compara nosso sinal (COMPRA/VENDA) com a recomendação bruta vinda da API
    ('BUY'/'SELL'/...). Retorna (rotulo_str, valor_numerico): o rótulo vai
    pra coluna Concorda_API de Sinais_IA; o valor numérico alimenta a
    recompensa combinada do Cérebro RL (ver PESO_RECOMPENSA_API).
    """
    rec = str(recomendacao_api_raw).strip().upper()
    if tipo_sinal == "COMPRA":
        if rec == "BUY":
            return "SIM", 1.0
        if rec == "SELL":
            return "NÃO", -1.0
        return "NEUTRO", 0.0
    if tipo_sinal == "VENDA":
        if rec == "SELL":
            return "SIM", 1.0
        if rec == "BUY":
            return "NÃO", -1.0
        return "NEUTRO", 0.0
    return "NEUTRO", 0.0


def construir_estado_rl_craft(row_mercado, roi_pred, alvo_in, alvo_out, margem_atencao, em_crash):
    """
    Estado_RL discreto de 7 caracteres (mesma arquitetura do bot original),
    trocando as 3 dimensões que dependiam de estoque/custódia (origem,
    lucro_disc, tempo_disc) por recomendação da API, concordância API x
    XGBoost, e proximidade da banda de entrada/saída:

      [0] RSI:         0 sobrevendido (<35) / 1 neutro / 2 sobrecomprado (>65)
      [1] Volatilidade: 0 baixa / 1 alta (desvio > 10% da média)
      [2] XGBoost:     0 previsão de queda (<-4%) / 2 previsão de alta (>6%) / 1 neutro
      [3] API:         0 API recomenda SELL / 2 API recomenda BUY / 1 neutro/outro
      [4] Concordância: 1 se XGBoost e API apontam pro mesmo lado / 0 caso contrário
      [5] Regime:      1 se em queda >12% nas últimas 24h / 0 normal
      [6] Banda:       0 fora das zonas / 1 perto da entrada / 2 perto da saída
    """
    rsi_val = row_mercado.get('RSI_Micro', 50)
    rsi_disc = 0 if rsi_val < 35 else (2 if rsi_val > 65 else 1)

    media = row_mercado.get('Media_Macro', 1)
    desvio = row_mercado.get('Desvio_Macro', 0)
    proporcao_volat = desvio / media if media > 0 else 0
    volat_disc = 1 if proporcao_volat > 0.10 else 0

    xgb_disc = 0 if roi_pred < -4 else (2 if roi_pred > 6 else 1)

    api_num = row_mercado.get('Recomendacao_Num', 0)
    api_disc = int(api_num) + 1  # -1/0/1 -> 0/1/2

    concorda_disc = 1 if (xgb_disc == 2 and api_disc == 2) or (xgb_disc == 0 and api_disc == 0) else 0
    crash_disc = 1 if em_crash else 0

    preco_atual = row_mercado.get('Price', 0)
    if preco_atual <= alvo_in + margem_atencao:
        banda_disc = 1
    elif preco_atual >= alvo_out - margem_atencao:
        banda_disc = 2
    else:
        banda_disc = 0

    return f"{rsi_disc}{volat_disc}{xgb_disc}{api_disc}{concorda_disc}{crash_disc}{banda_disc}"


def construir_estado_rl_continuo_craft(row_mercado, roi_pred, alvo_in, alvo_out, var24h):
    """
    Versão contínua (não discretizada) das mesmas dimensões acima — gravada
    em 'Estado_RL_Continuo' para, no futuro, treinar um modelo de regressão
    sobre o estado bruto e comparar contra a Q-table tabular. FASE ATUAL:
    apenas coleta, ainda não decide nada.
    """
    rsi_val = row_mercado.get('RSI_Micro', 50)
    media = row_mercado.get('Media_Macro', 1)
    desvio = row_mercado.get('Desvio_Macro', 0)
    proporcao_volat = desvio / media if media > 0 else 0
    api_num = row_mercado.get('Recomendacao_Num', 0)
    banda = max(0.0, alvo_out - alvo_in)
    preco_atual = row_mercado.get('Price', 0)
    dist_banda_norm = ((preco_atual - alvo_in) / banda) if banda > 0 else 0.5
    dist_banda_norm = min(1.0, max(0.0, dist_banda_norm))

    valores = [rsi_val, proporcao_volat, roi_pred, api_num, var24h, dist_banda_norm]
    return ",".join(f"{v:.4f}" for v in valores)


# ==============================================================================
# CACHE INCREMENTAL DE INDICADORES (herdado, sem alteração estrutural)
# ==============================================================================

def normalizar_grade_horaria(df):
    """
    Reduz a aba Dados a UMA linha por item por hora (o último preço e a
    última recomendação da API dentro daquela hora).

    Por quê: todo o resto do bot (janelas de média/RSI, Lags, Var_72h,
    SHIFT_ALVO, holdout, regra "3 horas no alvo") conta LINHAS e assume
    1 linha = 1 hora (REGISTROS_POR_DIA = 24). Com a coleta a cada 15 min
    — e os 2 meses de histórico antigos ainda em cadência horária — essas
    contagens mudariam de significado e misturariam duas escalas. Fixar a
    grade em 1 hora preserva o histórico e mantém todos os parâmetros
    valendo, enquanto o bot continua rodando a cada 15 min: a hora em
    andamento sempre carrega o preço mais recente, então o sinal da hora
    atual é reavaliado a cada ciclo.

    O Timestamp resultante é o início da hora (17:48:25 -> 17:00:00).
    """
    d = df.copy()
    ts = pd.to_datetime(d['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    faltou = ts.isna()
    if faltou.any():
        ts[faltou] = pd.to_datetime(d.loc[faltou, 'Timestamp'], dayfirst=True, errors='coerce')
    d['Timestamp'] = ts
    d = d.dropna(subset=['Timestamp'])
    d = d[d['Price'] > 0]   # linha com preço ilegível (limpar_numero -> 0) não pode virar o "último preço" da hora
    d['Hora'] = d['Timestamp'].dt.floor('h')
    d = d.sort_values('Timestamp').groupby(['Item', 'Hora'], as_index=False, sort=False).last()
    d['Timestamp'] = d['Hora']
    return (d[['Timestamp', 'Item', 'Price', 'Recomendacao_API']]
            .sort_values(['Item', 'Timestamp']).reset_index(drop=True))


# Sobe quando o formato/lógica do cache muda — um cache salvo com versão
# diferente é descartado e recalculado do zero (ver carregar_cache_indicadores).
# v2: grade horária + Preco_Futuro/Peso_Temporal recalculados a cada ciclo.
CACHE_VERSAO = 2


def carregar_cache_indicadores():
    if os.path.exists(CACHE_INDICADORES_PATH):
        try:
            with open(CACHE_INDICADORES_PATH, "rb") as f:
                cache = pickle.load(f)
            if not isinstance(cache, dict) or cache.get('__versao__') != CACHE_VERSAO:
                print("ℹ️  Cache de indicadores de versão anterior — descartado, recomputando do zero neste ciclo.")
                return {}
            return cache
        except Exception as e:
            print(f"⚠️ Cache de indicadores corrompido/ilegível ({e}) — recomputando do zero neste ciclo.")
    return {}


def salvar_cache_indicadores(cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache['__versao__'] = CACHE_VERSAO
    with open(CACHE_INDICADORES_PATH, "wb") as f:
        pickle.dump(cache, f)


def _calcular_indicadores_do_zero(d_bruto_item, nome_item):
    """
    Núcleo de cálculo de indicadores. Igual ao bot original, trocando
    Season_Num (não existe no Craft World) por Recomendacao_Num (vinda da
    API). is_holdout é atribuído à parte por _atribuir_holdout.
    """
    d = d_bruto_item.sort_values('Timestamp').reset_index(drop=True).copy()

    for col in ['Media_Macro', 'Desvio_Macro', 'RSI_Micro', 'Recomendacao_Num', 'Var_72h',
                'Lag_1', 'Lag_2', 'Lag_3', 'Hora_Dia', 'Dia_Semana',
                'Preco_Futuro', 'Peso_Temporal']:
        d[col] = np.nan
    d['is_holdout'] = False

    janela_macro_dinamica = JANELA_MACRO_DIAS * REGISTROS_POR_DIA
    janela_micro_dinamica = JANELA_MICRO_DIAS * REGISTROS_POR_DIA

    if len(d) < janela_micro_dinamica:
        return d

    d['Media_Macro'] = d['Price'].rolling(window=janela_macro_dinamica, min_periods=MACRO_MIN_PERIODS).mean()
    d['Desvio_Macro'] = d['Price'].rolling(window=janela_macro_dinamica, min_periods=MACRO_MIN_PERIODS).std()

    delta = d['Price'].diff()
    ganho = delta.clip(lower=0).rolling(window=janela_micro_dinamica, min_periods=MICRO_MIN_PERIODS).mean()
    perda = -1 * delta.clip(upper=0).rolling(window=janela_micro_dinamica, min_periods=MICRO_MIN_PERIODS).mean()
    rsi_raw = 100 - (100 / (1 + (ganho / perda.replace(0, np.nan))))
    d['RSI_Micro'] = rsi_raw.fillna(100)

    d['Recomendacao_Num'] = (
        d['Recomendacao_API'].astype(str).str.strip().str.upper().map(MAPA_API_NUM).fillna(0)
    )
    # Variação nas últimas 72h (3 dias) — alinhada ao ciclo máximo das
    # masterpieces, dá ao modelo um sinal direto desse tipo de evento.
    d['Var_72h'] = d['Price'].pct_change(periods=JANELA_EVENTO_H) * 100
    d['Lag_1'] = d['Price'].shift(1)
    d['Lag_2'] = d['Price'].shift(2)
    d['Lag_3'] = d['Price'].shift(3)
    d['Hora_Dia'] = d['Timestamp'].dt.hour
    d['Dia_Semana'] = d['Timestamp'].dt.dayofweek
    d['Preco_Futuro'] = d['Price'].shift(-SHIFT_ALVO)

    max_date = d['Timestamp'].max()
    d['Peso_Temporal'] = d['Timestamp'].apply(lambda x: np.exp(-0.01 * (max_date - x).days))

    return d


def _atribuir_holdout(d):
    """Marca is_holdout (3 janelas de 72h) sobre o histórico completo — igual ao bot original."""
    d = d.copy()
    d['is_holdout'] = False
    HOLDOUT_JANELA = JANELA_EVENTO_H   # 72h/3 dias — mesmo ciclo das masterpieces
    N_JANELAS_HOLDOUT = 3
    WARMUP_MACRO = MACRO_MIN_PERIODS

    zona_fim = len(d) - SHIFT_ALVO
    zona_disponivel = zona_fim - WARMUP_MACRO

    if zona_disponivel >= (N_JANELAS_HOLDOUT * HOLDOUT_JANELA + 50):
        tamanho_segmento = zona_disponivel // N_JANELAS_HOLDOUT
        for i in range(N_JANELAS_HOLDOUT):
            seg_fim = WARMUP_MACRO + tamanho_segmento * (i + 1)
            seg_ini = max(WARMUP_MACRO, seg_fim - HOLDOUT_JANELA)
            if seg_fim <= zona_fim:
                d.iloc[seg_ini:seg_fim, d.columns.get_loc('is_holdout')] = True
    else:
        holdout_ini = zona_fim - HOLDOUT_JANELA
        if holdout_ini > WARMUP_MACRO:
            d.iloc[holdout_ini:zona_fim, d.columns.get_loc('is_holdout')] = True

    return d


def preparar_indicadores_incremental(df_bruto_geral, nome_item, cache):
    """
    Versão com cache: só as linhas novas desde o último ciclo são
    recalculadas, evitando reprocessar o histórico inteiro a cada execução.
    cache[item] = {"df": DataFrame já processado, "ultimo_ts": Timestamp}
    """
    d_bruto_item = df_bruto_geral[df_bruto_geral['Item'] == nome_item].copy()
    d_bruto_item['Timestamp'] = pd.to_datetime(d_bruto_item['Timestamp'], dayfirst=True, errors='coerce')
    d_bruto_item = d_bruto_item.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)

    if d_bruto_item.empty:
        return pd.DataFrame()

    ultimo_ts_global = d_bruto_item['Timestamp'].max()
    cache_item = cache.get(nome_item)

    if not cache_item or cache_item.get('df') is None or cache_item['df'].empty:
        d_full = _calcular_indicadores_do_zero(d_bruto_item, nome_item)
        d_full = _atribuir_holdout(d_full)
        cache[nome_item] = {'df': d_full, 'ultimo_ts': ultimo_ts_global}
        return d_full

    ultimo_ts_cache = cache_item['ultimo_ts']
    # `>=` (e não `>`): com o bot rodando a cada 15 min, a hora em andamento
    # é regravada em cada ciclo com o preço mais recente. A linha dessa hora
    # que já está no cache fica velha — ela é descartada e recalculada.
    linhas_novas = d_bruto_item[d_bruto_item['Timestamp'] >= ultimo_ts_cache]

    if linhas_novas.empty:
        return cache_item['df']

    janela_macro_dinamica = JANELA_MACRO_DIAS * REGISTROS_POR_DIA
    contexto = d_bruto_item[d_bruto_item['Timestamp'] < ultimo_ts_cache].tail(janela_macro_dinamica)
    bloco = pd.concat([contexto, linhas_novas]).reset_index(drop=True)
    bloco_processado = _calcular_indicadores_do_zero(bloco, nome_item)
    novas_processadas = bloco_processado[bloco_processado['Timestamp'] >= ultimo_ts_cache]

    df_cache = cache_item['df']
    d_full = pd.concat([df_cache[df_cache['Timestamp'] < ultimo_ts_cache], novas_processadas],
                       ignore_index=True)

    # Estas duas colunas dependem do histórico inteiro, não só da linha: o
    # preço futuro de uma linha só existe SHIFT_ALVO linhas depois, e o peso
    # temporal é relativo à data mais recente. Calculadas só nas linhas novas
    # (como antes) ficavam NaN/velhas pra sempre — o treino do XGBoost
    # congelava na data em que o cache foi criado.
    d_full['Preco_Futuro'] = d_full['Price'].shift(-SHIFT_ALVO)
    max_date = d_full['Timestamp'].max()
    d_full['Peso_Temporal'] = np.exp(-0.01 * (max_date - d_full['Timestamp']).dt.days)
    d_full = _atribuir_holdout(d_full)

    cache[nome_item] = {'df': d_full, 'ultimo_ts': ultimo_ts_global}
    return d_full


# ==============================================================================
# MODELO ÚNICO POOLED (1 XGBoost pra todos os itens, herdado sem alteração)
# ==============================================================================

def montar_datasets_pooled(cache_indicadores):
    frames_treino, frames_teste = [], []

    for item, d_it in cache_indicadores.items():
        if d_it is None or d_it.empty:
            continue
        d_it = d_it.copy()
        d_it['Item_Cat'] = item
        d_treino = d_it[~d_it['is_holdout']].dropna(subset=['Preco_Futuro'] + X_COLS_BASE)
        d_teste = d_it[d_it['is_holdout']].dropna(subset=['Preco_Futuro'] + X_COLS_BASE)
        if len(d_treino) >= 50:
            frames_treino.append(d_treino)
        if not d_teste.empty:
            frames_teste.append(d_teste)

    if not frames_treino:
        return pd.DataFrame(), pd.DataFrame()

    df_treino = pd.concat(frames_treino, ignore_index=True)
    df_teste = pd.concat(frames_teste, ignore_index=True) if frames_teste else pd.DataFrame()

    categorias = sorted(df_treino['Item_Cat'].unique().tolist())
    df_treino['Item_Cat'] = pd.Categorical(df_treino['Item_Cat'], categories=categorias)
    if not df_teste.empty:
        df_teste['Item_Cat'] = pd.Categorical(df_teste['Item_Cat'], categories=categorias)

    return df_treino, df_teste


def treinar_modelo_pooled(df_treino):
    if df_treino.empty:
        return None

    cols_modelo = X_COLS_POOLED + ['Item_Cat']
    modelo = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        enable_categorical=True,
        verbosity=0
    ).fit(
        df_treino[cols_modelo],
        df_treino['Preco_Futuro'],
        sample_weight=df_treino['Peso_Temporal']
    )
    return modelo


def deve_retreinar_pooled():
    """Retorna True se o modelo pooled deve ser retreinado neste ciclo — 1x/dia."""
    if not os.path.exists(MODELO_POOLED_META_PATH):
        return True
    try:
        with open(MODELO_POOLED_META_PATH, "rb") as f:
            meta = pickle.load(f)
        ultimo_treino = meta.get("ultimo_treino")
        return ultimo_treino is None or (obter_agora_brt() - ultimo_treino).total_seconds() >= 86400
    except Exception:
        return True


def salvar_modelo_pooled(modelo, categorias, features_treino):
    os.makedirs(CACHE_DIR, exist_ok=True)
    modelo.save_model(MODELO_POOLED_PATH)
    with open(MODELO_POOLED_META_PATH, "wb") as f:
        pickle.dump({
            "ultimo_treino": obter_agora_brt(),
            "categorias_item": list(categorias),
            "features_treino": list(features_treino),
        }, f)


def carregar_modelo_pooled():
    if not (os.path.exists(MODELO_POOLED_PATH) and os.path.exists(MODELO_POOLED_META_PATH)):
        return None, None
    try:
        with open(MODELO_POOLED_META_PATH, "rb") as f:
            meta = pickle.load(f)

        features_esperadas = X_COLS_POOLED + ['Item_Cat']
        features_salvas = meta.get("features_treino")
        if features_salvas != features_esperadas:
            print(f"⚠️ Schema de features do modelo pooled em cache não bate com o esperado pelo "
                  f"código atual (salvo={features_salvas} | esperado={features_esperadas}) — "
                  f"forçando retreino neste ciclo em vez de usar o modelo desatualizado.")
            return None, None

        modelo = xgb.XGBRegressor(enable_categorical=True)
        modelo.load_model(MODELO_POOLED_PATH)
        return modelo, meta.get("categorias_item")
    except Exception as e:
        print(f"⚠️ Falha ao carregar modelo pooled em cache ({e}) — será retreinado neste ciclo.")
        return None, None


# ==============================================================================
# CALIBRAÇÃO HÍBRIDA (lê XGB_Audit e Sinais_IA — herdado, só troca nomes de aba)
# ==============================================================================

def construir_calibracao_hibrida(sh, df_val_mem=None, q_table_mem=None):
    mapa_mult = {}
    mapa_veto_local = {}
    mapa_roi = {}
    df_audit_retorno = pd.DataFrame(columns=HEADERS_AUDIT)

    if df_val_mem is not None and not df_val_mem.empty:
        df_concluidos = df_val_mem[df_val_mem['Status'].str.contains("🏆|❌|💰", na=False)].copy()
        if not df_concluidos.empty:
            calibrar_por_taxa_confianca(df_val_mem, mapa_mult, mapa_roi, mapa_veto_local)
            print(f"📈 Calibração Bayesiana: {df_concluidos['Item'].nunique()} itens | "
                  f"{len(df_concluidos)} sinais concluídos.")
        else:
            print("ℹ️  Calibração: nenhum sinal concluído ainda — fase bootstrap.")
    else:
        print("ℹ️  Calibração: Sinais_IA vazia — fase bootstrap.")

    try:
        ws_audit = sh.worksheet("XGB_Audit")
        audit_data = ws_audit.get_all_values()
        if len(audit_data) > 1:
            df_audit = pd.DataFrame(audit_data[1:], columns=audit_data[0])
            df_audit['MAE'] = df_audit['MAE'].apply(limpar_numero)
            df_audit['Acerto_Direcao_%'] = df_audit['Acerto_Direcao_%'].apply(limpar_numero)
            df_audit_retorno = df_audit.copy()
    except gspread.exceptions.WorksheetNotFound:
        print("ℹ️  XGB_Audit ainda não existe — será criada neste ciclo.")
    except Exception as e:
        print(f"ℹ️  XGB_Audit inacessível: {e}")

    n_estados = len(q_table_mem) if q_table_mem is not None else 0
    print(f"{'🌱 BOOTSTRAP' if n_estados < 10 else '🧠 Cerebro_RL operacional'}: {n_estados} estados.")
    print(f"✅ Calibração final: {len(mapa_mult)} itens com parâmetros ajustados.")
    return mapa_mult, mapa_veto_local, mapa_roi, df_audit_retorno


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

print("🧪 Iniciando Bot Craft World...")

escopo = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Credenciais: em CI (GitHub Actions), a chave da service account vem inteira
# como secret GOOGLE_CREDENTIALS_JSON (conteúdo do .json, não o caminho) —
# evita ter que commitar o arquivo no repo. Localmente, cai no arquivo
# credenciais_robo.json de sempre.
_credenciais_env = os.environ.get("GOOGLE_CREDENTIALS_JSON")
if _credenciais_env:
    import json
    credenciais = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(_credenciais_env), escopo)
else:
    credenciais = ServiceAccountCredentials.from_json_keyfile_name("credenciais_robo.json", escopo)
gc = gspread.authorize(credenciais)

sh = gc.open_by_key(SPREADSHEET_ID)   # uma planilha só: Dados + as 4 abas geridas pelo bot

# ------------------------------------------------------------------
# 1. Dados de base — aba "Dados" (equivalente ao RecLog), só leitura
# ------------------------------------------------------------------
rec_data = sh.worksheet(ABA_DADOS).get_all_values()[1:]   # pula cabeçalho
df_bruto = pd.DataFrame([l[:4] for l in rec_data], columns=['Timestamp', 'Item', 'Price', 'Recomendacao_API'])
df_bruto['Price'] = df_bruto['Price'].apply(limpar_numero)
df_bruto['Recomendacao_API'] = df_bruto['Recomendacao_API'].astype(str).str.strip().str.upper()

# Coleta a cada 15 min -> 1 linha por item por hora (ver normalizar_grade_horaria)
n_linhas_dados = len(df_bruto)
df_bruto = normalizar_grade_horaria(df_bruto)
print(f"📥 Dados: {n_linhas_dados} registros lidos -> {len(df_bruto)} na grade horária.")

# ==============================================================================
# VALIDAÇÃO DE SINAIS (Sinais_IA — substitui Validacao_Sinais/Sinais_Hoje)
# ==============================================================================
print("🔄 Processando validação de sinais (Sinais_IA)...")

try:
    ws_val = sh.worksheet("Sinais_IA")
    val_data = ws_val.get_all_values()
except Exception:
    ws_val = sh.add_worksheet(title="Sinais_IA", rows="10000", cols=str(len(HEADERS_SINAIS_IA)))
    val_data = []

if not val_data:
    ws_val.update(values=[HEADERS_SINAIS_IA], range_name='A1', value_input_option='USER_ENTERED')
    val_data = [HEADERS_SINAIS_IA]

if len(val_data) > 0 and ("Concorda_API" not in val_data[0] or "Lucro_Desejado_%" not in val_data[0]):
    print("⚠️ Ajustando schema de Sinais_IA para incluir as colunas novas "
          "(Lucro_Desejado_%/Recomendacao_API/Concorda_API)...")
    df_val = pd.DataFrame(val_data[1:], columns=val_data[0]) if len(val_data) > 1 else pd.DataFrame(columns=val_data[0])
    df_val = df_val.reindex(columns=HEADERS_SINAIS_IA).fillna("")
else:
    df_val = pd.DataFrame(val_data[1:], columns=val_data[0]) if len(val_data) > 1 else pd.DataFrame(columns=HEADERS_SINAIS_IA)

mapa_mult_desvio = {}
mapa_veto = {}
mapa_ajuste_roi = {}

# ------------------------------------------------------------------
# Cerebro_RL / Cerebro_RL_Item — Q-learning tabular, idêntico ao bot original
# ------------------------------------------------------------------
q_table = {}

try:
    ws_cerebro = sh.worksheet("Cerebro_RL")
    cerebro_data = ws_cerebro.get_all_values()
    if len(cerebro_data) > 1:
        tem_contadores = len(cerebro_data[0]) >= 5
        for row in cerebro_data[1:]:
            if len(row) >= 3:
                st = str(row[0]).strip().replace("'", "")
                q_table[st] = {
                    "COMPRA": limpar_numero(row[1]),
                    "VENDA": limpar_numero(row[2]),
                    "N_COMPRA": int(limpar_numero(row[3])) if tem_contadores and len(row) > 3 else 0,
                    "N_VENDA": int(limpar_numero(row[4])) if tem_contadores and len(row) > 4 else 0,
                }
        print(f"🧠 Cérebro RL carregado: {len(q_table)} estados.")
except Exception:
    try:
        ws_cerebro = sh.add_worksheet(title="Cerebro_RL", rows="5000", cols="5")
        ws_cerebro.update(values=[["Estado_RL", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"]],
                           range_name='A1', value_input_option='USER_ENTERED')
        print("🧠 Nova aba Cerebro_RL criada.")
    except Exception:
        print("⚠️ Aba Cerebro_RL já existente, mas vazia ou inacessível.")

q_table_item = {}
try:
    ws_cerebro_item = sh.worksheet("Cerebro_RL_Item")
    cerebro_item_data = ws_cerebro_item.get_all_values()
    if len(cerebro_item_data) > 1:
        for row in cerebro_item_data[1:]:
            if len(row) >= 3:
                chave_ci = str(row[0]).strip().replace("'", "")
                q_table_item[chave_ci] = {
                    "COMPRA": limpar_numero(row[1]),
                    "VENDA": limpar_numero(row[2]),
                    "N_COMPRA": int(limpar_numero(row[3])) if len(row) > 3 else 0,
                    "N_VENDA": int(limpar_numero(row[4])) if len(row) > 4 else 0,
                }
        print(f"🧠 Cérebro RL item-específico carregado: {len(q_table_item)} combinações.")
except Exception:
    try:
        ws_cerebro_item = sh.add_worksheet(title="Cerebro_RL_Item", rows="20000", cols="5")
        ws_cerebro_item.update(values=[["Item_Estado", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"]],
                                range_name='A1', value_input_option='USER_ENTERED')
        print("🧠 Nova aba Cerebro_RL_Item criada.")
    except Exception:
        print("⚠️ Aba Cerebro_RL_Item já existente, mas vazia ou inacessível.")

indices_concluidos_neste_ciclo = []

if not df_val.empty:
    for idx, row_val in df_val.iterrows():
        if row_val['Status'] == 'Aguardando' or row_val['Status'] == '':
            item_v = row_val['Item']
            tipo_v = row_val['Tipo']
            data_sinal_v = parse_data_sinal_robusta(row_val['Data_Sinal'])
            preco_sinal_v = limpar_numero(row_val['Preco_Sinal'])
            ref_fin_v = limpar_numero(row_val['Referencia_Fin'])

            if pd.isnull(data_sinal_v):
                df_val.at[idx, 'Dias_Passados'] = 0
                df_val.at[idx, 'Status'] = "⚠️ ERRO_DATA (Verificar Manualmente)"
                continue

            dias_p = (obter_agora_brt() - data_sinal_v).days

            d_hist = df_bruto[df_bruto['Item'] == item_v].copy()
            d_hist['Timestamp'] = pd.to_datetime(d_hist['Timestamp'], dayfirst=True, errors='coerce')
            df_pos_sinal = d_hist[d_hist['Timestamp'] >= data_sinal_v]

            if tipo_v == "COMPRA":
                preco_monitoramento = df_pos_sinal['Price'].max() if not df_pos_sinal.empty else preco_sinal_v
            else:
                preco_monitoramento = df_pos_sinal['Price'].min() if not df_pos_sinal.empty else preco_sinal_v

            max_pos = df_pos_sinal['Price'].max() if not df_pos_sinal.empty else preco_sinal_v
            horas_no_alvo = int((df_pos_sinal['Price'] >= ref_fin_v).sum()) if not df_pos_sinal.empty else 0
            horas_venda = int((df_pos_sinal['Price'] <= ref_fin_v).sum()) if not df_pos_sinal.empty else 0

            dias_ate_alvo = None
            if not df_pos_sinal.empty and pd.notnull(data_sinal_v):
                qualifica = df_pos_sinal['Price'] >= ref_fin_v if tipo_v == "COMPRA" else df_pos_sinal['Price'] <= ref_fin_v
                cum = qualifica.cumsum()
                atingiu = cum[cum >= 3]
                if not atingiu.empty:
                    ts_atingiu = df_pos_sinal.loc[atingiu.index[0], 'Timestamp']
                    dias_ate_alvo = round((ts_atingiu - data_sinal_v).total_seconds() / 86400, 2)

            status_v = "Aguardando"

            # Lucro_Desejado_% foi congelado no momento em que o sinal foi
            # gerado (ver calcular_lucro_desejado_dinamico) — reusa o mesmo
            # valor aqui em vez de recalcular com a volatilidade atual.
            lucro_desejado_pct_v = limpar_numero(row_val.get('Lucro_Desejado_%', ''))
            if lucro_desejado_pct_v <= 0:
                lucro_desejado_pct_v = LUCRO_DESEJADO_MINIMO * 100   # fallback p/ linhas antigas sem essa coluna

            if tipo_v == "COMPRA":
                if dias_p >= DIAS_SWING:
                    piso_lucro_real = preco_sinal_v * (1 + TAXA_MARKETPLACE + lucro_desejado_pct_v / 100)
                    if horas_no_alvo >= 3:
                        status_v = "🏆 ACERTOU"
                    elif horas_no_alvo >= 1 or max_pos >= piso_lucro_real:
                        if dias_p >= DIAS_MAX_MONITORAMENTO:
                            status_v = "💰 LUCRO PARCIAL"
                    else:
                        if dias_p >= DIAS_MAX_MONITORAMENTO:
                            status_v = "❌ FALHOU"
            else:  # VENDA — sempre sinal informativo de "zona de saída", sem custo/estoque associado
                if dias_p >= DIAS_SWING:
                    if horas_venda >= 3:
                        status_v = "🏆 ACERTOU"
                    elif horas_venda >= 1:
                        if dias_p >= DIAS_MAX_MONITORAMENTO:
                            status_v = "💰 LUCRO PARCIAL"
                    else:
                        if dias_p >= DIAS_MAX_MONITORAMENTO:
                            status_v = "❌ FALHOU"

            df_val.at[idx, 'Dias_Passados'] = int(dias_p)
            df_val.at[idx, 'Monitoramento'] = round(preco_monitoramento, 4)
            df_val.at[idx, 'Status'] = status_v

            if dias_ate_alvo is None and status_v not in ("Aguardando",):
                dias_ate_alvo = float(dias_p)
            df_val.at[idx, 'Dias_Ate_Alvo'] = dias_ate_alvo if dias_ate_alvo is not None else ""

            if status_v not in ("Aguardando",) and not df_pos_sinal.empty:
                df_val.at[idx, 'Preco_Final'] = round(df_pos_sinal['Price'].iloc[-1], 4)
            elif status_v not in ("Aguardando",):
                df_val.at[idx, 'Preco_Final'] = round(preco_sinal_v, 4)

            if "🏆" in status_v or "❌" in status_v or "💰" in status_v:
                indices_concluidos_neste_ciclo.append(idx)

    df_val['ID_Sinal'] = pd.to_numeric(df_val['ID_Sinal'].apply(limpar_numero), errors='coerce').fillna(0).astype(int)
    df_val['Preco_Sinal'] = df_val['Preco_Sinal'].apply(limpar_numero)
    df_val['Referencia_Fin'] = df_val['Referencia_Fin'].apply(limpar_numero)
    df_val['Lucro_Desejado_%'] = df_val['Lucro_Desejado_%'].apply(
        lambda x: round(limpar_numero(x), 2) if str(x).strip() not in ("", "nan", "None") else ""
    )
    df_val['Monitoramento'] = df_val['Monitoramento'].apply(limpar_numero)
    df_val['Dias_Passados'] = pd.to_numeric(df_val['Dias_Passados'].apply(limpar_numero), errors='coerce').fillna(0).astype(int)
    df_val['Dias_Ate_Alvo'] = df_val['Dias_Ate_Alvo'].apply(
        lambda x: round(limpar_numero(x), 2) if str(x).strip() not in ("", "nan", "None") else ""
    )
    df_val['Preco_Final'] = df_val['Preco_Final'].apply(
        lambda x: round(limpar_numero(x), 4) if str(x).strip() not in ("", "nan", "None") else ""
    )
    df_val['Estado_RL'] = df_val['Estado_RL'].astype(str).apply(lambda x: x.strip().replace("'", ""))
    df_val['Estado_RL'] = df_val['Estado_RL'].apply(lambda x: f"'{x}" if x != "" else "")

    ok_val = salvar_aba_atomica(sh, "Sinais_IA", HEADERS_SINAIS_IA, df_val.values.tolist(),
                                 rows_est=10000, cols_est=len(HEADERS_SINAIS_IA))
    if ok_val:
        ws_val = sh.worksheet("Sinais_IA")
    else:
        print("⚠️ Sinais_IA não foi atualizada neste ciclo — dados anteriores preservados.")

    # ------------------------------------------------------------------
    # Recompensa combinada do Cérebro RL: preço realizado + concordância API
    # ------------------------------------------------------------------
    if indices_concluidos_neste_ciclo:
        df_reforco = df_val.loc[indices_concluidos_neste_ciclo].copy()
        df_reforco = df_reforco[df_reforco['Estado_RL'].astype(str).str.strip() != ""].copy()

        for _, row_r in df_reforco.iterrows():
            st_rl = str(row_r['Estado_RL']).strip().replace("'", "")
            acao_rl = row_r['Tipo']
            status_rl = row_r['Status']

            preco_sinal_r = limpar_numero(row_r['Preco_Sinal'])
            preco_final_r = limpar_numero(row_r['Preco_Final'])
            lucro_desejado_pct_r = limpar_numero(row_r.get('Lucro_Desejado_%', ''))
            if lucro_desejado_pct_r <= 0:
                lucro_desejado_pct_r = LUCRO_DESEJADO_MINIMO * 100

            if preco_final_r > 0:
                roi_real_pct = calcular_roi_real(acao_rl, preco_sinal_r, preco_final_r)
                recompensa_preco = max(-2.0, min(2.0, roi_real_pct / lucro_desejado_pct_r))
            elif "🏆" in status_rl:
                recompensa_preco = 1.0
            elif "💰" in status_rl:
                recompensa_preco = 0.5
            else:
                recompensa_preco = -1.0

            recompensa_api = MAPA_CONCORDA_NUM.get(str(row_r.get('Concorda_API', 'NEUTRO')).strip(), 0.0)
            recompensa = max(-2.0, min(2.0,
                              PESO_RECOMPENSA_PRECO * recompensa_preco + PESO_RECOMPENSA_API * recompensa_api))

            if st_rl not in q_table:
                q_table[st_rl] = {"COMPRA": 0.0, "VENDA": 0.0, "N_COMPRA": 0, "N_VENDA": 0}

            chave_n = f"N_{acao_rl}"
            n_visitas = q_table[st_rl].get(chave_n, 0) + 1
            alpha_dinamico = max(ALPHA_MINIMO, 1.0 / (1 + n_visitas))

            q_atual = q_table[st_rl].get(acao_rl, 0.0)
            q_table[st_rl][acao_rl] = q_atual + alpha_dinamico * (recompensa - q_atual)
            q_table[st_rl][chave_n] = n_visitas

            chave_item_estado = f"{row_r['Item']}|{st_rl}"
            if chave_item_estado not in q_table_item:
                q_table_item[chave_item_estado] = {"COMPRA": 0.0, "VENDA": 0.0, "N_COMPRA": 0, "N_VENDA": 0}

            n_visitas_item = q_table_item[chave_item_estado].get(chave_n, 0) + 1
            alpha_item = max(ALPHA_MINIMO, 1.0 / (1 + n_visitas_item))
            q_atual_item = q_table_item[chave_item_estado].get(acao_rl, 0.0)
            q_table_item[chave_item_estado][acao_rl] = q_atual_item + alpha_item * (recompensa - q_atual_item)
            q_table_item[chave_item_estado][chave_n] = n_visitas_item

        print(f"🧠 Cérebro RL atualizado: {len(df_reforco)} novos estados processados "
              f"(recompensa combinada: {PESO_RECOMPENSA_PRECO:.0%} preço + {PESO_RECOMPENSA_API:.0%} API).")

n_obs_rl_map = contar_obs_por_estado(df_val)
mapa_mult_desvio, mapa_veto, mapa_ajuste_roi, df_audit_cache = construir_calibracao_hibrida(
    sh, df_val_mem=df_val, q_table_mem=q_table
)

# ==============================================================================
# 2. INDICADORES (via cache incremental)
# ==============================================================================
cache_indicadores_persistido = carregar_cache_indicadores()

cache_indicadores = {}
for item in df_bruto['Item'].unique():
    try:
        d_it = preparar_indicadores_incremental(df_bruto, item, cache_indicadores_persistido)
        if d_it.empty:
            continue
        cache_indicadores[item] = d_it
        cache_indicadores_persistido[item] = {
            'df': cache_indicadores_persistido.get(item, {}).get('df', d_it),
            'ultimo_ts': d_it['Timestamp'].max() if not d_it.empty else None
        }
    except Exception as e:
        print(f"⚠️ Indicadores falharam para '{item}': {e}")
        continue

# ==============================================================================
# 3. MODELO POOLED (treino diário, inferência todo ciclo)
# ==============================================================================
df_treino_pooled, df_teste_pooled = montar_datasets_pooled(cache_indicadores)

retreinou_neste_ciclo = False
if deve_retreinar_pooled():
    retreinou_neste_ciclo = True
    print("🏋️ Retreinando modelo pooled — cadência diária...")
    modelo_pooled = treinar_modelo_pooled(df_treino_pooled)
    categorias_item = sorted(df_treino_pooled['Item_Cat'].unique().tolist()) if not df_treino_pooled.empty else None
    if modelo_pooled is not None:
        salvar_modelo_pooled(modelo_pooled, categorias_item, X_COLS_POOLED + ['Item_Cat'])
else:
    modelo_pooled, categorias_item = carregar_modelo_pooled()
    if modelo_pooled is None and not df_treino_pooled.empty:
        retreinou_neste_ciclo = True
        modelo_pooled = treinar_modelo_pooled(df_treino_pooled)
        categorias_item = sorted(df_treino_pooled['Item_Cat'].unique().tolist())
        if modelo_pooled is not None:
            salvar_modelo_pooled(modelo_pooled, categorias_item, X_COLS_POOLED + ['Item_Cat'])

# --- Métricas de auditoria: só quando o modelo foi retreinado neste ciclo ---
# Com o bot rodando a cada 15 min, gravar a auditoria em todo ciclo enchia a
# XGB_Audit 4x mais rápido sem informação nova (o modelo só muda 1x/dia) e,
# com a retenção atual, deixava pouco mais de 1 dia de histórico.
audit_linhas = []
if retreinou_neste_ciclo and modelo_pooled is not None and not df_teste_pooled.empty:
    cols_modelo = X_COLS_POOLED + ['Item_Cat']
    pred_teste = modelo_pooled.predict(df_teste_pooled[cols_modelo])
    df_teste_eval = df_teste_pooled.copy()
    df_teste_eval['Pred'] = pred_teste

    for item_aud, grupo in df_teste_eval.groupby('Item_Cat', observed=True):
        if len(grupo) < 5:
            continue
        erros = grupo['Pred'].values - grupo['Preco_Futuro'].values
        mae = float(np.mean(np.abs(erros)))
        rmse = float(np.sqrt(np.mean(erros ** 2)))
        preco_base = grupo['Price'].values
        direcao_real = (grupo['Preco_Futuro'].values > preco_base).astype(int)
        direcao_pred = (grupo['Pred'].values > preco_base).astype(int)
        acerto_direcao = float(np.mean(direcao_real == direcao_pred) * 100)
        roi_medio_previsto = float(np.mean((grupo['Pred'].values - preco_base) / preco_base * 100))
        n_treino_item = int((df_treino_pooled['Item_Cat'] == item_aud).sum()) if not df_treino_pooled.empty else 0

        audit_linhas.append([
            obter_agora_brt().strftime("%d/%m/%Y %H:%M"), str(item_aud),
            round(mae, 4), round(rmse, 4), round(acerto_direcao, 1),
            n_treino_item, len(grupo), round(roi_medio_previsto, 2)
        ])

# --- Ranking / inferência de produção (última linha de cada item) ---
ranking = []
itens_no_modelo = set(categorias_item) if categorias_item else set()

for item in df_bruto['Item'].unique():
    d_it = cache_indicadores.get(item)
    if d_it is None or d_it.empty or modelo_pooled is None or item not in itens_no_modelo:
        continue
    if d_it[X_COLS_BASE].dropna().empty:
        continue

    linha_pred = d_it[X_COLS_POOLED].tail(1).copy()
    linha_pred['Item_Cat'] = pd.Categorical([item], categories=list(categorias_item))
    pred = modelo_pooled.predict(linha_pred[X_COLS_POOLED + ['Item_Cat']])[0]
    preco_atual = d_it.iloc[-1]['Price']
    if preco_atual <= 0:
        continue

    roi = ((pred - preco_atual) / preco_atual) * 100
    roi_ajustado = roi + mapa_ajuste_roi.get(item, 0.0)
    ranking.append({'Item': item, 'ROI': roi, 'ROI_Ajustado': roi_ajustado})

df_ranking = (
    pd.DataFrame(ranking).sort_values(by='ROI_Ajustado', ascending=False)
    if ranking else pd.DataFrame(columns=['Item', 'ROI', 'ROI_Ajustado'])
)

# ==============================================================================
# 4. GERAÇÃO DE SINAIS (sem estoque — todo item é avaliado todo ciclo)
# ==============================================================================
try:
    max_id_existente = int(pd.to_numeric(df_val['ID_Sinal'], errors='coerce').max()) if not df_val.empty else 0
except Exception:
    max_id_existente = len(df_val)
prox_id_sinal = max_id_existente + 1
novas_linhas_sinais = []

mapa_ultimo_sinal = {}
linhas_ignoradas_dedup = 0
if not df_val.empty:
    for _, row_u in df_val.iterrows():
        try:
            chave_u = f"{str(row_u['Item']).strip()}_{str(row_u['Tipo']).strip().upper()}"
            ts_u = parse_data_sinal_robusta(row_u['Data_Sinal'])
            if pd.isnull(ts_u):
                continue
            preco_u = limpar_numero(row_u['Preco_Sinal'])
            if chave_u not in mapa_ultimo_sinal or ts_u > mapa_ultimo_sinal[chave_u]['timestamp']:
                mapa_ultimo_sinal[chave_u] = {'preco': preco_u, 'timestamp': ts_u}
        except Exception:
            linhas_ignoradas_dedup += 1
            continue

if linhas_ignoradas_dedup > 0:
    print(f"⚠️  {linhas_ignoradas_dedup} linha(s) de Sinais_IA ignorada(s) ao montar mapa_ultimo_sinal "
          f"(dado corrompido) — dedup de 4h/3% seguiu normal para as demais linhas.")

for item in df_ranking['Item']:
    d_it = cache_indicadores.get(item)
    if d_it is None or d_it.empty:
        continue
    row = d_it.iloc[-1]
    if pd.isna(row.get('Media_Macro')) or pd.isna(row.get('Desvio_Macro')):
        continue
    roi_pred = df_ranking[df_ranking['Item'] == item]['ROI_Ajustado'].values[0]

    preco_atual = row['Price']
    m_desvio = mapa_mult_desvio.get(item, 2.0)
    alvo_in = row['Media_Macro'] - (m_desvio * row['Desvio_Macro'])
    alvo_out = row['Media_Macro'] + (m_desvio * row['Desvio_Macro'])
    margem_atencao = calcular_margem_atencao(alvo_in, alvo_out)

    em_crash, var24h = detectar_regime(d_it)

    estado_rl = construir_estado_rl_craft(row, roi_pred, alvo_in, alvo_out, margem_atencao, em_crash)
    estado_rl_continuo = construir_estado_rl_continuo_craft(row, roi_pred, alvo_in, alvo_out, var24h)

    acao = "ESPERAR"
    if preco_atual <= alvo_in:
        acao = "🟢 COMPRAR"
    elif preco_atual <= alvo_in + margem_atencao:
        acao = "👀 PERTO DA ENTRADA"
    elif preco_atual >= alvo_out:
        acao = "🔴 VENDER"
    elif preco_atual >= alvo_out - margem_atencao:
        acao = "🎯 PERTO DA SAÍDA"

    veto_manual = bool(mapa_veto.get(item, False))
    if acao == "🟢 COMPRAR" and veto_manual:
        acao = "🚫 COMPRA VETADA"

    roi_pred_exibido = roi_pred
    vetado_rl = False
    if acao == "🟢 COMPRAR":
        chave_ci_atual = f"{item}|{estado_rl}"
        q_item_dados = q_table_item.get(chave_ci_atual, {})
        q_compra_item = q_item_dados.get("COMPRA")
        n_obs_item_at = q_item_dados.get("N_COMPRA", 0)

        vetado_rl, bonus_roi_rl = calcular_veto_rl(
            estado_rl, q_table, n_obs_rl_map, q_compra_item=q_compra_item, n_obs_item=n_obs_item_at
        )
        roi_pred_exibido = roi_pred + bonus_roi_rl
        if vetado_rl:
            acao = "🚫 VETO DO CÉREBRO RL (Histórico de Prejuízo)"

    if em_crash and "COMPRAR" in acao:
        acao = f"⚠️ COMPRAR (QUEDA {var24h:.1f}% em 24h)"

    if acao in ["🟢 COMPRAR", "🔴 VENDER"] or (em_crash and "COMPRAR" in acao):
        agora_brt = obter_agora_brt()
        dt_hoje_str = agora_brt.strftime("%d%m%Y")
        dt_hhmm = agora_brt.strftime("%H%M")
        tipo_sinal = "COMPRA" if "COMPRAR" in acao else "VENDA"

        INTERVALO_MIN_H = 4
        VARIACAO_MIN_PCT = 0.03

        chave_item_tipo = f"{item}_{tipo_sinal}"
        ultimo = mapa_ultimo_sinal.get(chave_item_tipo)
        pode_gerar = True

        if ultimo:
            horas_desde = (agora_brt - ultimo['timestamp']).total_seconds() / 3600
            variacao = abs(preco_atual - ultimo['preco']) / ultimo['preco'] if ultimo['preco'] > 0 else 1.0
            if horas_desde >= 24:
                pode_gerar = True
            elif horas_desde < INTERVALO_MIN_H:
                pode_gerar = False
            elif variacao < VARIACAO_MIN_PCT:
                pode_gerar = False

        chave_unica = f"{item}_{tipo_sinal}_{dt_hoje_str}_{dt_hhmm}"

        if pode_gerar and not any(x[13] == chave_unica for x in novas_linhas_sinais):
            ref_financeira = alvo_out if tipo_sinal == "COMPRA" else row['Media_Macro']
            recomendacao_raw = row.get('Recomendacao_API', '')
            concorda_label, _ = avaliar_concordancia_api(tipo_sinal, recomendacao_raw)
            # Congela a meta de lucro dinâmica deste momento — validação e
            # recompensa do RL vão reusar este mesmo valor depois.
            lucro_desejado_pct = round(
                calcular_lucro_desejado_dinamico(row['Desvio_Macro'], row['Media_Macro']) * 100, 2
            )

            enviar_telegram(acao, item, preco_atual, prox_id_sinal)

            novas_linhas_sinais.append([
                int(prox_id_sinal), tipo_sinal, agora_brt.strftime("%d/%m/%Y %H:%M"), item,
                round(preco_atual, 4), round(ref_financeira, 4), lucro_desejado_pct,
                recomendacao_raw, concorda_label,
                round(preco_atual, 4), "Aguardando", 0, f"'{estado_rl}", chave_unica,
                "", "", estado_rl_continuo
            ])
            prox_id_sinal += 1
            mapa_ultimo_sinal[chave_item_tipo] = {'preco': preco_atual, 'timestamp': agora_brt}

if novas_linhas_sinais:
    ws_val.append_rows(novas_linhas_sinais, value_input_option='USER_ENTERED')
    print(f"📌 {len(novas_linhas_sinais)} novos sinais integrados para monitorização (Sinais_IA).")

# ==============================================================================
# 5. PERSISTÊNCIA — XGB_Audit, Cerebro_RL, Cerebro_RL_Item
# ==============================================================================
try:
    try:
        ws_audit = sh.worksheet("XGB_Audit")
    except Exception:
        n_itens_init = df_bruto['Item'].nunique()
        rows_init = n_itens_init * 72 * 2 + 500
        ws_audit = sh.add_worksheet(title="XGB_Audit", rows=str(rows_init), cols="8")
        ws_audit.update(values=[HEADERS_AUDIT], range_name='A1', value_input_option='USER_ENTERED')
        set_frozen(ws_audit, rows=1)
        format_cell_range(ws_audit, 'A1:H1', CellFormat(textFormat=TextFormat(bold=True)))

    JANELA_AUDIT_H = JANELA_EVENTO_H   # mesmo ciclo de 3 dias das masterpieces
    n_itens_reais = df_bruto['Item'].nunique()
    RETENCAO_AUDIT = n_itens_reais * JANELA_AUDIT_H * 2

    if audit_linhas:
        dados_atuais = ws_audit.get_all_values()
        header = dados_atuais[0] if dados_atuais else HEADERS_AUDIT
        linhas_existentes = dados_atuais[1:] if len(dados_atuais) > 1 else []
        todas = linhas_existentes + audit_linhas
        if len(todas) > RETENCAO_AUDIT:
            todas = todas[-RETENCAO_AUDIT:]
        ws_audit.clear()
        ws_audit.update(values=([header] + todas), range_name='A1', value_input_option='USER_ENTERED')
        print(f"📊 XGB_Audit: {len(audit_linhas)} novos | {len(todas)} retidos.")
    else:
        print("📊 XGB_Audit: nenhum item com dados suficientes para auditoria.")
except Exception as e:
    print(f"❌ Erro ao salvar XGB_Audit: {e}")

try:
    cerebro_linhas = [
        [f"'{st}", round(acoes.get("COMPRA", 0.0), 4), round(acoes.get("VENDA", 0.0), 4),
         int(acoes.get("N_COMPRA", 0)), int(acoes.get("N_VENDA", 0))]
        for st, acoes in q_table.items()
    ]
    ok_cerebro = salvar_aba_atomica(sh, "Cerebro_RL", ["Estado_RL", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"],
                                     cerebro_linhas, rows_est=5000, cols_est=5)
    print("✅ Cerebro_RL sincronizado!" if ok_cerebro else "⚠️ Cerebro_RL não atualizado neste ciclo.")
except Exception as e:
    print(f"❌ Erro ao salvar Cerebro_RL: {e}")

try:
    cerebro_item_linhas = [
        [f"'{chave_ci}", round(acoes.get("COMPRA", 0.0), 4), round(acoes.get("VENDA", 0.0), 4),
         int(acoes.get("N_COMPRA", 0)), int(acoes.get("N_VENDA", 0))]
        for chave_ci, acoes in q_table_item.items()
    ]
    ok_cerebro_item = salvar_aba_atomica(sh, "Cerebro_RL_Item",
                                          ["Item_Estado", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"],
                                          cerebro_item_linhas, rows_est=20000, cols_est=5)
    print("✅ Cerebro_RL_Item sincronizado!" if ok_cerebro_item else "⚠️ Cerebro_RL_Item não atualizado.")
except Exception as e:
    print(f"❌ Erro ao salvar Cerebro_RL_Item: {e}")

# ==============================================================================
# 6. SALVAR CACHE DE INDICADORES EM DISCO (para o próximo ciclo)
# ==============================================================================
try:
    salvar_cache_indicadores(cache_indicadores_persistido)
    print("💾 Cache incremental de indicadores salvo em disco.")
except Exception as e:
    print(f"⚠️ Falha ao salvar cache de indicadores: {e}")

print("🏁 Ciclo (Craft World) finalizado.")
