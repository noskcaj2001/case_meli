"""
Regulatory Tracker — Mercado Pago
Dashboard executivo para acompanhamento do Índice Bacen.
Executar: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# ── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Regulatory Tracker · Mercado Pago",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Remove padding excessivo do topo */
.block-container { padding-top: 1.5rem; }

/* Banner de status */
.status-banner {
    padding: 10px 18px;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.status-risco  { background: #fce8e6; color: #c5221f; border-left: 4px solid #d93025; }
.status-alerta { background: #fef7e0; color: #b06000; border-left: 4px solid #f29900; }
.status-ok     { background: #e6f4ea; color: #137333; border-left: 4px solid #1e8e3e; }

/* Cards de explicação */
.info-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #444;
    margin-bottom: 0.5rem;
    border-left: 3px solid #dadce0;
}

/* Legenda compacta do benchmark */
.bench-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    padding: 4px 0;
}

/* Tabela da fila — destaque de urgência */
.urgencia-critica { color: #d93025; font-weight: 700; }
.urgencia-alta    { color: #f29900; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Constantes de negócio ────────────────────────────────────────────────────
INDICE_ATUAL = 29.79
INDICE_META  = 20.00
PROCEDENTES_ATUAIS = 1_862
CLIENTES_MP = 62_500_000

BENCHMARKS = {
    'Nubank':      {'indice': 13.90, 'cor': '#1e8e3e'},
    'Banco Inter': {'indice': 24.10, 'cor': '#f29900'},
    'PicPay':      {'indice': 35.20, 'cor': '#9334e6'},
    'C6 Bank':     {'indice': 64.30, 'cor': '#d93025'},
    'PagBank':     {'indice': 68.50, 'cor': '#795548'},
}

# ── Caminhos ─────────────────────────────────────────────────────────────────
OUTPUTS = os.path.join(os.path.dirname(__file__), '..', 'outputs')


# ── Funções de dados ─────────────────────────────────────────────────────────
@st.cache_data
def carregar_scores():
    path = os.path.join(OUTPUTS, '02_scores.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['INCOMING_DTTM'])
        if 'id' not in df.columns:
            df['id'] = range(1000, 1000 + len(df))
        return df
    np.random.seed(42)
    n = 800
    canais = [
        'BR_Chat', 'BR_SOS_MP_IVR', 'BR_MPCartões_Offline',
        'BR_MPContas_Security_Offline', 'OUV - RDR/Bacen - Acionamento',
        'BR_FBM_Seller_Offline_Premium', 'BR_C2C',
    ]
    datas = pd.date_range('2025-09-01', '2025-11-30', periods=n)
    return pd.DataFrame({
        'id': np.random.randint(10_000, 99_999, n),
        'INCOMING_DTTM': datas,
        'score_risco': np.clip(np.random.beta(2, 5, n) * 100, 0, 100).round(1),
        'abriu_rdr_48h': np.random.binomial(1, 0.08, n),
        'CX_TEAM_NAME': np.random.choice(canais, n),
    })


@st.cache_data
def carregar_projecao():
    path = os.path.join(OUTPUTS, '01_projecao_mp.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['ds'])
        # CSV real tem indice_pessimista / indice_base / indice_otimista
        col_base = next((c for c in df.columns if 'base' in c.lower()), None)
        col_pess = next((c for c in df.columns if 'pess' in c.lower()), None)
        col_otim = next((c for c in df.columns if 'otim' in c.lower()), None)
        if col_base:
            df['indice_base']       = df[col_base].clip(lower=0)
            df['indice_pessimista'] = df[col_pess].clip(lower=0) if col_pess else df['indice_base'] * 1.4
            df['indice_otimista']   = df[col_otim].clip(lower=0) if col_otim else df['indice_base'] * 0.7
        elif 'indice' in df.columns:
            df['indice_base']       = df['indice'].clip(lower=0)
            df['indice_pessimista'] = (df['indice'] * 1.4).clip(lower=0)
            df['indice_otimista']   = (df['indice'] * 0.7).clip(lower=0)
        return df
    # Sintético com 3 cenários
    datas = pd.date_range('2025-09-01', '2027-06-01', freq='MS')
    n = len(datas)
    trend_base = np.linspace(0, -10, n)
    noise = np.random.normal(0, 0.4, n)
    base = INDICE_ATUAL + trend_base + noise
    return pd.DataFrame({
        'ds': datas,
        'indice_base':       np.clip(base, 8, 60),
        'indice_pessimista': np.clip(base * 1.5, 8, 80),
        'indice_otimista':   np.clip(base * 0.65, 8, 60),
    })


@st.cache_data
def carregar_clusters():
    path = os.path.join(OUTPUTS, '03_cluster_analise.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({
        'cluster': [0, 1, 2, 3, 4, 5, 6],
        'produto': ['Conta', 'Conta', 'Cartão de Crédito',
                    'Conta', 'Cartão de Crédito', 'Consumer Credits', 'Pix'],
        'label_llm': [
            'Reclamações Bacen MP', 'Fraudes em pagamentos MP',
            'Problemas com pagamentos', 'Reclamações Procedentes Bacen',
            'Reclamações Cartão de Crédito', 'Reclamações de Créditos', 'Fraude PIX',
        ],
        'urgencia': ['Alta'] * 7,
        'causa_raiz_llm': [
            'Bugs e gestão inadequada de casos com impacto em LGPD.',
            'Fraudes em pagamentos e transferências sem segurança adequada.',
            'Pagamentos não reconhecidos e falhas de comunicação.',
            'Restrições de conta sem justificativa ou comunicação.',
            'Problemas em consultas de resumo e funcionalidades do cartão.',
            'Atrasos e bugs no módulo de créditos.',
            'Transações PIX realizadas sem consentimento do titular.',
        ],
        'acao_recomendada': [
            'Reduzir bugs de plataforma; reforçar fluxo de cancelamento.',
            'MFA em pagamentos acima de R$1k + monitoramento em tempo real.',
            'Melhorar comunicação de status; suporte proativo.',
            'Avisar cliente antes do bloqueio; comunicar motivo e prazo.',
            'Melhorar interface de resumo; treinamento de atendimento.',
            'Corrigir bugs no módulo de crédito; melhorar SLA.',
            'MFA reforçado para PIX + alertas de atividade suspeita.',
        ],
        'volume_procedentes': [81, 68, 59, 37, 39, 43, 30],
        'pct_do_indice': [22.7, 19.0, 16.5, 10.4, 10.9, 12.0, 8.4],
    })


# ── Carregar dados ────────────────────────────────────────────────────────────
df_scores   = carregar_scores()
df_proj     = carregar_projecao()
df_clusters = carregar_clusters()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), '..', 'mercado-pago-logo-png_seeklogo-198430.png')
    if os.path.exists(logo_path):
        st.image(logo_path, width=160)
    else:
        st.markdown("### Mercado Pago")

    st.markdown("## Regulatory Tracker")
    st.caption("Monitoramento do Índice Bacen em tempo real")
    st.divider()

    # ── Filtro de período ────────────────────────────────────────────────────
    st.markdown("#### Período de análise")
    periodo = st.radio(
        "Selecione o intervalo",
        ["Últimos 30 dias", "Últimos 90 dias", "Todo o histórico"],
        index=1,
        help="Filtra todos os dados de acionamentos e scores pelo período escolhido.",
    )

    st.divider()

    # ── Filtro de score ──────────────────────────────────────────────────────
    st.markdown("#### Fila de interceptação")
    score_threshold = st.slider(
        "Score mínimo de risco",
        min_value=50, max_value=95, value=70, step=5,
        help="Clientes com score acima deste valor entram na fila de contato proativo. "
             "Score mais alto = fila menor e mais precisa.",
    )
    st.caption(f"Score 0–100 · Quanto maior, maior o risco de escalar ao Bacen.")

    st.divider()

    # ── Referência de mercado ────────────────────────────────────────────────
    st.markdown("#### Referência de mercado (Q4/2025)")
    dados_bench = [
        ("🟢", "Nubank",      13.90, "Meta de referência"),
        ("🟡", "Banco Inter", 24.10, "Próximo concorrente"),
        ("🔴", "Mercado Pago", INDICE_ATUAL, "Posição atual"),
        ("🎯", "Nossa meta",   INDICE_META,  "Objetivo do ano"),
    ]
    for emoji, nome, valor, desc in dados_bench:
        st.markdown(
            f'<div class="bench-item">'
            f'{emoji} <strong>{nome}</strong>&nbsp;&nbsp;'
            f'<span style="color:#5f6368">{valor:.2f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.caption(desc)

    st.divider()
    st.caption("🕐 Atualização: D-1 · Score: batch/1h")


# ── Aplicar filtro de período ─────────────────────────────────────────────────
data_max = df_scores['INCOMING_DTTM'].max()
if periodo == "Últimos 30 dias":
    data_inicio = data_max - pd.Timedelta(days=30)
elif periodo == "Últimos 90 dias":
    data_inicio = data_max - pd.Timedelta(days=90)
else:
    data_inicio = df_scores['INCOMING_DTTM'].min()

df_filtrado = df_scores[df_scores['INCOMING_DTTM'] >= data_inicio].copy()

# Métricas derivadas do período filtrado
n_alto_risco  = len(df_filtrado[df_filtrado['score_risco'] >= score_threshold])
score_medio   = df_filtrado[df_filtrado['score_risco'] >= score_threshold]['score_risco'].mean()
score_medio   = score_medio if not np.isnan(score_medio) else 0.0
pct_alto      = n_alto_risco / max(len(df_filtrado), 1)
procedentes_periodo = int(df_filtrado['abriu_rdr_48h'].sum()) if 'abriu_rdr_48h' in df_filtrado else 357
taxa_interceptacao = 0.42  # KPI operacional — simulado

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.markdown("## 📊 Regulatory Tracker — Índice Bacen")
st.markdown(
    "Acompanhamento do ranking regulatório do Mercado Pago junto ao "
    "**Banco Central do Brasil**. Meta: atingir a **15ª posição** no ranking público de reclamações."
)

# Banner de status dinâmico
if INDICE_ATUAL > 30:
    st.markdown(
        '<div class="status-banner status-risco">'
        '🔴 <strong>Atenção:</strong> Índice atual de <strong>29,79</strong> está '
        'acima da meta de 20,00. Ação imediata necessária nas 4 frentes prioritárias.'
        '</div>',
        unsafe_allow_html=True,
    )
elif INDICE_ATUAL > INDICE_META:
    st.markdown(
        '<div class="status-banner status-alerta">'
        '🟡 <strong>Em progresso:</strong> Índice em queda, mas ainda acima da meta de 20,00.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-banner status-ok">'
        '🟢 <strong>Meta atingida!</strong> Índice abaixo de 20,00.'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Tabs de navegação ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Visão Executiva", "🎯 Fila Operacional", "🔍 Causa Raiz"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — VISÃO EXECUTIVA
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── O que é o índice Bacen? ───────────────────────────────────────────────
    with st.expander("ℹ️ O que é o Índice Bacen e por que acompanhamos?", expanded=False):
        st.markdown("""
**O Índice Bacen** mede quantas reclamações de clientes foram consideradas **procedentes** (legítimas)
pelo Banco Central do Brasil a cada 1 milhão de clientes.

```
Índice = (Reclamações Procedentes ÷ Base de Clientes) × 1.000.000
```

**Por que importa?**
- 📋 O Bacen publica esse ranking trimestralmente — é público e consultado por consumidores
- 🏆 Bancos e fintechs com índice menor têm vantagem de reputação
- ⚖️ Índice alto pode atrair fiscalização regulatória
- 🎯 **Nossa meta:** sair da ~29ª para a **15ª posição** até o fim de 2026

**O que é uma reclamação procedente?** É quando um cliente aciona o Bacen (via RDR)
e o regulador entende que a reclamação era legítima — ou seja, o MP não resolveu corretamente.
        """)

    st.markdown("### Indicadores-chave")

    # ── 4 KPIs ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=INDICE_ATUAL,
            delta={
                "reference": INDICE_META,
                "increasing": {"color": "#d93025"},
                "decreasing": {"color": "#1e8e3e"},
                "suffix": " vs meta",
            },
            gauge={
                "axis": {"range": [0, 70], "tickwidth": 1, "tickcolor": "#5f6368"},
                "bar": {"color": "#1a73e8", "thickness": 0.25},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, INDICE_META],   "color": "#e6f4ea"},
                    {"range": [INDICE_META, 25],  "color": "#fef7e0"},
                    {"range": [25, 70],           "color": "#fce8e6"},
                ],
                "threshold": {
                    "line": {"color": "#1e8e3e", "width": 3},
                    "thickness": 0.75,
                    "value": INDICE_META,
                },
            },
            title={"text": "Índice Bacen Atual<br><span style='font-size:11px;color:#5f6368'>Meta: ≤ 20,00</span>",
                   "font": {"size": 13}},
            number={"font": {"size": 36}, "valueformat": ".2f"},
        ))
        fig_gauge.update_layout(
            height=230,
            margin=dict(t=40, b=5, l=15, r=15),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with k2:
        delta_proc = procedentes_periodo - 395
        st.metric(
            label="Reclamações Procedentes",
            value=f"{procedentes_periodo:,}",
            delta=f"{delta_proc:+d} vs período anterior",
            delta_color="inverse",
            help=(
                "Reclamações julgadas PROCEDENTES pelo Bacen no período selecionado. "
                "Quanto menor, melhor. Meta mensal: ≤ 310."
            ),
        )
        st.caption(f"🎯 Meta mensal: ≤ 310 · Período: {periodo}")

    with k3:
        fila_label = "—" if n_alto_risco == 0 else f"{score_medio:.1f}"
        st.metric(
            label=f"Score Médio na Fila (≥ {score_threshold})",
            value=fila_label,
            delta=f"{n_alto_risco:,} clientes aguardando contato",
            delta_color="off",
            help=(
                f"Score médio dos clientes com risco ≥ {score_threshold} — "
                "esses clientes precisam de contato proativo antes de abrirem uma reclamação no Bacen. "
                "O score vai de 0 (baixo risco) a 100 (alto risco)."
            ),
        )
        st.caption(f"📊 {pct_alto:.1%} da base no período selecionado")

    with k4:
        st.metric(
            label="Taxa de Interceptação",
            value=f"{taxa_interceptacao:.0%}",
            delta="dos alto-risco contactados",
            delta_color="off",
            help=(
                "Percentual dos clientes com score alto que já foram contactados proativamente. "
                "Meta: ≥ 60% de cobertura para reduzir o número de RDRs abertos."
            ),
        )
        st.progress(taxa_interceptacao)
        st.caption("🎯 Meta: ≥ 60% de cobertura")

    st.divider()

    # ── Gráfico de projeção ───────────────────────────────────────────────────
    st.markdown("### Trajetória do Índice Bacen — 3 Cenários")
    st.caption(
        "Modelo de **janela móvel de 12 meses** ancorado no índice oficial Q4/2025 = **29,79**. "
        "Os 4.734 casos pendentes se resolvem em H1/2026 — causando um pico — antes de cair em 2027 "
        "quando o choque sai da janela."
    )

    col_base = 'indice_base'       if 'indice_base'       in df_proj.columns else 'indice'
    col_pess = 'indice_pessimista' if 'indice_pessimista' in df_proj.columns else col_base
    col_otim = 'indice_otimista'   if 'indice_otimista'   in df_proj.columns else col_base

    # Âncora: Q4/2025 = 29,79 (último trimestre conhecido)
    ANCHOR_DATE = pd.Timestamp('2025-12-31')
    PROJ_FIM    = pd.Timestamp('2027-07-01')
    df_fut = df_proj[df_proj['ds'] >= ANCHOR_DATE].copy()

    # Prepend do ponto de ancoragem (29,79) para todas as séries
    row_ancora = pd.DataFrame({
        'ds': [ANCHOR_DATE], 'periodo': ['Q4/2025'],
        col_base: [INDICE_ATUAL], col_pess: [INDICE_ATUAL], col_otim: [INDICE_ATUAL],
    })
    df_chart = pd.concat([row_ancora, df_fut[df_fut['ds'] > ANCHOR_DATE]], ignore_index=True)

    # Histórico MP (Q1–Q4/2025) — dados oficiais Aba 3
    hist_ds  = [pd.Timestamp('2025-03-31'), pd.Timestamp('2025-06-30'),
                pd.Timestamp('2025-09-30'), pd.Timestamp('2025-12-31')]
    hist_idx = [31.18, 32.50, 30.10, 29.79]

    # Detectar cruzamentos por cenário
    def primeiro_cruzamento_q(df, col, threshold):
        for _, row in df.iterrows():
            if row.get('periodo', '') == 'Q4/2025':
                continue
            if row[col] <= threshold:
                periodo = row.get('periodo', row['ds'].strftime('%b/%Y'))
                return row['ds'], row[col], periodo
        return None, None, '> H1/2027'

    cruz_picpay_base = primeiro_cruzamento_q(df_chart, col_base, 35.20)
    cruz_picpay_otim = primeiro_cruzamento_q(df_chart, col_otim, 35.20)
    cruz_inter_base  = primeiro_cruzamento_q(df_chart, col_base, 24.10)
    cruz_meta_otim   = primeiro_cruzamento_q(df_chart, col_otim, INDICE_META)

    fig_proj = go.Figure()

    # ── Black Friday Q4/2026 ──────────────────────────────────────────────────
    fig_proj.add_vrect(
        x0='2026-10-01', x1='2026-12-31',
        fillcolor='rgba(230,100,0,0.10)', line_width=0,
        annotation_text='🔥 Black Friday 2026',
        annotation_position='top left',
        annotation_font_size=9,
        annotation_font_color='#e06000',
    )

    # ── Zonas de meta ─────────────────────────────────────────────────────────
    fig_proj.add_hrect(y0=INDICE_META, y1=85, fillcolor='rgba(217,48,37,0.04)', line_width=0)
    fig_proj.add_hrect(y0=0, y1=INDICE_META, fillcolor='rgba(30,142,62,0.06)', line_width=0)

    # ── Histórico MP ──────────────────────────────────────────────────────────
    fig_proj.add_trace(go.Scatter(
        x=hist_ds, y=hist_idx,
        name='MP — histórico (Q1–Q4/2025)',
        mode='lines+markers',
        line=dict(color='#2c3e50', width=2.5),
        marker=dict(size=9, color='#2c3e50', line=dict(color='white', width=2)),
        hovertemplate='<b>Histórico</b>  %{y:.2f}<extra></extra>',
    ))

    # ── Faixa de incerteza ────────────────────────────────────────────────────
    fig_proj.add_trace(go.Scatter(
        x=list(df_chart['ds']) + list(df_chart['ds'][::-1]),
        y=list(df_chart[col_pess]) + list(df_chart[col_otim][::-1]),
        fill='toself',
        fillcolor='rgba(26,115,232,0.07)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Intervalo Pessimista–Otimista',
        hoverinfo='skip',
    ))

    # ── Cenário pessimista ────────────────────────────────────────────────────
    fig_proj.add_trace(go.Scatter(
        x=df_chart['ds'], y=df_chart[col_pess],
        name='Pessimista — 70% viram procedentes (+3.314)',
        mode='lines+markers',
        line=dict(color='#c0392b', width=2, dash='dot'),
        marker=dict(size=7, color='#c0392b'),
        hovertemplate='<b>Pessimista</b>  %{y:.2f}<extra></extra>',
    ))

    # ── Cenário otimista ──────────────────────────────────────────────────────
    fig_proj.add_trace(go.Scatter(
        x=df_chart['ds'], y=df_chart[col_otim],
        name='Otimista — 20% viram procedentes (+947)',
        mode='lines+markers',
        line=dict(color='#27ae60', width=2, dash='dash'),
        marker=dict(size=7, color='#27ae60'),
        hovertemplate='<b>Otimista</b>  %{y:.2f}<extra></extra>',
    ))

    # ── Cenário base — linha principal ────────────────────────────────────────
    fig_proj.add_trace(go.Scatter(
        x=df_chart['ds'], y=df_chart[col_base],
        name='Base — 45% viram procedentes (+2.130)',
        mode='lines+markers',
        line=dict(color='#2980b9', width=3),
        marker=dict(size=9, color='#2980b9', line=dict(color='white', width=2)),
        hovertemplate='<b>Base</b>  %{y:.2f}<extra></extra>',
    ))

    # ── Ponto âncora ─────────────────────────────────────────────────────────
    fig_proj.add_annotation(
        x=ANCHOR_DATE, y=INDICE_ATUAL,
        text=f'<b>{INDICE_ATUAL}</b><br>Q4/2025<br>(âncora)',
        showarrow=True, arrowhead=2, arrowcolor='#2c3e50', ax=30, ay=-40,
        font=dict(size=10, color='#2c3e50'),
        bgcolor='rgba(255,255,255,0.9)', bordercolor='#2c3e50', borderpad=3,
    )

    # ── Separador histórico / projeção ────────────────────────────────────────
    fig_proj.add_vline(
        x=ANCHOR_DATE.timestamp() * 1000,
        line_dash='dot', line_color='#9e9e9e', line_width=1.5,
        annotation_text='← Histórico  |  Projeção →',
        annotation_position='top right',
        annotation_font_size=9, annotation_font_color='#757575',
    )

    # ── Benchmarks ────────────────────────────────────────────────────────────
    bench_relevante = [
        ('PicPay',      35.20, '#9334e6'),
        ('Banco Inter', 24.10, '#e65100'),
        ('Nubank',      13.90, '#1e8e3e'),
    ]
    for nome, valor, cor in bench_relevante:
        fig_proj.add_hline(
            y=valor,
            line_dash='dash', line_color=cor, line_width=1.3, opacity=0.70,
            annotation_text=f'  {nome} ({valor})',
            annotation_position='right',
            annotation_font_size=10, annotation_font_color=cor,
        )

    # ── Meta regulatória ──────────────────────────────────────────────────────
    fig_proj.add_hline(
        y=INDICE_META,
        line_color='#1e8e3e', line_dash='dot', line_width=2.5,
        annotation_text='  🎯 Meta 20,00',
        annotation_position='right',
        annotation_font_size=11, annotation_font_color='#1e8e3e',
    )

    # ── Anotações de cruzamento ───────────────────────────────────────────────
    for ds_cruz, val_cruz, periodo, rotulo, cor, ay_off in [
        (cruz_picpay_base[0], cruz_picpay_base[1], cruz_picpay_base[2],
         '✓ Base supera PicPay', '#9334e6', -45),
        (cruz_picpay_otim[0], cruz_picpay_otim[1], cruz_picpay_otim[2],
         '✓ Otimista supera PicPay', '#27ae60', -70),
    ]:
        if ds_cruz is not None:
            fig_proj.add_annotation(
                x=ds_cruz, y=val_cruz,
                text=f'<b>{rotulo}</b><br>{periodo}',
                showarrow=True, arrowhead=2, arrowcolor=cor,
                ax=0, ay=ay_off,
                font=dict(size=9, color=cor),
                bgcolor='white', bordercolor=cor, borderwidth=1, borderpad=4,
            )

    # ── Layout ────────────────────────────────────────────────────────────────
    y_max = max(df_chart[col_pess].max() * 1.12, 45)
    ticks_q = [
        pd.Timestamp('2025-03-31'), pd.Timestamp('2025-06-30'),
        pd.Timestamp('2025-09-30'), pd.Timestamp('2025-12-31'),
        pd.Timestamp('2026-03-31'), pd.Timestamp('2026-06-30'),
        pd.Timestamp('2026-09-30'), pd.Timestamp('2026-12-31'),
        pd.Timestamp('2027-03-31'), pd.Timestamp('2027-06-30'),
    ]
    ticks_lbl = [
        'Q1/25', 'Q2/25', 'Q3/25', 'Q4/25',
        'Q1/26', 'Q2/26', 'Q3/26', 'Q4/26',
        'Q1/27', 'Q2/27',
    ]
    fig_proj.update_layout(
        height=460,
        hovermode='x unified',
        xaxis_title='Trimestre',
        yaxis_title='Índice Bacen',
        legend=dict(
            orientation='h',
            yanchor='top', y=-0.16,
            xanchor='center', x=0.5,
            font=dict(size=9),
            bgcolor='rgba(255,255,255,0)',
        ),
        margin=dict(t=20, b=110, r=150, l=55),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True, gridcolor='#f0f0f0',
            range=[pd.Timestamp('2025-01-01'), PROJ_FIM],
            tickvals=ticks_q, ticktext=ticks_lbl,
            tickangle=0,
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#f0f0f0',
            range=[0, y_max],
        ),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # ── Tabela de ultrapassagem ───────────────────────────────────────────────
    with st.expander("📋 Em que trimestre o MP supera cada concorrente? (todos os cenários)", expanded=True):
        bench_cruzamentos = {
            'PicPay (35,20)':      (35.20, '#9334e6'),
            'Banco Inter (24,10)': (24.10, '#e65100'),
            'Nubank (13,90)':      (13.90, '#1e8e3e'),
        }
        linhas = []
        for inst, (thresh, _) in bench_cruzamentos.items():
            _, _, q_pess = primeiro_cruzamento_q(df_chart, col_pess, thresh)
            _, _, q_base = primeiro_cruzamento_q(df_chart, col_base, thresh)
            _, _, q_otim = primeiro_cruzamento_q(df_chart, col_otim, thresh)
            linhas.append({'Concorrente': inst,
                           'Pessimista (70%)': q_pess,
                           'Base (45%)':       q_base,
                           'Otimista (20%)':   q_otim})
        st.dataframe(pd.DataFrame(linhas).set_index('Concorrente'), use_container_width=True)
        st.caption(
            "Superar = índice MP cai abaixo do índice do concorrente (posição melhor no ranking). "
            "O modelo usa janela móvel de 12 meses com os 4.734 pendentes distribuídos em Q1–Q3/2026."
        )

    # ── Cards de milestone ────────────────────────────────────────────────────
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown(
            f'<div class="info-box">🟣 <strong>1ª meta: superar PicPay (35,20)</strong><br>'
            f'Cenário base: <strong>{cruz_picpay_base[2]}</strong><br>'
            f'Cenário otimista: <strong>{cruz_picpay_otim[2]}</strong></div>',
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f'<div class="info-box">🟠 <strong>2ª meta: superar Banco Inter (24,10)</strong><br>'
            f'Cenário base: <strong>{cruz_inter_base[2]}</strong><br>'
            f'Requer redução ativa de procedentes (Q4).</div>',
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f'<div class="info-box">🎯 <strong>Meta regulatória (≤ 20,00 · 15ª posição)</strong><br>'
            f'Cenário otimista: <strong>{cruz_meta_otim[2]}</strong><br>'
            f'Exige intervenção nas 4 frentes prioritárias.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Ranking de concorrentes visual ────────────────────────────────────────
    st.markdown("### Posicionamento no mercado")
    col_rank, col_acoes = st.columns([3, 2])

    with col_rank:
        rank_data = {
            'Instituição': ['PagBank', 'C6 Bank', 'PicPay', 'Mercado Pago', 'Banco Inter', 'Nubank', 'Meta MP'],
            'Índice':      [68.50,     64.30,     35.20,    29.79,          24.10,         13.90,    20.00],
            'Tipo':        ['Concorrente', 'Concorrente', 'Concorrente',
                            'Mercado Pago', 'Concorrente', 'Concorrente', 'Meta'],
        }
        df_rank = pd.DataFrame(rank_data).sort_values('Índice', ascending=True)

        cor_map = {
            'Mercado Pago': '#1a73e8',
            'Meta':         '#1e8e3e',
            'Concorrente':  '#dadce0',
        }

        fig_rank = px.bar(
            df_rank,
            x='Índice', y='Instituição',
            orientation='h',
            color='Tipo',
            color_discrete_map=cor_map,
            text='Índice',
        )
        fig_rank.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_rank.update_layout(
            height=300,
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=60),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Índice Bacen'),
            yaxis=dict(title=''),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    with col_acoes:
        st.markdown("**Plano de ação — impacto esperado**")
        st.markdown("""
| # | Frente | Redução |
|---|--------|---------|
| 1 | Interceptação proativa (score ≥ 70) | –10% |
| 2 | Fraude PIX — MFA + reversão <2h | –10% |
| 3 | Task force 4.734 pendentes | –6% |
| 4 | Comunicação de restrições de conta | –4% |
""")
        st.metric(
            "Impacto combinado estimado",
            "–30%",
            "→ índice ~20,85 · ~15ª posição",
            help="Com as 4 frentes executadas em 90 dias, o índice sai de 29,79 para ~20,85.",
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — FILA OPERACIONAL
# ════════════════════════════════════════════════════════════════════════════════
with tab2:

    with st.expander("ℹ️ O que é a fila de interceptação e como usar?", expanded=False):
        st.markdown(f"""
**O modelo de Machine Learning** analisa cada cliente que entrou em contato com o atendimento
e atribui um **score de risco de 0 a 100** — a probabilidade de esse cliente abrir uma
reclamação formal no Bacen nas **próximas 48 horas**.

**Como usar esta fila:**
- Clientes com score ≥ **{score_threshold}** (ajustável na barra lateral) estão nesta lista
- A equipe de CX deve contactá-los **proativamente**, antes que eles acionem o Bacen
- Prioridade: 🔴 **Crítico** (score ≥ 85) → retorno em até 1h · 🟡 **Alto** (score ≥ 70) → retorno em 4h

**Por que isso funciona?** Um cliente contactado antes de abrir o RDR tem ~55% menos chance
de registrar a reclamação formal. O custo de uma ligação proativa (R$15) é muito menor que o
custo de um caso procedente (R$350 estimado).
        """)

    # ── Filtros da fila ───────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 3])
    with fc1:
        urgencia_filtro = st.selectbox(
            "Filtrar por urgência",
            ["Todas", "🔴 Crítico (≥ 85)", "🟡 Alto (≥ 70 e < 85)"],
            help="Filtra a fila pelo nível de urgência do cliente.",
        )
    with fc2:
        canal_opcoes = ["Todos os canais"]
        if 'CX_TEAM_NAME' in df_filtrado.columns:
            canal_opcoes += sorted(df_filtrado['CX_TEAM_NAME'].dropna().unique().tolist())
        canal_filtro = st.selectbox(
            "Filtrar por canal",
            canal_opcoes,
            help="Filtra a fila pelo canal de atendimento original do cliente.",
        )
    with fc3:
        st.markdown(
            f'<div class="info-box">📋 Período: <strong>{periodo}</strong> · '
            f'Threshold ativo: <strong>{score_threshold}</strong> · '
            f'<strong>{n_alto_risco:,}</strong> clientes elegíveis</div>',
            unsafe_allow_html=True,
        )

    # ── Construir fila filtrada ───────────────────────────────────────────────
    fila = df_filtrado[df_filtrado['score_risco'] >= score_threshold].copy()

    # Filtro de canal
    if canal_filtro != "Todos os canais" and 'CX_TEAM_NAME' in fila.columns:
        fila = fila[fila['CX_TEAM_NAME'] == canal_filtro]

    # Urgência e ação
    def urgencia_label(score):
        if score >= 85:   return '🔴 Crítico'
        if score >= 70:   return '🟡 Alto'
        return '🟢 Moderado'

    def acao_label(score):
        if score >= 85: return 'Retorno imediato (<1h)'
        if score >= 70: return 'Retorno em até 4h'
        return 'Monitorar'

    fila['Urgência']          = fila['score_risco'].apply(urgencia_label)
    fila['Ação Recomendada']  = fila['score_risco'].apply(acao_label)
    fila['Score']             = fila['score_risco']
    fila['ID Cliente']        = fila['id'].astype(int)
    fila['Data Contato']      = pd.to_datetime(fila['INCOMING_DTTM']).dt.strftime('%d/%m %H:%M')

    if 'CX_TEAM_NAME' in fila.columns:
        fila['Canal'] = fila['CX_TEAM_NAME'].str.replace('BR_', '', regex=False).str[:25]
    else:
        fila['Canal'] = '—'

    # Filtro de urgência
    if urgencia_filtro == "🔴 Crítico (≥ 85)":
        fila = fila[fila['score_risco'] >= 85]
    elif urgencia_filtro == "🟡 Alto (≥ 70 e < 85)":
        fila = fila[(fila['score_risco'] >= 70) & (fila['score_risco'] < 85)]

    fila = fila.sort_values('score_risco', ascending=False).head(100)

    # ── Métricas da fila ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    n_critico  = len(fila[fila['score_risco'] >= 85])
    n_alto     = len(fila[(fila['score_risco'] >= 70) & (fila['score_risco'] < 85)])
    n_moderado = len(fila[fila['score_risco'] < 70])

    m1.metric("Total na fila", f"{len(fila):,}",
              help="Total de clientes elegíveis com os filtros atuais.")
    m2.metric("🔴 Crítico (≥ 85)", f"{n_critico:,}",
              help="Retorno em até 1 hora.")
    m3.metric("🟡 Alto (70–84)", f"{n_alto:,}",
              help="Retorno em até 4 horas.")
    m4.metric("Taxa interceptação", f"{taxa_interceptacao:.0%}",
              delta="Meta: ≥ 60%", delta_color="off")

    st.divider()

    # ── Tabela e gráfico lado a lado ──────────────────────────────────────────
    col_tabela, col_dist = st.columns([3, 2])

    with col_tabela:
        colunas_exibir = ['Urgência', 'Score', 'ID Cliente', 'Data Contato', 'Canal', 'Ação Recomendada']
        colunas_exibir = [c for c in colunas_exibir if c in fila.columns]

        st.dataframe(
            fila[colunas_exibir],
            use_container_width=True,
            height=400,
            hide_index=True,
            column_config={
                'Score': st.column_config.ProgressColumn(
                    'Score de Risco',
                    min_value=0,
                    max_value=100,
                    format='%.1f',
                ),
            },
        )
        st.caption(
            f"Exibindo {len(fila):,} clientes · filtro: {urgencia_filtro} · canal: {canal_filtro} · período: {periodo}"
        )

    with col_dist:
        st.markdown("**Distribuição dos scores no período**")

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_filtrado['score_risco'],
            nbinsx=30,
            marker_color='#1a73e8',
            opacity=0.75,
            name='Todos os clientes',
        ))
        # Highlight da área acima do threshold
        acima = df_filtrado[df_filtrado['score_risco'] >= score_threshold]['score_risco']
        fig_hist.add_trace(go.Histogram(
            x=acima,
            nbinsx=15,
            marker_color='#d93025',
            opacity=0.6,
            name=f'Na fila (≥ {score_threshold})',
        ))
        fig_hist.add_vline(
            x=score_threshold,
            line_color='#d93025',
            line_dash='dash',
            line_width=2,
            annotation_text=f'Threshold {score_threshold}',
            annotation_position='top right',
            annotation_font_size=11,
        )
        fig_hist.update_layout(
            height=250,
            margin=dict(t=10, b=30, l=10, r=10),
            barmode='overlay',
            legend=dict(orientation='h', y=1.1, x=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(title='Score de Risco', showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(title='Clientes', showgrid=True, gridcolor='#f0f0f0'),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ROI rápido
        st.markdown(
            '<div class="info-box">'
            '💰 <strong>ROI da interceptação</strong><br>'
            'Custo por contato: R$ 15 · Economia por procedente evitado: R$ 350<br>'
            '<strong>ROI estimado: 483% ao mês</strong>'
            '</div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — CAUSA RAIZ
# ════════════════════════════════════════════════════════════════════════════════
with tab3:

    with st.expander("ℹ️ Como a IA identificou as causas raiz?", expanded=False):
        st.markdown("""
Usamos um pipeline de **Inteligência Artificial** em 4 etapas para descobrir por que os casos são julgados
procedentes pelo Bacen — sem precisar de texto livre de atendimento:

1. **Embeddings semânticos** (OpenAI `text-embedding-3-small`): cada caso vira um vetor numérico
   que captura o *significado* do problema, não apenas as palavras
2. **Redução dimensional** (t-SNE): os vetores de 1.536 dimensões são comprimidos para 2D
3. **Clustering** (KMeans): algoritmo agrupa automaticamente casos com problemas similares
4. **Labeling por LLM** (GPT-4o-mini): um modelo de linguagem descreve cada grupo com linguagem natural

**Resultado:** 7 grupos de causa raiz com volumes, urgências e ações concretas por produto.
        """)

    if df_clusters.empty:
        st.info("Dados de clustering não disponíveis. Execute o notebook 03_q3_root_cause_nlp.ipynb.")
        st.stop()

    # ── Filtros de causa raiz ─────────────────────────────────────────────────
    cr1, cr2 = st.columns([2, 3])
    with cr1:
        produtos_disponiveis = ['Todos'] + sorted(df_clusters['produto'].dropna().unique().tolist())
        produto_filtro = st.selectbox(
            "Filtrar por produto",
            produtos_disponiveis,
            help="Filtra os clusters pelo produto afetado.",
        )
    with cr2:
        st.markdown(
            '<div class="info-box">'
            f'📊 <strong>{len(df_clusters)}</strong> grupos identificados · '
            f'<strong>{df_clusters["volume_procedentes"].sum()}</strong> casos procedentes analisados · '
            'Todos classificados com urgência Alta pelo modelo'
            '</div>',
            unsafe_allow_html=True,
        )

    df_cl = df_clusters.copy()
    if produto_filtro != 'Todos':
        df_cl = df_cl[df_cl['produto'] == produto_filtro]

    st.divider()

    # ── Gráfico de barras + tabela detalhada ──────────────────────────────────
    col_chart, col_detail = st.columns([2, 3])

    with col_chart:
        st.markdown("**Volume de procedentes por grupo**")

        df_sorted = df_cl.sort_values('volume_procedentes', ascending=True)

        cores_produto = {
            'Conta':            '#1a73e8',
            'Cartão de Crédito':'#d93025',
            'Consumer Credits': '#f29900',
            'Pix':              '#1e8e3e',
            'Múltiplos':        '#9334e6',
        }
        cores = [cores_produto.get(p, '#5f6368') for p in df_sorted['produto']]

        fig_bar = go.Figure(go.Bar(
            y=df_sorted['label_llm'],
            x=df_sorted['volume_procedentes'],
            orientation='h',
            marker_color=cores,
            text=[f"{v} ({p}%)" for v, p in
                  zip(df_sorted['volume_procedentes'], df_sorted['pct_do_indice'])],
            textposition='outside',
            textfont=dict(size=11),
        ))
        fig_bar.update_layout(
            height=max(250, len(df_sorted) * 48),
            margin=dict(t=10, b=10, l=10, r=80),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(title='Procedentes', showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(title='', automargin=True, tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Legenda de produtos
        st.markdown("**Legenda de produtos:**")
        for prod, cor in cores_produto.items():
            if prod in df_cl['produto'].values:
                st.markdown(
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:{cor};border-radius:2px;margin-right:6px"></span>'
                    f'<span style="font-size:0.85rem">{prod}</span> &nbsp;',
                    unsafe_allow_html=True,
                )

    with col_detail:
        st.markdown("**Causas raiz e ações recomendadas**")

        for _, row in df_cl.sort_values('volume_procedentes', ascending=False).iterrows():
            cor_prod = cores_produto.get(row['produto'], '#5f6368')
            pct = row.get('pct_do_indice', 0)

            with st.expander(
                f"**{row['label_llm']}** · {row['produto']} · {row['volume_procedentes']} casos ({pct}%)",
                expanded=False,
            ):
                st.markdown(f"**Produto afetado:** {row['produto']}")
                st.markdown(f"**Volume:** {row['volume_procedentes']} casos procedentes ({pct}% do índice)")

                if 'top_cdu_names' in row and pd.notna(row.get('top_cdu_names')):
                    st.markdown(f"**Principais causas:** {row['top_cdu_names']}")

                st.markdown("**Por que o Bacen julga procedente:**")
                st.info(row.get('causa_raiz_llm', '—'))

                st.markdown("**Ação recomendada:**")
                st.success(row.get('acao_recomendada', '—'))

    st.divider()

    # ── Treemap de impacto ────────────────────────────────────────────────────
    st.markdown("### Mapa de impacto — Produto × Volume")

    with st.expander("ℹ️ Como ler o mapa de impacto?", expanded=False):
        st.markdown("""
Cada retângulo representa um grupo de causa raiz. O **tamanho** é proporcional ao número de casos
procedentes naquele grupo — quanto maior o retângulo, maior o impacto no índice Bacen.
A **cor** identifica o produto. Clique em um produto para ver apenas seus grupos.
        """)

    fig_tree = px.treemap(
        df_cl,
        path=['produto', 'label_llm'],
        values='volume_procedentes',
        color='produto',
        color_discrete_map=cores_produto,
        custom_data=['pct_do_indice', 'acao_recomendada'],
    )
    fig_tree.update_traces(
        texttemplate='<b>%{label}</b><br>%{value} casos',
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Procedentes: %{value}<br>'
            '% do índice: %{customdata[0]}%<br>'
            '<extra></extra>'
        ),
    )
    fig_tree.update_layout(
        height=350,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_tree, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Regulatory Tracker v2.0 · Mercado Pago CS Shared Services · "
    "Atualização D-1 · Score batch/1h · "
    f"Período exibido: {periodo} ({data_inicio.strftime('%d/%m/%Y')} – {data_max.strftime('%d/%m/%Y')})"
)
