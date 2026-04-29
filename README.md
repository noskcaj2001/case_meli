# Case Técnico — Analista de Dados Sênior · CS Shared Services
## Mercado Pago · Redução do Índice Bacen

Solução técnica completa para o case de dados da vaga de **Analista de Dados Sênior — CS Shared Services** do Mercado Pago.

**Objetivo:** apoiar a OKR de atingir a **15ª posição no Ranking Público de Reclamações do Banco Central do Brasil** por meio de modelagem de dados, machine learning e inteligência artificial.

---

## Estrutura do repositório

```
├── notebooks/
│   ├── 00_eda.ipynb                  # Análise exploratória das 3 bases
│   ├── 01_q1_forecasting.ipynb       # Q1 · Diagnóstico e Projeção do Índice Bacen
│   ├── 02_q2_predictive_model.ipynb  # Q2 · Score de risco regulatório (ML)
│   ├── 03_q3_root_cause_nlp.ipynb    # Q3 · Causa raiz via NLP + LLM
│   └── 04_q4_business_case.ipynb     # Q4 · Business case e estratégia de mitigação
│
├── dashboard/
│   └── app.py                        # Q5 · Dashboard executivo (Streamlit)
│
└── requirements.txt
```

---

## As 5 entregas

### Q1 · Diagnóstico e Projeção do Índice Bacen
**Modelo:** Janela móvel de 12 meses ancorada no índice oficial Q4/2025 = **29,79** (dados públicos do Bacen).

**Metodologia:** projeção determinística por cenários — mais robusta que ARIMA/Prophet para série de apenas 4 trimestres.

**3 cenários** para os 4.734 casos pendentes (Pendente: IF/AC):

| Cenário | Pendentes procedentes | Pico do índice | Supera PicPay (35,2) |
|---------|----------------------|----------------|----------------------|
| Pessimista (70%) | 3.314 extras | 74,57 em Q3/26 | Q2/2027 |
| Base (45%) | 2.130 extras | 57,52 em Q3/26 | Q2/2027 |
| Otimista (20%) | 947 extras | 40,47 em Q3/26 | **Q1/2027** |

**Insight central:** o índice piora em H1/2026 (choque dos pendentes) antes de cair em Q1/2027, quando o maior bloco de casos sai da janela de 12 meses. Sem intervenção ativa, apenas o PicPay é superado no horizonte H2/2026–H1/2027.

---

### Q2 · Score de Risco Regulatório (ML)
**Modelo:** `GradientBoostingClassifier` com validação temporal (treino set–out/2025, teste nov/2025).

**Target:** `abriu_rdr_48h` — se o cliente abriu uma reclamação formal no Bacen nas 48h seguintes ao contato de 1ª instância.

**Features mais relevantes (SHAP):**
- `CX_PR_NAME` — produto reclamado (Conta, Cartão, PIX)
- `CDU_NAME` — tipo de caso (Fraude PIX, Restrições, Gestão de casos)
- `CX_TEAM_NAME` — canal de atendimento
- Tempo de resolução do caso
- Reincidência do cliente

**Threshold operacional:** score ≥ 70 → enfileirar para interceptação proativa (contato antes das 48h).

---

### Q3 · Causa Raiz via NLP + LLM
**Pipeline:**
1. Embeddings dos campos `CDU_NAME` + `CX_PR_NAME` via OpenAI (`text-embedding-3-small`)
2. Clustering com K-Means (7 clusters)
3. Labeling automático dos clusters via GPT-4o-mini
4. Mapeamento cluster → produto → impacto no índice

**Top clusters por volume de procedentes:**

| Cluster | Label | Produto | % do índice |
|---------|-------|---------|-------------|
| 5 | Reclamações Bacen Mercado Pago | Conta | 22,7% |
| 0 | Fraudes em pagamentos MP | Conta | 19,0% |
| 1 | Problemas com pagamentos | Cartão de Crédito | 16,5% |
| 6 | Reclamações de Créditos | Consumer Credits | 12,0% |
| 4 | Fraude PIX | Pix | 8,4% |

---

### Q4 · Business Case e Estratégia de Mitigação

**Impacto de –30% nas procedentes:**
- Atual: 1.862 procedentes → índice **29,79**
- Com –30%: 1.303 procedentes → índice **~20,85** → ~15ª posição

**4 frentes prioritárias:**

| # | Frente | Redução estimada |
|---|--------|-----------------|
| 1 | Interceptação proativa (score ≥ 70) | –10% |
| 2 | Fraude PIX — MFA reforçado + reversão < 2h | –10% |
| 3 | Task force: 4.734 casos pendentes | –6% |
| 4 | Comunicação proativa de restrições de conta | –4% |

---

### Q5 · Dashboard Executivo (Streamlit)

Dashboard interativo com 3 abas:

- **Visão Executiva:** gauge do índice atual vs meta, projeção de cenários com cruzamentos competitivos, ranking de mercado
- **Fila Operacional:** lista em tempo real dos clientes com score ≥ threshold, filtros por canal e urgência
- **Causa Raiz:** mapa de clusters NLP com volume e % de impacto no índice

**Para executar:**
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## Contexto do problema

O **Índice Bacen** é calculado como:

```
Índice = (Reclamações Procedentes ÷ Base de Clientes) × 1.000.000
```

**Posicionamento Q4/2025:**

| Instituição | Índice | Clientes |
|-------------|--------|----------|
| Nubank | 13,90 | 95,0M |
| Banco Inter | 24,10 | 30,8M |
| **Mercado Pago** | **29,79** | **62,5M** |
| PicPay | 35,20 | 36,2M |
| C6 Bank | 64,30 | 32,1M |
| PagBank | 68,50 | 34,5M |

---

## Stack

| Camada | Tecnologia |
|--------|-----------|
| Processamento | Python · pandas · scikit-learn |
| ML | GradientBoostingClassifier · SHAP |
| NLP / Embeddings | OpenAI API (`text-embedding-3-small`) · K-Means |
| LLM para labeling | GPT-4o-mini |
| Série temporal | Modelo determinístico de janela móvel |
| Visualização | Streamlit · Plotly · Matplotlib |

---

## Configuração

Crie um arquivo `.env` na raiz com:

```
OPENAI_API_KEY=sua_chave_aqui
```
