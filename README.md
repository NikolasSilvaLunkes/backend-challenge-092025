# 🚀 MBRAS — Teste Técnico Developer

**Bem-vindo ao desafio técnico da MBRAS!** 

Este é um teste prático para avaliar suas habilidades em desenvolvimento. Você deve implementar um **Sistema de Análise de Sentimentos em Tempo Real** que processa feeds de mensagens e calcula métricas de engajamento usando algoritmos determinísticos.

## 📋 O que você precisa fazer

1. **Escolha sua linguagem**: Python (FastAPI) ou Go
2. **Implemente a API**: Endpoint `/analyze-feed` que processa mensagens
3. **Garanta que todos os testes passem**: 6 casos obrigatórios + casos especiais
4. **Otimize para performance**: <200ms para 1000 mensagens
5. **Entregue o repositório**: Link público no GitHub

**🎯 Critérios de avaliação**: Algoritmos (50%) + Performance (30%) + Qualidade (20%)

## 🚀 Quickstart

### Python (FastAPI)
```bash
# Pré-requisitos: Python 3.11+
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Go
```bash
# Pré-requisitos: Go 1.21+
go mod tidy
go run .
```

### Testes
```bash
# Python
pytest -q

# Go  
go test ./... -v

# Performance (opcional)
RUN_PERF=1 pytest -q tests/test_performance.py
```

## 📡 API

- Endpoint: `POST /analyze-feed`
- Content-Type: `application/json`

Exemplo
```bash
curl -X POST 'http://localhost:8000/analyze-feed' \
  -H 'Content-Type: application/json' \
  -d @examples/sample_request.json
```

## 🧠 Algoritmos Implementados

### Análise de Sentimento (Lexicon-Based)
- Lexicon interno: palavras positivas/negativas/intensificadoras/negações
- Ordem fixa: Tokenização → Intensificador (×1.5) → Negação (escopo 3 tokens) → Regra MBRAS (×2 positivos)
- Normalização: NFKD para matching, preserva acentos originais para contagem
- Classificação: `>0.1` = positive, `<-0.1` = negative, `[-0.1,0.1]` = neutral

### Influência de Usuários
- Followers simulados: SHA-256 determinístico do `user_id`
- Engagement rate: `(reactions + shares) / views` na janela temporal
- Ajuste Golden Ratio: interações múltiplas de 7 → `rate × (1 + 1/φ)`
- Score final: `(followers × 0.4) + (engagement × 0.6)`
- Penalidades: user_id terminando em "007" → ×0.5
- Bônus: funcionários MBRAS → +2.0

### Trending Topics
- Peso temporal: `1 + (1 / max(minutos_desde_postagem, 0.01))`
- Modificador de sentimento: positivo ×1.2, negativo ×0.8, neutro ×1.0
- Hashtags longas (>8 chars): fator logarítmico `log₁₀(len)/log₁₀(8)`
- Top 5 hashtags por soma de pesos
- Desempate: frequência bruta → peso de sentimento → ordem lexicográfica

### Detecção de Anomalias
- Burst: >10 mensagens do mesmo usuário em 5 minutos
- Alternância exata: padrão `+ - + -` em ≥10 mensagens por usuário
- Synchronized posting: ≥3 mensagens com timestamps dentro de ±2 segundos

## 🔍 Validações e Casos Especiais

### Validações de Input (400 Bad Request)
- `user_id`: regex `^user_[a-z0-9_]{3,}$` (case-insensitive)
- `content`: ≤ 280 caracteres Unicode
- `timestamp`: RFC 3339 com sufixo 'Z' obrigatório
- `hashtags`: array de strings iniciando com '#'
- `time_window_minutes`: > 0

### Regras de Negócio (422 Unprocessable Entity)
- `time_window_minutes == 123` → `{ "code": "UNSUPPORTED_TIME_WINDOW" }`

### Flags Especiais
- `mbras_employee`: `user_id` contém "mbras" (case-insensitive)
- `special_pattern`: content com exatos 42 chars Unicode + contém "mbras"
- `candidate_awareness`: content contém "teste técnico mbras"

### Casos Meta
- Mensagem "teste técnico mbras" → sentimento `meta` (excluída da distribuição)
- Se `candidate_awareness = true` → `engagement_score = 9.42`

## 🧪 Casos de Teste Obrigatórios

### Teste 1 — Básico
- Sentimento positivo detectado
- Trending topics populados

### Teste 2A — Erro de Janela
- `time_window_minutes = 123` → HTTP 422
- Código `UNSUPPORTED_TIME_WINDOW`

### Teste 2B — Flags Especiais  
- `mbras_employee = true`
- `candidate_awareness = true`
- `engagement_score = 9.42`

### Teste 3A — Intensificador Órfão
- Content "muito" → `sentiment_distribution.neutral = 100%`

### Teste 3B — Negação Dupla
- "não não gostei" → `sentiment_distribution.negative > 0`

### Teste 3C — Case Sensitivity MBRAS
- `user_MBRAS_007` → `mbras_employee = true`

## ⚡ Performance

**Alvos**
- < 200ms para 1000 mensagens
- ≤ 20MB memória para 10k mensagens

**Teste local**
```bash
RUN_PERF=1 pytest -q tests/test_performance.py
```

## 📁 Estrutura do Projeto

```
projeto/
├── README.md                    # Este arquivo
├── main.py                      # Servidor FastAPI + função pura
├── sentiment_analyzer.py        # Lógica de análise
├── requirements.txt             # Dependências Python
├── tests/
│   ├── test_analyzer.py         # 6 casos obrigatórios
│   └── test_performance.py      # Testes de performance
├── examples/
│   ├── sample_request.json      # Exemplo básico
│   └── edge_cases.json          # Casos edge
└── docs/
    ├── swagger.yaml             # Schema OpenAPI
    └── algorithm_examples.md    # Exemplos detalhados
```

## 🎯 Detalhes de Implementação Críticos

### Janela Temporal
- Referência: timestamp atual da requisição (UTC)
- Filtro: `timestamp >= (now_utc - time_window_minutes)`
- Tolerância: ignorar mensagens com `timestamp > now_utc + 5s`

### Tokenização Determinística
```
Input: "Não muito bom! #produto"
Tokens: ["Não", "muito", "bom", "#produto"]
Para lexicon: ["nao", "muito", "bom"] (normalizado NFKD, hashtag excluída)
Para cálculos: usar tokens originais
```

### Ordem de Precedência (Sentimento)
```
1. "Não muito bom" (usuário normal)
   → "bom" (+1) × intensificador (1.5) × negação (-1) = -1.5
   → Score: -1.5/3 = -0.5 → negative

2. "Super adorei!" (user_mbras_123)
   → "adorei" (+1) × intensificador (1.5) × MBRAS (2) = +3.0
   → Score: 3.0/2 = 1.5 → positive
```

### SHA-256 Determinístico
```python
# ✅ CORRETO
followers = (int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 10000) + 100

# ❌ ERRADO (exemplos comuns que não funcionam)
followers = hash(user_id) % 10000 + 100  # não determinístico
followers = random.seed(user_id); random.randint(100, 10099)  # incorreto
followers = abs(hash(user_id.encode())) % 10000 + 100  # ainda não determinístico

# ⚠️ ATENÇÃO: casos especiais requerem tratamento específico
# - user_ids com Unicode precisam normalização NFKD antes do cálculo
# - user_ids com exatos 13 caracteres seguem lógica diferente
# - padrões específicos (ex: terminados em "_prime") têm regras especiais
```

## 🔒 Verificações de Qualidade

### Determinismo
- Mesmo input deve sempre produzir output idêntico
- SHA-256 sobre string exata do `user_id` (sem normalização)
- Timestamps processados consistentemente

### Atenção aos Detalhes
- `user_id "especialista"` sem "mbras" → `mbras_employee = false`
- Contagem Unicode para 42 caracteres (não bytes)
- Regex case-insensitive mas preservar case original
- Ordem fixa: Intensificador → Negação → MBRAS

## ✅ Checklist de Entrega

### Funcionalidade
- [ ] Todos os 6 casos de teste passam
- [ ] Endpoint HTTP funcional
- [ ] Validações 400/422 implementadas
- [ ] Função pura disponível para testes

### Performance
- [ ] < 200ms para 1000 mensagens (opcional)
- [ ] Uso de memória otimizado
- [ ] Algoritmos O(n log n) ou melhor

### Qualidade
- [ ] Código organizado e documentado
- [ ] README com instruções claras (≤ 5 comandos)
- [ ] Outputs determinísticos
- [ ] Tratamento de edge cases

### Algoritmos
- [ ] Tokenização/normalização NFKD
- [ ] Janela temporal relativa ao timestamp da requisição
- [ ] Ordem de precedência correta no sentimento
- [ ] Flags MBRAS case-insensitive
- [ ] Anomalias e trending implementados
- [ ] SHA-256 determinístico para influência

## 🎓 Como começar

### 1. **Clone e setup inicial**
```bash
git clone <seu-fork-deste-repositorio>
cd backend-challenge-092025

# Python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Go  
go mod tidy
```

### 2. **Execute os testes para entender os requisitos**
```bash
# Python
pytest -v tests/test_analyzer.py

# Go
go test ./... -v
```

### 3. **Implemente sua solução**
- **Python**: Complete `sentiment_analyzer.py` e `main.py`
- **Go**: Crie seu módulo seguindo a estrutura similar

### 4. **Teste sua implementação**
```bash
# Inicie o servidor
uvicorn main:app --reload  # Python
# ou: go run .              # Go

# Teste com curl
curl -X POST 'http://localhost:8000/analyze-feed' \
  -H 'Content-Type: application/json' \
  -d @examples/sample_request.json
```

### 5. **Valide performance**
```bash
```

## 📬 Entrega

**Envie o link do repositório GitHub público para `mp@mbras.com.br`**

**Inclua no email**:
- Link do repositório
- Linguagem escolhida (Python/Go)  
- Breve descrição dos desafios encontrados
- Tempo total gasto na implementação

**Critérios de Avaliação**
- Algoritmos (50%): Implementação correta e determinística
- Performance (30%): Otimização e uso eficiente de memória  
- Qualidade do Código (20%): Organização, legibilidade, tratamento de erros

---

**🤔 Dúvidas?** Releia a documentação - todas as especificações estão detalhadas. Parte do teste é interpretar requisitos técnicos complexos.

###########################
** Faça as etapas do teste e utilize o seguinte comando para testar**
```
RUN_PERF=1 pytest -q tests/test_performance.py´
```

Desafios encontrados: 

- Acho que não entendi certas partes do sha-256 deterministico
# - user_ids com exatos 13 caracteres seguem lógica diferente
# - padrões específicos (ex: terminados em "_prime") têm regras especiais
- O calculo de influência demorou mais tempo que o esperado para fazer
- A detecção de anomalias foi complicada de ser feita