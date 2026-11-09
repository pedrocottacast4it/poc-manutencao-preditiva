# PoC — Manutenção Preditiva (WTTE-RNN)

Prova de conceito da iniciativa **Previsão de Falhas em Equipamentos Críticos (Weibull + LSTM)**. Uma LSTM lê a telemetria de cada ativo ciclo a ciclo e prevê os parâmetros (α, β) de uma **distribuição de Weibull** para o tempo até a falha. Assim, em vez de um alerta binário, a demo entrega o RUL (tempo restante de vida útil) com intervalo de confiança e a probabilidade de falha dentro da janela de planejamento do PCM.

Construída sobre o template de demo da CAST, descrito na segunda parte deste README: imagem Docker única com FastAPI + `index.html`, sem WebSockets, pronta para App Runner.

## Dataset

**NASA C-MAPSS FD001** (Saxena et al., PHM08), de domínio público, baixado do [repositório de prognósticos da NASA](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip). Ele é o benchmark em que o WTTE-RNN foi apresentado (Martinsson, 2016).

| Arquivo | Conteúdo |
| --- | --- |
| `data/cmapss/train_FD001.txt` | 100 unidades rodadas até a falha (histórico de "ordens de serviço" com data de falha) |
| `data/cmapss/test_FD001.txt` | 100 unidades com a série interrompida antes da falha (a "frota em operação") |
| `data/cmapss/RUL_FD001.txt` | RUL real de cada unidade de teste, usado só para validar |

Por que este dataset: tem degradação gradual com ruído de sensor, dados censurados e gabarito de RUL, que é exatamente o que precisamos para validar o modelo antes de ter dados de uma usina. **Limitação:** são motores turbofan simulados, não moendas. Por isso a demo usa os dados como proxy de degradação de equipamento rotativo e dá aos ativos nomes de usina apenas ilustrativos (`MOE-01`, `EXC-02`…). O tempo é medido em ciclos de operação.

## Modelo

- **Entrada:** 14 sensores informativos do FD001, normalizados por z-score (os outros 7 são constantes nessa condição operacional).
- **Rede:** LSTM(64) → camada linear → `α = α₀·exp(·)`, `β = 10·sigmoid(·)`.
- **Perda:** log-verossimilhança da Weibull discreta com censura. Além das 80 unidades rodadas até a falha, o treino usa 4 cópias truncadas de cada uma como dados censurados ("ainda não falhou"), que é o caso da maioria dos ativos de uma usina.
- **Validação:** 20 unidades de treino separadas para early stopping. A avaliação final usa as 100 unidades de teste.
- **Baseline:** Weibull populacional sem sensores (a análise de sobrevivência clássica do discovery), condicionada à idade do ativo.

Resultado no teste (`backend/model/metrics.json`):

| Métrica | WTTE-RNN | Weibull populacional |
| --- | --- | --- |
| RMSE com teto de 125 ciclos (convenção do C-MAPSS) | **16,8** | 34,2 |
| RMSE bruto | **29,6** | 39,4 |
| Cobertura do intervalo P10–P90 (alvo 80%) | 74% | 87% |
| Largura média do intervalo | **57** | 118 |
| Alertas com H=30 e limiar de 50% | 20 antecipadas, 5 perdidas, 0 alarmes falsos | — |

O treino roda em PyTorch. A imagem de runtime só precisa de NumPy: os pesos são exportados para `backend/model/wtte_fd001.npz` e o forward da LSTM é refeito em `backend/wtte.py` (o script de treino confere que os dois batem).

```bash
uv venv -p 3.12 ml/.venv
VIRTUAL_ENV=ml/.venv uv pip install -r ml/requirements.txt -r backend/requirements.txt
ml/.venv/bin/python ml/train.py        # ~30 s em CPU; regrava backend/model/
```

## A demo

| Aba | O que mostra |
| --- | --- |
| **Frota** | Os 100 ativos ordenados por P(falha no horizonte), com RUL P50 e P10–P90, status e, opcionalmente, o RUL real |
| **Ativo** | Replay ciclo a ciclo do que o modelo previa usando só o histórico até ali, a curva de Weibull atual e a telemetria |
| **Validação** | Comparação com o baseline, matriz de alertas (que muda com o horizonte e o limiar), previsto × real e curva de treino |
| **Impacto** | Custo reativo × preditivo com as premissas da barra lateral (lucro cessante, horas paradas, multiplicador emergencial) |

API: `GET /api/fleet`, `GET /api/assets/{id}?cycle=`, `GET /api/validation` (todas aceitam `horizon`, `threshold` e `warn`) e `GET /api/info`.

Rodar localmente sem Docker (o backend serve o frontend de `FRONTEND_DIR`):

```bash
mkdir -p .dev-frontend && cp -R index.html logo_cast.png favicon.ico vendor .dev-frontend/
FRONTEND_DIR=.dev-frontend ml/.venv/bin/uvicorn backend.main:app --port 8080 --reload
```

Chart.js 4.4.4 fica versionado em `vendor/`, para a demo não depender de CDN.

---

# Template de demo (base deste repositório)

Repositório de **template** para demonstrar entrega contínua com **GitHub Actions**, imagem **Docker** única (frontend + backend) e deploy em **AWS App Runner** via ECR.

Não há banco de dados: apenas uma demo HTTP stateless.

## Por que não usar Streamlit neste template (App Runner)

O [Streamlit](https://streamlit.io/) depende de **conexões WebSocket persistentes** entre o navegador e o servidor para atualizar a UI em tempo real. O **AWS App Runner não oferece suporte nativo a WebSockets** — limitação documentada no [roadmap do App Runner](https://github.com/aws/apprunner-roadmap/issues/13) e discutida na [comunidade Streamlit](https://discuss.streamlit.io/t/deploying-on-aws-app-runner/85000/5).

Sintomas típicos no App Runner: deploy “ok”, `/health` responde, mas a página fica em **“Please wait…”** ou não carrega a interface, porque o canal WebSocket não se estabelece.

**Sugestões para este template no App Runner:** **JavaScript** no browser (HTML/CSS/JS estático ou SPA já buildada) e API em **Node.js** ou em **linguagem compilada** (Go, Rust, Java, etc.) — apenas HTTP request/response, sem WebSockets. A referência mínima deste repositório usa **FastAPI + `index.html`** só como exemplo de imagem única; ao forking, adapte para a stack JS/compilada que o time já opera.

## Arquitetura da imagem

Uma única imagem expõe a porta **8080**:

| Caminho     | Responsável                           |
| ----------- | ------------------------------------- |
| `/`         | `index.html` (demo mobile no browser) |
| `/health`   | Health check (App Runner / Docker)    |
| `/api/info` | Exemplo de endpoint JSON da API       |

```
┌─────────────────────────────────────┐
│  Container (python:3.12-slim)       │
│  uvicorn → FastAPI                  │
│    ├── /          → frontend/       │
│    ├── /health                      │
│    └── /api/*     → backend         │
└─────────────────────────────────────┘
```

### Estrutura no repositório

```
.
├── index.html              # Frontend (copiado para /app/frontend na imagem)
├── logo_cast.png           # Assets referenciados pelo HTML (copiar junto no Dockerfile)
├── backend/
│   ├── main.py             # API FastAPI
│   └── requirements.txt
├── docker-entrypoint.sh
├── Dockerfile
└── .github/workflows/
    └── main.yml            # Esteira reutilizável → ECR
```

### Tudo embutido no Dockerfile

A imagem de runtime **não monta código do host**: o `Dockerfile` instala dependências Python, copia `backend/`, `index.html`, assets estáticos (ex.: `logo_cast.png`) e o entrypoint. O processo de build na CI gera um artefato autocontido pronto para o App Runner.

Build local (mesmo alvo da esteira):

```bash
docker build --target runtime -t demo-template .
docker run --rm -p 8080:8080 demo-template
```

Abra `http://localhost:8080` e teste `http://localhost:8080/api/info`.

## GitHub Actions (`github-workflows`)

O workflow [`.github/workflows/main.yml`](.github/workflows/main.yml) dispara em **push na branch `main`** e reutiliza o workflow corporativo `castinformaticasa/github-workflows` → `build_and_push_image_ecr.yml`, publicando a imagem no **Amazon ECR**.

### Secrets obrigatórios (repositório ou organization)

Configure em **Settings → Secrets and variables → Actions**:

| Secret                  | Uso                                                             |
| ----------------------- | --------------------------------------------------------------- |
| `AWS_ACCESS_KEY_ID`     | Credencial IAM para login no ECR e push da imagem               |
| `AWS_SECRET_ACCESS_KEY` | Par da credencial acima                                         |
| `AWS_REGION`            | Região AWS do registry (ex.: `us-east-1`)                       |
| `AWS_REPOSITORY`        | Identificador/nome do repositório ECR de destino                |
| `AWS_REGISTRY_TAG`      | Tag ou sufixo de versionamento usado pelo workflow reutilizável |

### Inputs do workflow (`with`)

| Input         | Valor no template | Descrição                                          |
| ------------- | ----------------- | -------------------------------------------------- |
| `environment` | `prd`             | Ambiente lógico repassado ao workflow reutilizável |
| `version`     | `latest`          | Versão/tag de build consumida pela esteira         |

Ajuste `environment` e `version` conforme o contrato do workflow em `castinformaticasa/github-workflows`. Garanta que o job de build use **`docker build --target runtime`** (ou equivalente) para não publicar o estágio `development` do Dockerfile.

## Personalizando o template

1. Edite `index.html` para o frontend da sua demo; inclua no `Dockerfile` qualquer novo arquivo estático (imagens, fontes, etc.) em `/app/frontend/`.
2. Estenda `backend/main.py` com novos endpoints em `/api/...`.
3. Atualize `backend/requirements.txt` se adicionar bibliotecas.
4. Mantenha **HTTP simples** se o destino continuar sendo App Runner (evite WebSockets, SSE long-lived ou sticky session).

## Licença

MIT (ver `package.json`).
