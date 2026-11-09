# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PoC de manutenção preditiva (WTTE-RNN: LSTM que prevê os parâmetros α, β de uma Weibull para o tempo até a falha) sobre o NASA C-MAPSS FD001. Derivada do template de demo da CAST (`../../template-demo-repository`): imagem Docker única com FastAPI + `index.html` estático, deploy em AWS App Runner. Documentação e código estão em pt-BR.

## Comandos

Ambiente (uv, Python 3.12; o venv fica em `ml/.venv`, ignorado pelo git):

```bash
uv venv -p 3.12 ml/.venv
VIRTUAL_ENV=ml/.venv uv pip install -r ml/requirements.txt -r backend/requirements.txt
```

Treinar o modelo (~30 s em CPU; regrava `backend/model/wtte_fd001.npz` e `backend/model/metrics.json`):

```bash
ml/.venv/bin/python ml/train.py
```

Rodar a demo sem Docker (o backend serve o frontend a partir de `FRONTEND_DIR`; por padrão espera `frontend/`, que só existe na imagem):

```bash
mkdir -p .dev-frontend && cp -R index.html logo_cast.png favicon.ico vendor .dev-frontend/
FRONTEND_DIR=.dev-frontend ml/.venv/bin/uvicorn backend.main:app --port 8080 --reload
```

Depois de editar `index.html` ou `vendor/`, copie de novo para `.dev-frontend/` — ele não é sincronizado.

Docker (mesmo alvo da CI; o estágio final `development` adiciona `--reload` e não deve ser publicado):

```bash
docker build --target runtime -t manutencao-preditiva .
docker run --rm -p 8080:8080 manutencao-preditiva
```

Não há suíte de testes nem linter configurados. A verificação de consistência do modelo é feita dentro de `ml/train.py` (ver abaixo).

## Arquitetura

**Treino × inferência são separados por dependência.** `ml/train.py` usa PyTorch; a imagem de runtime só tem NumPy. O treino exporta pesos da LSTM + parâmetros de normalização para `.npz`, e `backend/wtte.py` reimplementa o forward da LSTM em NumPy. O próprio `train.py` faz `assert np.allclose(...)` comparando os dois forwards — qualquer mudança na arquitetura (`HIDDEN`, ativações de α/β, `MAX_BETA`, features) precisa ser espelhada nos dois arquivos, senão o treino falha nesse assert. `FEATURES`, `load_cmapss` e `feature_matrix` vivem em `backend/wtte.py` e são importados pelo treino, então são a fonte única de verdade.

**Backend sem estado, pré-computado na subida.** `fleet_state()` em `backend/main.py` (com `lru_cache`) carrega o modelo, roda a LSTM uma vez sobre as 100 unidades de `test_FD001.txt` e guarda α, β para cada ciclo de cada ativo. Os endpoints (`/api/fleet`, `/api/assets/{id}?cycle=`, `/api/validation`, `/api/info`) só recortam essas séries e aplicam funções fechadas da Weibull (`weibull_cdf`, `weibull_quantile`, etc.) com os parâmetros de consulta `horizon`, `threshold` e `warn`. O status (`critico`/`atencao`/`normal`) vem de P(falha ≤ horizon) comparada com `threshold`/`warn`. O "replay" ciclo a ciclo do frontend é só ler α, β num ciclo anterior.

**Dados e artefatos que vão para a imagem.** O `Dockerfile` copia explicitamente `backend/` (incluindo `backend/model/`), apenas `test_FD001.txt` e `RUL_FD001.txt` de `data/cmapss/` (o arquivo de treino não entra), `index.html`, `logo_cast.png`, `favicon.ico` e `vendor/` → `/app/frontend/`. Qualquer novo asset estático ou arquivo de dados precisa ser adicionado ao `Dockerfile`. Os artefatos do modelo em `backend/model/` são versionados: retreinar muda o que é servido em produção.

**Frontend.** `index.html` é um arquivo único (HTML/CSS/JS vanilla, sem build) com as abas Frota, Ativo, Validação e Impacto; chama a API via `fetch` em caminhos relativos. Chart.js 4.4.4 está versionado em `vendor/chart.umd.min.js` (sem CDN). Os nomes de ativos (`MOE-01`, `EXC-02`…) são gerados em `ASSET_TYPES` no backend e são apenas ilustrativos — os dados são de motores turbofan simulados usados como proxy.

## Restrições de deploy

- Destino é AWS App Runner: apenas HTTP request/response. Não usar WebSockets, SSE de longa duração, sticky sessions nem Streamlit (o README explica o motivo).
- Porta 8080, health check em `/health`, container roda como UID 10001.
- CI: `.github/workflows/main.yml` dispara em push na `main` e chama o workflow reutilizável `castinformaticasa/github-workflows/.../deploy_demo_dsc.yml` (build + push para ECR). Os inputs `department`/`application`/`project-name` ainda estão com os valores de outro projeto (`project-name: "defeitos-aco-laminado"`) — confirme antes de mexer na esteira.
- Ao lançar versão, atualizar `version` em `package.json`, em `FastAPI(..., version=...)` em `backend/main.py` e o `CHANGELOG.md`.
