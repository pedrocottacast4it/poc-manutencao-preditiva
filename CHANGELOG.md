# Changelog

## [0.3.0] - 2026-10-06

### Adicionado
- PoC de manutenção preditiva com WTTE-RNN (LSTM + Weibull) treinado no NASA C-MAPSS FD001
- Script de treino `ml/train.py` (PyTorch) com dados censurados, early stopping e baseline Weibull populacional
- Inferência em NumPy (`backend/wtte.py`) e endpoints `/api/fleet`, `/api/assets/{id}` e `/api/validation`
- Frontend com abas Frota, Ativo (replay ciclo a ciclo), Validação e Impacto financeiro

## [0.2.0] - 2026-09-25

### Adicionado
- Template demo mobile em `index.html`, com estilos, navbar, sidebar, área de conteúdo e footer
- Logo CAST (`logo_cast.png`)

