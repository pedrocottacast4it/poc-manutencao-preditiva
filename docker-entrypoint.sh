#!/bin/sh
set -eu
exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8080}"
