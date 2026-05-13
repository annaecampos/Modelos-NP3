#!/usr/bin/env bash
# =============================================================================
# setup.sh — Instalação completa do ambiente Modelos NP3
# Uso: bash setup.sh
# =============================================================================
set -e

PYTHON_MIN="3.10"
VENV_DIR=".venv"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo "  Modelos NP3 — Setup do ambiente Python"
echo "========================================================"

# ── 1. Verificar Python ───────────────────────────────────────────────────────
echo ""
echo "[1/6] Verificando Python..."

# Escolhe o melhor Python disponível (prefere 3.12, aceita 3.10–3.13)
for ver in 3.12 3.11 3.10 3.13 3; do
    if command -v "python$ver" &>/dev/null; then
        PYTHON="python$ver"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERRO: Python 3.10+ não encontrado."
    echo "Instale com: sudo apt install python3.12 python3.12-venv"
    exit 1
fi

PY_VER=$($PYTHON --version 2>&1)
echo "  Usando: $PYTHON ($PY_VER)"

# ── 2. Instalar dependências de sistema ───────────────────────────────────────
echo ""
echo "[2/6] Instalando dependências de sistema (requer sudo)..."

PKGS_APT=""

# python3-venv (necessário para criar virtualenv)
if ! $PYTHON -c "import ensurepip" &>/dev/null; then
    PY_SHORT=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PKGS_APT="$PKGS_APT python${PY_SHORT}-venv"
fi

# graphviz (opcional, para plot_model)
if ! command -v dot &>/dev/null; then
    PKGS_APT="$PKGS_APT graphviz"
fi

# PostgreSQL (tenta pgdg; cai no pacote do sistema se o distro não for suportado)
if ! command -v psql &>/dev/null; then
    echo "  PostgreSQL não encontrado. Tentando instalar via pgdg..."
    DISTRO=$(lsb_release -cs 2>/dev/null || echo "unknown")
    KEYRING=/usr/share/keyrings/postgresql.gpg

    # Método moderno de adicionar chave GPG
    if curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
            | gpg --dearmor -o "$KEYRING" 2>/dev/null; then
        # Verificar se o distro existe no repositório pgdg antes de adicionar
        if curl -fsSL "https://apt.postgresql.org/pub/repos/apt/dists/${DISTRO}-pgdg/Release" \
                &>/dev/null; then
            echo "deb [signed-by=$KEYRING] https://apt.postgresql.org/pub/repos/apt \
${DISTRO}-pgdg main" \
                | tee /etc/apt/sources.list.d/pgdg.list >/dev/null
            PKGS_APT="$PKGS_APT postgresql postgresql-postgis"
            echo "  Repositório pgdg adicionado para '$DISTRO'."
        else
            echo "  Repositório pgdg não suporta '$DISTRO' ainda."
            echo "  Usando postgresql do repositório do sistema..."
            PKGS_APT="$PKGS_APT postgresql"
        fi
    else
        echo "  Não foi possível obter chave pgdg. Usando pacote do sistema..."
        PKGS_APT="$PKGS_APT postgresql"
    fi
fi

if [ -n "$PKGS_APT" ]; then
    sudo apt-get update -qq
    sudo apt-get install -y $PKGS_APT
    echo "  Pacotes instalados: $PKGS_APT"
else
    echo "  Dependências de sistema já presentes."
fi

# ── 3. Criar ambiente virtual ─────────────────────────────────────────────────
echo ""
echo "[3/6] Criando ambiente virtual em '$VENV_DIR'..."

cd "$PROJECT_DIR"

if [ -d "$VENV_DIR" ]; then
    echo "  '$VENV_DIR' já existe — recriando..."
    rm -rf "$VENV_DIR"
fi

$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "  Virtualenv criado e ativado."

# ── 4. Atualizar pip e ferramentas base ───────────────────────────────────────
echo ""
echo "[4/6] Atualizando pip, setuptools e wheel..."
pip install --upgrade pip setuptools wheel --quiet
pip --version

# ── 5. Instalar dependências Python ───────────────────────────────────────────
echo ""
echo "[5/6] Instalando dependências do projeto..."
pip install -r requirements.txt

echo ""
echo "  Pacotes instalados:"
pip list | grep -E "tensorflow|keras|numpy|pandas|scipy|scikit|matplotlib|psycopg2|pydot"

# ── 6. Verificar instalação ───────────────────────────────────────────────────
echo ""
echo "[6/6] Verificando importações..."
$PYTHON - <<'EOF'
import sys
erros = []
pacotes = [
    ("numpy",       "numpy"),
    ("pandas",      "pandas"),
    ("scipy",       "scipy"),
    ("sklearn",     "scikit-learn"),
    ("matplotlib",  "matplotlib"),
    ("keras",       "keras"),
    ("keras_tuner", "keras-tuner"),
    ("tensorflow",  "tensorflow"),
    ("psycopg2",    "psycopg2-binary"),
]
for mod, nome in pacotes:
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print(f"  ✓ {nome:<20} {v}")
    except ImportError:
        erros.append(nome)
        print(f"  ✗ {nome:<20} FALHOU")

if erros:
    print(f"\nATENÇÃO: {len(erros)} pacote(s) com problema: {', '.join(erros)}")
    sys.exit(1)
else:
    print("\n  Todos os pacotes OK!")
EOF

# ── Instruções finais ─────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Setup concluído com sucesso!"
echo "========================================================"
echo ""
echo "  Para ativar o ambiente em novos terminais:"
echo "    source $PROJECT_DIR/$VENV_DIR/bin/activate"
echo ""
echo "  Para extrair dados do banco:"
echo "    cd 'MANIPULAÇÃO DE DADOS/MODELO ORIGINAL'"
echo "    python3 consolida_dados.py"
echo ""
echo "  Para executar um modelo:"
echo "    cd MODELOS"
echo "    python3 'MODELO ORIGINAL.py'"
echo "    python3 'MODELO AUTOAJUSTADO COM KT.py' 5"
echo "========================================================"
