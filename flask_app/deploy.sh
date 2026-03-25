#!/usr/bin/env bash
# ============================================================
#  deploy.sh — Deploy Age Verification Flask App to AWS EC2
#  Usage:
#    1. Set EC2_HOST and PEM_PATH below (or pass as env vars)
#    2. chmod +x deploy.sh
#    3. ./deploy.sh
# ============================================================

set -euo pipefail

# ── Configuration (edit these) ──────────────────────────────
EC2_HOST="${EC2_HOST:-}"           # e.g. ec2-12-34-56-78.compute-1.amazonaws.com
EC2_USER="${EC2_USER:-ubuntu}"     # ubuntu (Ubuntu AMI) or ec2-user (Amazon Linux)
PEM_PATH="${PEM_PATH:-}"           # /path/to/your-key.pem
APP_DIR="/home/${EC2_USER}/age-verify-app"
PORT="${PORT:-5000}"

# ── Validation ──────────────────────────────────────────────
if [[ -z "$EC2_HOST" || -z "$PEM_PATH" ]]; then
  echo "ERROR: Set EC2_HOST and PEM_PATH environment variables."
  echo "  export EC2_HOST=ec2-xx-xx-xx-xx.compute-1.amazonaws.com"
  echo "  export PEM_PATH=/path/to/your.pem"
  exit 1
fi

SSH="ssh -i ${PEM_PATH} -o StrictHostKeyChecking=no ${EC2_USER}@${EC2_HOST}"
SCP="scp -i ${PEM_PATH} -o StrictHostKeyChecking=no"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "======================================================"
echo "  Deploying Age Verification App to EC2"
echo "  Host: ${EC2_HOST}"
echo "  User: ${EC2_USER}"
echo "  App:  ${APP_DIR}"
echo "======================================================"

# ── Step 1: Install system deps on EC2 ─────────────────────
echo ""
echo "[1/5] Installing system dependencies on EC2..."
$SSH "bash -s" <<'REMOTE_SETUP'
set -e
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv \
  libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
  libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
  git curl wget nginx supervisor
echo "System deps installed."
REMOTE_SETUP

# ── Step 2: Copy app files to EC2 ──────────────────────────
echo ""
echo "[2/5] Copying application files to EC2..."
$SSH "mkdir -p ${APP_DIR}"

# Create a local tarball, excluding large files
cd "$SCRIPT_DIR"
tar --exclude='./models' \
    --exclude='./uploads' \
    --exclude='./__pycache__' \
    --exclude='./*.pyc' \
    --exclude='./venv' \
    -czf /tmp/age_verify_app.tar.gz .

$SCP /tmp/age_verify_app.tar.gz "${EC2_USER}@${EC2_HOST}:/tmp/age_verify_app.tar.gz"
$SSH "cd ${APP_DIR} && tar -xzf /tmp/age_verify_app.tar.gz && rm /tmp/age_verify_app.tar.gz"
$SSH "mkdir -p ${APP_DIR}/models ${APP_DIR}/uploads"
echo "Files copied."

# ── Step 3: Set up Python virtualenv and install deps ───────
echo ""
echo "[3/5] Setting up Python virtual environment..."
$SSH "bash -s" <<REMOTE_VENV
set -e
cd ${APP_DIR}
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Python dependencies installed."
REMOTE_VENV

# ── Step 4: Configure Supervisor (process manager) ─────────
echo ""
echo "[4/5] Configuring Supervisor..."
$SSH "sudo bash -s" <<REMOTE_SUPERVISOR
cat > /etc/supervisor/conf.d/age_verify.conf <<EOF
[program:age_verify]
command=/home/${EC2_USER}/age-verify-app/venv/bin/gunicorn \
  --workers 2 \
  --bind 0.0.0.0:${PORT} \
  --timeout 300 \
  --keep-alive 5 \
  --log-level info \
  --access-logfile /var/log/age_verify_access.log \
  --error-logfile /var/log/age_verify_error.log \
  app:app
directory=/home/${EC2_USER}/age-verify-app
user=${EC2_USER}
autostart=true
autorestart=true
environment=
  PORT="${PORT}",
  MODEL_DIR="/home/${EC2_USER}/age-verify-app/models",
  OPENAI_API_KEY=""
stderr_logfile=/var/log/age_verify.err.log
stdout_logfile=/var/log/age_verify.out.log
EOF

supervisorctl reread
supervisorctl update
supervisorctl restart age_verify || supervisorctl start age_verify
echo "Supervisor configured."
REMOTE_SUPERVISOR

# ── Step 5: Configure Nginx reverse proxy ──────────────────
echo ""
echo "[5/5] Configuring Nginx..."
$SSH "sudo bash -s" <<REMOTE_NGINX
cat > /etc/nginx/sites-available/age_verify <<EOF
server {
    listen 80;
    server_name _;

    client_max_body_size 500M;

    location / {
        proxy_pass         http://127.0.0.1:${PORT};
        proxy_set_header   Host \$host;
        proxy_set_header   X-Real-IP \$remote_addr;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 300;
        proxy_send_timeout 300;
    }
}
EOF

ln -sf /etc/nginx/sites-available/age_verify /etc/nginx/sites-enabled/age_verify
rm -f /etc/nginx/sites-enabled/default
nginx -t && sudo systemctl reload nginx
echo "Nginx configured."
REMOTE_NGINX

echo ""
echo "======================================================"
echo "  DEPLOYMENT COMPLETE!"
echo "  App URL: http://${EC2_HOST}"
echo ""
echo "  Routes:"
echo "    http://${EC2_HOST}/             — Main verification tool"
echo "    http://${EC2_HOST}/adultvault   — AdultVault platform"
echo "    http://${EC2_HOST}/royalbet     — RoyalBet platform"
echo "    http://${EC2_HOST}/spiritshop   — SpiritShop platform"
echo "    http://${EC2_HOST}/api/predict  — REST API (POST)"
echo ""
echo "  Logs:"
echo "    sudo tail -f /var/log/age_verify_error.log"
echo "    sudo tail -f /var/log/age_verify.err.log"
echo ""
echo "  Set OpenAI key for VLM reasoning:"
echo "    sudo nano /etc/supervisor/conf.d/age_verify.conf"
echo "    # Set OPENAI_API_KEY=\"sk-...\" in environment"
echo "    sudo supervisorctl restart age_verify"
echo "======================================================"
