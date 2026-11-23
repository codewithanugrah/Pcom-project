#!/usr/bin/env bash
set -e

VM_IP="172.17.72.12"
KEY_PATH="$HOME/.ssh/pcom-key.pem"
REPO_DIR="/home/ubuntu/pcom-course-setup"

echo "=========================================================="
echo "🚀 PCOM-01 VM Deployment Pipeline (Local -> GitHub -> VM)"
echo "=========================================================="

echo "1. Pushing latest local commits and tags to GitHub..."
git push origin main
git push origin --tags --force

echo "2. Pulling latest code on remote VM ($VM_IP)..."
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ubuntu@"$VM_IP" "cd $REPO_DIR && git pull origin main"

echo "3. Restarting pcom-api.service on remote VM..."
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ubuntu@"$VM_IP" "sudo systemctl restart pcom-api.service"

echo "4. Verifying service health..."
sleep 4
ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ubuntu@"$VM_IP" "curl --retry 3 --retry-connrefused -s http://127.0.0.1:8000/health && echo ''"

echo "=========================================================="
echo "✅ Deployment complete! VM is live and ready for Compy."
echo "=========================================================="
