#!/bin/bash

# Navigasi ke folder MuseTalk
cd /home/ubuntu/OmniCastPro/MuseTalk

# Aktifkan virtual environment
if [ -f "musetalk_env/bin/activate" ]; then
    source musetalk_env/bin/activate
else
    echo "Error: musetalk_env tidak ditemukan!"
    exit 1
fi

# Jalankan API di background menggunakan nohup
# > mengarahkan output standar ke log
# 2>&1 menggabungkan error ke log yang sama
# & menjalankan di background
echo "------------------------------------------"
echo "Starting MuseTalk API at $(date)"
echo "------------------------------------------"
nohup python3 api_service.py > api_service.log 2>&1 &

# Simpan Process ID (PID)
echo $! > api_service.pid

echo "✅ MuseTalk API sedang berjalan di background (PID: $(cat api_service.pid))"
echo "📊 Untuk melihat log: tail -f api_service.log"
echo "🛑 Untuk mematikan: kill \$(cat api_service.pid)"
