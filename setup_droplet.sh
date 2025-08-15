#!/bin/bash

# DigitalOcean Droplet Setup Script for IB Trading System
# This script sets up the environment to run IB Gateway and the trading system

set -e  # Exit on any error

echo "🚀 Setting up DigitalOcean droplet for IB trading system..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required packages
echo "📦 Installing required packages..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    openjdk-11-jdk \
    screen \
    curl \
    wget \
    unzip \
    git

# Create trading user (optional but recommended)
echo "👤 Creating trading user..."
sudo useradd -m -s /bin/bash trading || echo "User trading already exists"
sudo usermod -aG sudo trading

# Switch to trading user
echo "🔄 Switching to trading user..."
sudo -u trading bash << 'EOF'

# Create trading directory
mkdir -p ~/trading_system
cd ~/trading_system

# Clone or copy your trading code here
# git clone your-repo-url . || echo "Repository already exists"

# Create Python virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create IB Gateway directory
echo "🏦 Setting up IB Gateway..."
mkdir -p ~/IBJts/ibgateway
cd ~/IBJts/ibgateway

# Download IB Gateway (you'll need to manually download and upload)
echo "⚠️  IMPORTANT: You need to manually download IB Gateway and upload it to this directory"
echo "   Download from: https://www.interactivebrokers.com/en/trading/ib-api.php"
echo "   Upload to: ~/IBJts/ibgateway/"

# Create IB Gateway configuration
echo "⚙️  Creating IB Gateway configuration..."
cat > ibgateway.properties << 'IBGATEWAY_CONFIG'
# IB Gateway Configuration
ibgateway.port=4002
ibgateway.paper=true
ibgateway.live=false
ibgateway.headless=true

# Logging
ibgateway.log.level=INFO
ibgateway.log.file=ibgateway.log

# Connection settings
ibgateway.connection.timeout=60
ibgateway.connection.retry=3

# API settings
ibgateway.api.port=4002
ibgateway.api.allow.connections=true
ibgateway.api.readonly=false
IBGATEWAY_CONFIG

# Create startup script for IB Gateway
echo "📜 Creating IB Gateway startup script..."
cat > start_ibgateway.sh << 'STARTUP_SCRIPT'
#!/bin/bash

# IB Gateway Startup Script
cd ~/IBJts/ibgateway

# Check if IB Gateway jar exists
if [ ! -f "ibgateway.jar" ]; then
    echo "❌ IB Gateway jar file not found!"
    echo "   Please download IB Gateway and place ibgateway.jar in this directory"
    exit 1
fi

echo "🏦 Starting IB Gateway in headless mode..."
echo "   Port: 4002 (Paper Trading)"
echo "   Log file: ibgateway.log"

# Start IB Gateway in headless mode
java -jar ibgateway.jar \
    --headless \
    --port=4002 \
    --properties=ibgateway.properties \
    > ibgateway.log 2>&1 &

echo "✅ IB Gateway started with PID: $!"
echo "   Check logs with: tail -f ibgateway.log"
STARTUP_SCRIPT

chmod +x start_ibgateway.sh

# Create trading system startup script
echo "📜 Creating trading system startup script..."
cat > ~/trading_system/start_trading.sh << 'TRADING_SCRIPT'
#!/bin/bash

# Trading System Startup Script
cd ~/trading_system

# Activate virtual environment
source venv/bin/activate

# Check if IB Gateway is running
if ! pgrep -f "ibgateway.jar" > /dev/null; then
    echo "❌ IB Gateway is not running!"
    echo "   Please start IB Gateway first: ~/IBJts/ibgateway/start_ibgateway.sh"
    exit 1
fi

echo "📈 Starting trading system..."
echo "   Mode: Paper Trading"
echo "   Port: 4002"

# Start trading system
python live_trader.py
TRADING_SCRIPT

chmod +x ~/trading_system/start_trading.sh

# Create systemd service for IB Gateway (optional)
echo "🔧 Creating systemd service for IB Gateway..."
sudo tee /etc/systemd/system/ibgateway.service > /dev/null << 'SYSTEMD_SERVICE'
[Unit]
Description=IB Gateway
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/home/trading/IBJts/ibgateway
ExecStart=/usr/bin/java -jar ibgateway.jar --headless --port=4002
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SYSTEMD_SERVICE

# Create monitoring script
echo "📊 Creating monitoring script..."
cat > ~/trading_system/monitor.sh << 'MONITOR_SCRIPT'
#!/bin/bash

# Trading System Monitoring Script
echo "📊 Trading System Status"
echo "========================"

# Check IB Gateway
if pgrep -f "ibgateway.jar" > /dev/null; then
    echo "✅ IB Gateway: RUNNING"
    IB_PID=$(pgrep -f "ibgateway.jar")
    echo "   PID: $IB_PID"
else
    echo "❌ IB Gateway: NOT RUNNING"
fi

# Check trading system
if pgrep -f "live_trader.py" > /dev/null; then
    echo "✅ Trading System: RUNNING"
    TRADING_PID=$(pgrep -f "live_trader.py")
    echo "   PID: $TRADING_PID"
else
    echo "❌ Trading System: NOT RUNNING"
fi

# Check ports
echo ""
echo "🌐 Port Status:"
if netstat -tlnp 2>/dev/null | grep ":4002" > /dev/null; then
    echo "✅ Port 4002: LISTENING (IB Gateway)"
else
    echo "❌ Port 4002: NOT LISTENING"
fi

# Show recent logs
echo ""
echo "📋 Recent IB Gateway Logs:"
tail -5 ~/IBJts/ibgateway/ibgateway.log 2>/dev/null || echo "   No log file found"

echo ""
echo "📋 Recent Trading System Logs:"
tail -5 ~/trading_system/logs/live_trader.log 2>/dev/null || echo "   No log file found"
MONITOR_SCRIPT

chmod +x ~/trading_system/monitor.sh

echo "✅ Setup complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Download IB Gateway from Interactive Brokers"
echo "2. Upload ibgateway.jar to ~/IBJts/ibgateway/"
echo "3. Configure IB Gateway credentials (run locally first)"
echo "4. Start IB Gateway: ~/IBJts/ibgateway/start_ibgateway.sh"
echo "5. Start trading system: ~/trading_system/start_trading.sh"
echo "6. Monitor: ~/trading_system/monitor.sh"
echo ""
echo "🔐 Authentication Note:"
echo "   You'll need to log into IB Gateway locally first to save credentials"
echo "   Then transfer the configuration to the droplet"

EOF

echo "🎉 DigitalOcean droplet setup complete!"
echo ""
echo "📋 Quick Commands:"
echo "  SSH into droplet: ssh user@your-droplet-ip"
echo "  Switch to trading user: sudo su - trading"
echo "  Start IB Gateway: ~/IBJts/ibgateway/start_ibgateway.sh"
echo "  Start trading: ~/trading_system/start_trading.sh"
echo "  Monitor: ~/trading_system/monitor.sh"
