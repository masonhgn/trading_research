"""
Simple test Flask server to isolate SocketIO and market status issues.
"""

from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Test Dashboard</title>
    <script src="https://cdn.socket.io/4.0.1/socket.io.min.js"></script>
</head>
<body>
    <h1>Simple Test Dashboard</h1>
    <div id="status">Status: CHECKING</div>
    <div id="time">Time: --</div>
    <div id="logs"></div>
    
    <script>
        console.log('Initializing Socket.IO...');
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to Socket.IO');
            document.getElementById('logs').innerHTML += '<p>Connected to Socket.IO</p>';
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from Socket.IO');
            document.getElementById('logs').innerHTML += '<p>Disconnected from Socket.IO</p>';
        });
        
        socket.on('market_status', function(data) {
            console.log('Received market status:', data);
            document.getElementById('status').textContent = 'Status: ' + (data.markets_open ? 'OPEN' : 'CLOSED');
            document.getElementById('time').textContent = 'Time: ' + data.current_time_est;
            document.getElementById('logs').innerHTML += '<p>Received market status: ' + JSON.stringify(data) + '</p>';
        });
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/market-status')
def api_market_status():
    """Simple market status API."""
    import datetime
    import pytz
    
    est_tz = pytz.timezone('US/Eastern')
    current_time_est = datetime.datetime.now(est_tz)
    
    is_weekday = current_time_est.weekday() < 5
    market_open = current_time_est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time_est.replace(hour=16, minute=0, second=0, microsecond=0)
    is_market_hours = market_open <= current_time_est <= market_close
    markets_open = is_weekday and is_market_hours
    
    return jsonify({
        'current_time_est': current_time_est.strftime('%Y-%m-%d %H:%M:%S EST'),
        'markets_open': markets_open,
        'is_weekday': is_weekday,
        'is_market_hours': is_market_hours
    })

def emit_market_status():
    """Emit market status every second."""
    while True:
        try:
            import datetime
            import pytz
            
            est_tz = pytz.timezone('US/Eastern')
            current_time_est = datetime.datetime.now(est_tz)
            
            is_weekday = current_time_est.weekday() < 5
            market_open = current_time_est.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = current_time_est.replace(hour=16, minute=0, second=0, microsecond=0)
            is_market_hours = market_open <= current_time_est <= market_close
            markets_open = is_weekday and is_market_hours
            
            data = {
                'current_time_est': current_time_est.strftime('%Y-%m-%d %H:%M:%S EST'),
                'markets_open': markets_open,
                'is_weekday': is_weekday,
                'is_market_hours': is_market_hours
            }
            
            logger.info(f"Emitting market status: {data}")
            socketio.emit('market_status', data)
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in emit loop: {e}")
            time.sleep(5)

if __name__ == '__main__':
    # Start the emit thread
    emit_thread = threading.Thread(target=emit_market_status, daemon=True)
    emit_thread.start()
    logger.info("Started emit thread")
    
    # Run the server
    logger.info("Starting simple test server on http://localhost:8081")
    socketio.run(app, host='0.0.0.0', port=8081, debug=False)
