"""
Simple Flask trading dashboard.
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import logging
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_layer.models.user import User, UserManager
from web_layer.services.ibkr_service import IBKRService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize IBKR service
ibkr_service = IBKRService()

# Initialize user manager
user_manager = UserManager()

# Global variables to track server status and process
server_running = False
trading_process = None


@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login."""
    return user_manager.get_user(user_id)


@app.route('/')
@login_required
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = user_manager.authenticate_user(username, password)
        if user:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Logout user."""
    logout_user()
    return redirect(url_for('login'))


@app.route('/api/status')
@login_required
def api_status():
    """API endpoint for server and market status."""
    try:
        # Get market status
        market_status = ibkr_service.check_market_status_sync()
        
        # Return combined status
        status = {
            'server_running': server_running,
            'market_status': market_status
        }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            'server_running': server_running,
            'market_status': {
                'markets_open': False,
                'current_time_est': 'Error',
                'error': str(e)
            }
        })


@app.route('/api/server-control', methods=['POST'])
@login_required
def api_server_control():
    """API endpoint for server control."""
    global server_running, trading_process
    
    if not current_user.has_permission('modify_parameters'):
        return jsonify({'error': 'Insufficient permissions'}), 403
    
    try:
        data = request.get_json()
        action = data.get('action')
        
        if not action:
            return jsonify({'error': 'Action is required'}), 400
        
        if action == 'start':
            # Check if markets are open before starting
            market_status = ibkr_service.check_market_status_sync()
            
            if not market_status.get('markets_open', False):
                return jsonify({
                    'success': False,
                    'error': f"Markets are closed. {market_status.get('next_event', '')}"
                })
            
            # Start the trading server
            if not server_running:
                import subprocess
                import sys
                
                # Start live_trader.py in a subprocess (paper mode by default)
                trading_process = subprocess.Popen(
                    [sys.executable, 'live_trader.py', '--paper'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,  # Redirect stdin to avoid tty input issues
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                server_running = True
                logger.info("Trading server started")
            
            return jsonify({
                'success': True,
                'message': 'Trading server started successfully'
            })
            
        elif action == 'stop':
            # Stop the trading server
            if server_running and trading_process:
                trading_process.terminate()
                try:
                    trading_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    trading_process.kill()
                trading_process = None
                server_running = False
                logger.info("Trading server stopped")
            
            return jsonify({
                'success': True,
                'message': 'Trading server stopped successfully'
            })
        else:
            return jsonify({'error': 'Invalid action'}), 400
            
    except Exception as e:
        logger.error(f"Error controlling server: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/console-output')
@login_required
def api_console_output():
    """API endpoint for console output."""
    global trading_process
    
    try:
        if trading_process and server_running:
            # Read available output from the process (non-blocking)
            import select
            
            output = ""
            # Check if there's data available to read
            if select.select([trading_process.stdout], [], [], 0.1)[0]:
                # Read all available lines
                while True:
                    line = trading_process.stdout.readline()
                    if not line:
                        break
                    output += line
            
            return jsonify({'output': output})
        else:
            return jsonify({'output': ''})
    except Exception as e:
        logger.error(f"Error reading console output: {e}")
        return jsonify({'output': f'Error: {str(e)}'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
