"""
Script to run the Flask trading dashboard server.
"""

import sys
import os
import subprocess
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def kill_existing_servers():
    """Kill any existing Flask server instances."""
    logger.info("Checking for existing server instances...")
    try:
        # Find processes running the Flask server
        result = subprocess.run(
            ['pgrep', '-f', 'run_flask_server.py'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            logger.info(f"Found {len(pids)} existing server instance(s)")
            
            for pid in pids:
                if pid.strip():
                    try:
                        logger.info(f"Killing process {pid}")
                        subprocess.run(['kill', '-9', pid], check=True)
                        time.sleep(0.5)  # Give it time to die
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Failed to kill process {pid}: {e}")
        else:
            logger.info("No existing server instances found")
            
    except Exception as e:
        logger.error(f"Error checking for existing servers: {e}")


def main():
    """Main function to run the Flask server."""
    logger.info("Starting Flask trading dashboard server...")
    
    # Kill any existing server instances first
    kill_existing_servers()
    
    logger.info("Server will be available at: http://localhost:8080")
    logger.info("Default users:")
    logger.info("  - admin/admin123 (full access)")
    logger.info("  - trader/trader123 (trading access)")
    logger.info("  - viewer/viewer123 (view-only access)")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        # Import and run the Flask app
        from web_layer.app import app
        app.run(host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        raise


if __name__ == "__main__":
    main()
