#!/usr/bin/env python3
"""
Setup script for Ollama + Custom Parser Hybrid System
Installs and configures Ollama with vision models for cost-effective AI descriptions
"""

import os
import sys
import subprocess
import requests
import time
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaSetup:
    """Setup and configuration for Ollama system"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.recommended_models = [
            "llava:7b",      # Best balance of quality/speed
            "llava:13b",     # Higher quality, slower
            "bakllava:7b",   # Alternative vision model
        ]
        self.selected_model = "llava:7b"  # Default choice
    
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements for Ollama"""
        
        logger.info("Checking system requirements...")
        
        # Check available RAM
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            for line in meminfo.split('\n'):
                if 'MemTotal:' in line:
                    total_ram_kb = int(line.split()[1])
                    total_ram_gb = total_ram_kb / (1024 * 1024)
                    break
            
            logger.info(f"Total RAM: {total_ram_gb:.1f} GB")
            
            if total_ram_gb < 8:
                logger.warning("⚠️  Less than 8GB RAM detected. Performance may be limited.")
                logger.info("Consider using smaller models or cloud deployment.")
                return False
            elif total_ram_gb >= 16:
                logger.info("✅ Excellent RAM for running larger models")
                self.selected_model = "llava:13b"
            else:
                logger.info("✅ Sufficient RAM for standard models")
                
        except Exception as e:
            logger.warning(f"Could not check RAM: {e}")
        
        # Check disk space
        try:
            disk_usage = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
            logger.info("Disk space check:")
            logger.info(disk_usage.stdout.split('\n')[1])
        except:
            logger.warning("Could not check disk space")
        
        # Check if GPU is available
        try:
            gpu_check = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if gpu_check.returncode == 0:
                logger.info("✅ NVIDIA GPU detected - will enable GPU acceleration")
                return True
            else:
                logger.info("ℹ️  No NVIDIA GPU detected - will use CPU")
        except:
            logger.info("ℹ️  No NVIDIA GPU detected - will use CPU")
        
        return True
    
    def install_ollama(self) -> bool:
        """Install Ollama if not already installed"""
        
        logger.info("Checking Ollama installation...")
        
        # Check if Ollama is already installed
        try:
            result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Ollama already installed: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        logger.info("Installing Ollama...")
        
        try:
            # Download and install Ollama
            install_cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
            result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Ollama installed successfully")
                return True
            else:
                logger.error(f"❌ Ollama installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Installation error: {str(e)}")
            return False
    
    def start_ollama_service(self) -> bool:
        """Start Ollama service"""
        
        logger.info("Starting Ollama service...")
        
        try:
            # Check if service is already running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama service already running")
                return True
        except:
            pass
        
        try:
            # Start Ollama service in background
            subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            for i in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        logger.info("✅ Ollama service started successfully")
                        return True
                except:
                    time.sleep(1)
            
            logger.error("❌ Ollama service failed to start within 30 seconds")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to start Ollama service: {str(e)}")
            return False
    
    def download_vision_model(self, model_name: str = None) -> bool:
        """Download and setup vision model"""
        
        if model_name is None:
            model_name = self.selected_model
        
        logger.info(f"Downloading vision model: {model_name}")
        logger.info("This may take several minutes depending on your internet connection...")
        
        try:
            # Pull the model
            result = subprocess.run(['ollama', 'pull', model_name], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Model {model_name} downloaded successfully")
                return True
            else:
                logger.error(f"❌ Failed to download model: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model download error: {str(e)}")
            return False
    
    def test_vision_model(self, model_name: str = None) -> bool:
        """Test the vision model with a simple image"""
        
        if model_name is None:
            model_name = self.selected_model
        
        logger.info(f"Testing vision model: {model_name}")
        
        try:
            # Create a simple test image (red square)
            from PIL import Image
            import base64
            from io import BytesIO
            
            # Create test image
            test_image = Image.new('RGB', (100, 100), color='red')
            img_buffer = BytesIO()
            test_image.save(img_buffer, format='JPEG')
            img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Test API call
            payload = {
                "model": model_name,
                "prompt": "Describe this image briefly.",
                "images": [img_b64],
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result.get('response', '').strip()
                logger.info(f"✅ Model test successful!")
                logger.info(f"Test description: {description[:100]}...")
                return True
            else:
                logger.error(f"❌ Model test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model test error: {str(e)}")
            return False
    
    def optimize_ollama_config(self) -> bool:
        """Optimize Ollama configuration for production use"""
        
        logger.info("Optimizing Ollama configuration...")
        
        try:
            # Create Ollama config directory
            config_dir = Path.home() / '.ollama'
            config_dir.mkdir(exist_ok=True)
            
            # Optimize model parameters
            config = {
                "num_ctx": 2048,        # Context length
                "num_predict": 200,     # Max tokens to predict
                "temperature": 0.7,     # Creativity level
                "top_p": 0.9,          # Nucleus sampling
                "repeat_penalty": 1.1,  # Avoid repetition
            }
            
            # Save configuration
            config_file = config_dir / 'config.json'
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("✅ Ollama configuration optimized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration optimization failed: {str(e)}")
            return False
    
    def create_systemd_service(self) -> bool:
        """Create systemd service for automatic startup"""
        
        logger.info("Creating systemd service for Ollama...")
        
        try:
            service_content = """[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=ubuntu
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=3
Environment=OLLAMA_HOST=0.0.0.0:11434

[Install]
WantedBy=multi-user.target
"""
            
            # Write service file
            service_file = '/etc/systemd/system/ollama.service'
            
            with open('ollama.service', 'w') as f:
                f.write(service_content)
            
            # Install service (requires sudo)
            logger.info("Installing systemd service (requires sudo)...")
            subprocess.run(['sudo', 'cp', 'ollama.service', service_file], check=True)
            subprocess.run(['sudo', 'systemctl', 'daemon-reload'], check=True)
            subprocess.run(['sudo', 'systemctl', 'enable', 'ollama'], check=True)
            
            # Clean up
            os.remove('ollama.service')
            
            logger.info("✅ Systemd service created and enabled")
            return True
            
        except Exception as e:
            logger.error(f"❌ Systemd service creation failed: {str(e)}")
            logger.info("You can manually start Ollama with: ollama serve")
            return False
    
    def install_python_dependencies(self) -> bool:
        """Install required Python dependencies"""
        
        logger.info("Installing Python dependencies...")
        
        dependencies = [
            'pillow>=9.0.0',
            'opencv-python>=4.5.0',
            'numpy>=1.21.0',
            'scikit-learn>=1.0.0',
            'requests>=2.25.0',
        ]
        
        try:
            for dep in dependencies:
                logger.info(f"Installing {dep}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                             check=True, capture_output=True)
            
            logger.info("✅ Python dependencies installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Python dependencies installation failed: {str(e)}")
            return False
    
    def create_demo_script(self) -> bool:
        """Create a demo script to test the hybrid system"""
        
        logger.info("Creating demo script...")
        
        demo_script = f"""#!/usr/bin/env python3
\"\"\"
Demo script for Ollama + Custom Parser Hybrid System
\"\"\"

import sys
import os
from pathlib import Path

# Add the hybrid system to path
sys.path.append(str(Path(__file__).parent))

from ollama_hybrid_system import HybridImageAnalyzer
from PIL import Image
from io import BytesIO

def main():
    print("🧠 Ollama + Custom Parser Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = HybridImageAnalyzer(ollama_model="{self.selected_model}")
    
    # Check system status
    status = analyzer.get_system_status()
    print(f"System Status: {{status['system_health']}}")
    print(f"Ollama Available: {{status['ollama_available']}}")
    print(f"Model: {{status['ollama_model']}}")
    print()
    
    # Create test image
    print("Creating test image...")
    test_image = Image.new('RGB', (800, 600), color=(70, 130, 180))  # Steel blue
    img_byte_arr = BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    image_data = img_byte_arr.getvalue()
    
    # Analyze image
    print("Analyzing image...")
    result = analyzer.analyze_image_complete(image_data, "test_blue_image.jpg")
    
    # Display results
    print("\\n📊 Analysis Results:")
    print("-" * 30)
    print(f"Overall Quality Score: {{result['quality_analysis']['overall_score']}}")
    print(f"Processing Time: {{result['processing_metadata']['total_time']}} seconds")
    print(f"AI Model Used: {{result['processing_metadata']['ai_model']}}")
    print(f"Cost per Image: ${{result['cost_analysis']['cost_per_image']}}")
    print()
    
    print("🤖 AI Description:")
    print(result['ai_description'])
    print()
    
    print("🏷️  Smart Tags:")
    print(", ".join(result['smart_tags'][:10]))
    print()
    
    print("💡 Suggested Use Cases:")
    for use_case in result['use_cases'][:5]:
        print(f"  • {{use_case}}")
    print()
    
    print("💰 Cost Analysis:")
    print(f"  • Cost per image: ${{result['cost_analysis']['cost_per_image']}}")
    print(f"  • Monthly cost (1000 images): ${{result['cost_analysis']['monthly_cost_1000_images']}}")
    print(f"  • Savings vs OpenAI: ${{result['cost_analysis']['savings_vs_openai']}}")
    print()
    
    print("✅ Demo completed successfully!")

if __name__ == "__main__":
    main()
"""
        
        try:
            with open('ollama_demo.py', 'w') as f:
                f.write(demo_script)
            
            # Make executable
            os.chmod('ollama_demo.py', 0o755)
            
            logger.info("✅ Demo script created: ollama_demo.py")
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo script creation failed: {str(e)}")
            return False
    
    def run_complete_setup(self) -> bool:
        """Run complete setup process"""
        
        logger.info("🚀 Starting Ollama + Custom Parser Setup")
        logger.info("=" * 60)
        
        steps = [
            ("System Requirements Check", self.check_system_requirements),
            ("Python Dependencies", self.install_python_dependencies),
            ("Ollama Installation", self.install_ollama),
            ("Ollama Service Start", self.start_ollama_service),
            ("Vision Model Download", self.download_vision_model),
            ("Model Testing", self.test_vision_model),
            ("Configuration Optimization", self.optimize_ollama_config),
            ("Demo Script Creation", self.create_demo_script),
        ]
        
        success_count = 0
        
        for step_name, step_func in steps:
            logger.info(f"\\n📋 Step: {step_name}")
            logger.info("-" * 40)
            
            try:
                if step_func():
                    success_count += 1
                    logger.info(f"✅ {step_name} completed successfully")
                else:
                    logger.error(f"❌ {step_name} failed")
            except Exception as e:
                logger.error(f"❌ {step_name} failed with error: {str(e)}")
        
        # Optional systemd service (may fail without sudo)
        logger.info("\\n📋 Optional: Systemd Service")
        logger.info("-" * 40)
        try:
            if self.create_systemd_service():
                logger.info("✅ Systemd service created")
            else:
                logger.info("ℹ️  Systemd service creation skipped (manual start required)")
        except:
            logger.info("ℹ️  Systemd service creation skipped (manual start required)")
        
        # Final summary
        logger.info("\\n" + "=" * 60)
        logger.info("🎯 SETUP SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Completed steps: {success_count}/{len(steps)}")
        
        if success_count >= len(steps) - 1:  # Allow 1 failure
            logger.info("✅ Setup completed successfully!")
            logger.info("\\n🚀 Next Steps:")
            logger.info("1. Run the demo: python3 ollama_demo.py")
            logger.info("2. Integrate with your Image RAG SaaS")
            logger.info("3. Monitor performance and costs")
            logger.info(f"\\n💰 Expected monthly cost for 1000 images: ~$1-3")
            logger.info(f"💰 Savings vs OpenAI GPT-4 Vision: ~$15-20/month")
            return True
        else:
            logger.error("❌ Setup completed with errors")
            logger.info("\\n🔧 Troubleshooting:")
            logger.info("1. Check system requirements")
            logger.info("2. Ensure internet connection")
            logger.info("3. Try manual Ollama installation")
            return False


def main():
    \"\"\"Main setup function\"\"\"
    
    setup = OllamaSetup()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--model':
            if len(sys.argv) > 2:
                setup.selected_model = sys.argv[2]
                logger.info(f"Using model: {setup.selected_model}")
        elif sys.argv[1] == '--help':
            print("Ollama Setup Script")
            print("Usage: python3 setup_ollama_system.py [--model MODEL_NAME]")
            print("\\nAvailable models:")
            for model in setup.recommended_models:
                print(f"  • {model}")
            print("\\nDefault model: llava:7b")
            return
    
    # Run setup
    success = setup.run_complete_setup()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

