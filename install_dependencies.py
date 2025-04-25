"""
Script to install required dependencies for the C++ Code Analyzer web application
"""

import subprocess
import sys
import os
import time
import platform
import re

def install_dependencies():
    """Install required dependencies from requirements.txt"""
    print("Installing required dependencies...")

    # Check Python version
    py_version = sys.version_info
    print(f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major == 3 and py_version.minor >= 12:
        print("Detected Python 3.12+. Adjusting package versions for compatibility...")

    # Get the path to the requirements.txt file
    requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')

    if not os.path.exists(requirements_path):
        print(f"Error: requirements.txt not found at {requirements_path}")
        return False
    
    # Set environment variables to prevent HuggingFace authentication prompts
    env = os.environ.copy()
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    
    # Process requirements file and adjust versions if needed
    adjusted_requirements = adjust_requirements_for_python_version(requirements_path, py_version)
    
    # First attempt: Try with standard options + no-cache-dir to avoid auth issues
    try:
        print("Attempting installation with --no-cache-dir and --prefer-binary...")
        # Create a temporary adjusted requirements file
        temp_req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adjusted_requirements.txt')
        with open(temp_req_path, 'w') as f:
            for req in adjusted_requirements:
                f.write(f"{req}\n")
        
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            '--no-cache-dir', '--prefer-binary', '-r', temp_req_path
        ], env=env)
        
        # Clean up temporary file
        os.remove(temp_req_path)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("Trying to install packages individually...")
        # Clean up temporary file if it exists
        if os.path.exists(temp_req_path):
            os.remove(temp_req_path)

    # Try installing packages one by one
    try:
        success_count = 0
        for req in adjusted_requirements:
            try:
                print(f"Installing {req}...")
                
                # Special handling for known problematic packages
                if "sentencepiece" in req:
                    if install_sentencepiece():
                        success_count += 1
                        print(f"Successfully installed {req}")
                        continue
                
                # Try with --no-cache-dir and --prefer-binary
                try:
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install',
                        '--no-cache-dir', '--prefer-binary', req
                    ], env=env)
                    success_count += 1
                    print(f"Successfully installed {req}")
                except subprocess.CalledProcessError:
                    # For modelscope, try with special handling
                    if "modelscope" in req:
                        try:
                            subprocess.check_call([
                                sys.executable, '-m', 'pip', 'install', 
                                '--no-cache-dir',
                                '--index-url', 'https://pypi.org/simple',
                                '--extra-index-url', 'https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html',
                                '--prefer-binary', '--no-deps', req
                            ], env=env)
                            success_count += 1
                            print(f"Successfully installed {req} with special handling")
                        except subprocess.CalledProcessError:
                            raise Exception(f"Failed to install {req}")
                    else:
                        # Final attempt: Try with --only-binary flag
                        try:
                            subprocess.check_call([
                                sys.executable, '-m', 'pip', 'install',
                                '--no-cache-dir', '--only-binary=:all:', req
                            ], env=env)
                            success_count += 1
                            print(f"Successfully installed {req} with --only-binary")
                        except subprocess.CalledProcessError:
                            raise Exception(f"Failed to install {req}")
            except Exception as e:
                print(f"Failed to install {req}: {e}")
                print("Continuing with other packages...")

        if success_count == len(adjusted_requirements):
            print("All dependencies installed successfully!")
            return True
        else:
            print(f"Installed {success_count} out of {len(adjusted_requirements)} packages.")
            show_build_tools_instructions()
            return False
    except Exception as e:
        print(f"Error during individual package installation: {e}")
        show_build_tools_instructions()
        return False

def adjust_requirements_for_python_version(requirements_path, py_version):
    """Adjust package versions based on Python version compatibility"""
    adjusted_requirements = []
    
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('--')]
    
    for req in requirements:
        # Python 3.12+ compatibility adjustments
        if py_version.major == 3 and py_version.minor >= 12:
            # Handle numpy specifically for Python 3.12+
            if req.startswith("numpy==") or req.startswith("numpy>=") or req.startswith("numpy<="):
                version_match = re.search(r'([<>=]+)(\d+\.\d+\.\d+)', req)
                if version_match:
                    operator, version = version_match.groups()
                    version_parts = [int(x) for x in version.split('.')]
                    
                    # For Python 3.12, numpy needs to be at least 1.26.0
                    if version_parts[0] == 1 and (version_parts[1] < 26 or (version_parts[1] == 26 and version_parts[2] < 0)):
                        print(f"Adjusting {req} to numpy>=1.26.0 for Python 3.12+ compatibility")
                        req = "numpy>=1.26.0"
            
            # Handle setuptools issue for Python 3.12+
            elif req.startswith("setuptools==") or req.startswith("setuptools>=") or req.startswith("setuptools<="):
                print(f"Adjusting {req} to setuptools>=68.0.0 for Python 3.12+ compatibility")
                req = "setuptools>=68.0.0"
                
            # Handle other problematic packages
            elif req.startswith("pkgutil_resolve_name==") or req.startswith("pkgutil_resolve_name>=") or req.startswith("pkgutil_resolve_name<="):
                print(f"Keeping {req} with minimum version for Python 3.12+ compatibility")
            
            # You can add more package-specific adjustments here
        
        adjusted_requirements.append(req)
    
    return adjusted_requirements

def install_sentencepiece():
    """Special handling for sentencepiece package which often fails to build"""
    system = platform.system().lower()
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    
    print("Attempting to install sentencepiece with special handling...")
    
    # For Python 3.12, use a newer version of sentencepiece
    if sys.version_info.major == 3 and sys.version_info.minor >= 12:
        sentencepiece_version = "0.2.0"
        print(f"Using sentencepiece=={sentencepiece_version} for Python 3.12+ compatibility")
    else:
        sentencepiece_version = "0.1.99"
    
    try:
        # Try to find a pre-compiled wheel first
        if system == "windows":
            # Try installing from a wheel directly if available
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '--no-cache-dir',
                '--index-url', 'https://pypi.org/simple',
                '--extra-index-url', 'https://download.pytorch.org/whl/torch_stable.html',
                '--only-binary=:all:',
                f'sentencepiece=={sentencepiece_version}'
            ])
            return True
        else:
            # On non-Windows systems, we can try regular install
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '--no-cache-dir',
                f'sentencepiece=={sentencepiece_version}'
            ])
            return True
    except subprocess.CalledProcessError:
        try:
            # Try with no binary constraints as a fallback
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                '--no-cache-dir', '--no-deps',
                f'sentencepiece=={sentencepiece_version}'
            ])
            return True
        except subprocess.CalledProcessError:
            print("Failed to install sentencepiece automatically.")
            print("You may need to install it manually after installing build tools.")
            return False

def show_build_tools_instructions():
    """Show instructions for installing build tools based on platform"""
    system = platform.system().lower()
    print("\nSome packages could not be installed. You need to install build tools:")
    
    if system == "windows":
        print("Windows: Install Visual C++ Build Tools:")
        print("1. Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print("2. Run the installer and select 'C++ build tools'")
        print("3. Make sure to select the Windows 10 SDK and the latest MSVC Compiler")
        print("4. After installation, restart your computer and try again")
    elif system == "linux":
        print("Linux: Run the following commands:")
        print("sudo apt-get update")
        print("sudo apt-get install -y build-essential python3-dev")
    elif system == "darwin":  # Mac OS
        print("Mac: Run the following command:")
        print("xcode-select --install")
    
    print("\nAfter installing build tools, you may need to manually install problematic packages:")
    print("pip install sentencepiece==0.1.99 --no-cache-dir")

if __name__ == "__main__":
    success = install_dependencies()
    if success:
        print("\nYou can now run the web application using:")
        print("python run_web.py")

        # Verify imports
        print("\nVerifying package imports...")
        try:
            subprocess.check_call([sys.executable, 'test_imports.py'])
            print("All packages imported successfully!")
        except subprocess.CalledProcessError:
            print("Some packages could not be imported. The installation may be incomplete.")
            print("Please check the error messages above and install any missing dependencies.")
    else:
        print("\nFailed to install all dependencies. Please try installing them manually:")
        print("pip install -r requirements.txt")

