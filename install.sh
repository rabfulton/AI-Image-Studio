#!/usr/bin/env bash
#
# AI Image Studio Installer
# Usage: curl -fsSL https://example.com/install.sh | bash
#
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════╗"
echo "║         AI Image Studio Installer             ║"
echo "╚═══════════════════════════════════════════════╝"
echo -e "${NC}"

# Check requirements
check_requirements() {
    local missing=()
    
    if ! command -v python3 &> /dev/null; then
        missing+=("python3")
    fi
    
    if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
        missing+=("pip")
    fi
    
    if ! command -v git &> /dev/null; then
        missing+=("git")
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        echo -e "${RED}Error: Missing required tools: ${missing[*]}${NC}"
        echo "Please install them and try again."
        exit 1
    fi
}

# Detect GPU
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "nvidia"
    elif command -v vulkaninfo &> /dev/null; then
        echo "vulkan"
    elif [ -d "/sys/class/drm" ] && ls /sys/class/drm/card*/device/vendor 2>/dev/null | xargs cat 2>/dev/null | grep -q "0x1002"; then
        echo "amd"
    else
        echo "none"
    fi
}

# Prompt for backend
select_backend() {
    local detected_gpu=$(detect_gpu)
    
    echo -e "${YELLOW}Select compute backend for local image generation:${NC}"
    echo ""
    echo "  1) CPU only (works everywhere, slower)"
    echo "  2) CUDA (NVIDIA GPU, fastest)"
    echo "  3) Vulkan (AMD/NVIDIA/Intel GPU)"
    echo "  4) Skip (install without local generation)"
    echo ""
    
    # Show recommendation
    case $detected_gpu in
        nvidia)
            echo -e "${GREEN}Detected: NVIDIA GPU - CUDA recommended${NC}"
            default="2"
            ;;
        vulkan|amd)
            echo -e "${GREEN}Detected: GPU with Vulkan support${NC}"
            default="3"
            ;;
        *)
            echo -e "${BLUE}No GPU detected - CPU recommended${NC}"
            default="1"
            ;;
    esac
    
    echo ""
    read -p "Enter choice [${default}]: " choice
    choice=${choice:-$default}
    
    case $choice in
        1) echo "cpu" ;;
        2) echo "cuda" ;;
        3) echo "vulkan" ;;
        4) echo "skip" ;;
        *) echo "cpu" ;;
    esac
}

# Install function
install_app() {
    local backend=$1
    local install_dir="${HOME}/.local/share/ai-image-studio"
    local venv_dir="${install_dir}/venv"
    
    echo ""
    echo -e "${CYAN}Installing to: ${install_dir}${NC}"
    
    # Create directory
    mkdir -p "$install_dir"
    cd "$install_dir"
    
    # Clone or update repo
    if [ -d ".git" ]; then
        echo -e "${BLUE}Updating existing installation...${NC}"
        git pull
    else
        echo -e "${BLUE}Cloning repository...${NC}"
        git clone https://github.com/your-org/ai-image-studio.git .
    fi
    
    # Create virtual environment
    echo -e "${BLUE}Setting up Python environment...${NC}"
    python3 -m venv "$venv_dir"
    source "$venv_dir/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install core dependencies
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install -r requirements.txt
    
    # Install stable-diffusion-cpp with selected backend
    if [ "$backend" != "skip" ]; then
        echo -e "${BLUE}Installing stable-diffusion-cpp (${backend})...${NC}"
        
        case $backend in
            cpu)
                pip install stable-diffusion-cpp-python
                ;;
            cuda)
                CMAKE_ARGS="-DSD_CUDA=ON" pip install stable-diffusion-cpp-python
                ;;
            vulkan)
                CMAKE_ARGS="-DSD_VULKAN=ON" pip install stable-diffusion-cpp-python
                ;;
        esac
    fi
    
    # Create launcher script
    cat > "${install_dir}/ai-image-studio" << 'EOF'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/venv/bin/activate"
cd "${SCRIPT_DIR}"
python -m ai_image_studio "$@"
EOF
    chmod +x "${install_dir}/ai-image-studio"
    
    # Create desktop entry
    mkdir -p "${HOME}/.local/share/applications"
    cat > "${HOME}/.local/share/applications/ai-image-studio.desktop" << EOF
[Desktop Entry]
Name=AI Image Studio
Comment=Node-based AI image generation
Exec=${install_dir}/ai-image-studio
Icon=${install_dir}/assets/icon.png
Terminal=false
Type=Application
Categories=Graphics;
EOF
    
    # Add to PATH suggestion
    local bin_dir="${HOME}/.local/bin"
    mkdir -p "$bin_dir"
    ln -sf "${install_dir}/ai-image-studio" "${bin_dir}/ai-image-studio"
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         Installation Complete!                ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Run with: ${CYAN}ai-image-studio${NC}"
    echo -e "Or from: ${CYAN}${install_dir}/ai-image-studio${NC}"
    echo ""
    
    if [ "$backend" != "skip" ]; then
        echo -e "${YELLOW}Tip: Download a model from the app's Provider Settings${NC}"
    fi
}

# Main
main() {
    check_requirements
    backend=$(select_backend)
    install_app "$backend"
}

main "$@"
