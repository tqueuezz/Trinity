#!/usr/bin/env python3
"""
DeepCode CLI - Complete Command Line Launcher
DeepCode CLI - 完整命令行启动器

🧬 Open-Source Code Agent by Data Intelligence Lab @ HKU (CLI Edition)  
⚡ Revolutionizing research reproducibility through collaborative AI via command line

这是CLI版本的主入口点，提供与DeepCode web版本完全相同的功能，
但通过命令行界面而非Web界面运行。
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查必要的依赖是否已安装 / Check if necessary dependencies are installed"""
    print("🔍 Checking CLI dependencies...")
    
    # 首先设置路径
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    missing_deps = []
    
    try:
        import asyncio
        print("✅ Asyncio is available")
    except ImportError:
        missing_deps.append("asyncio")
    
    try:
        import yaml
        print("✅ PyYAML is installed")
    except ImportError:
        missing_deps.append("pyyaml")
    
    try:
        import tkinter
        print("✅ Tkinter is available (for file dialogs)")
    except ImportError:
        print("⚠️  Tkinter not available - file dialogs will use manual input")
    
    # Check for MCP agent dependencies
    try:
        from mcp_agent.app import MCPApp
        print("✅ MCP Agent framework is available")
    except ImportError:
        missing_deps.append("mcp-agent")
    
    # Check for workflow dependencies
    try:
        from workflows.agent_orchestration_engine import execute_multi_agent_research_pipeline
        print("✅ Workflow modules are available")
    except ImportError:
        print("⚠️  Workflow modules may not be properly configured")
    
    # Check for CLI components
    try:
        from cli.cli_app import main as cli_main
        print("✅ CLI application components are available")
    except ImportError as e:
        print(f"⚠️  CLI application components have import issues: {e}")
        print("✅ CLI files exist, attempting to continue...")
    
    if missing_deps:
        print("\n❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies using:")
        print(f"pip install {' '.join([d for d in missing_deps if d != 'cli-components'])}")
        return False
    
    print("✅ All CLI dependencies satisfied")
    return True

def cleanup_cache():
    """清理Python缓存文件 / Clean up Python cache files"""
    try:
        print("🧹 Cleaning up cache files...")
        # 清理__pycache__目录
        os.system('find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null')
        # 清理.pyc文件
        os.system('find . -name "*.pyc" -delete 2>/dev/null')
        print("✅ Cache cleanup completed")
    except Exception as e:
        print(f"⚠️  Cache cleanup failed: {e}")

def print_banner():
    """显示CLI启动横幅 / Display CLI startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║    🧬 DeepCode - Open-Source Code Agent                                              ║
║                                                                                      ║
║    ⚡ DATA INTELLIGENCE LAB @ HKU ⚡                                                ║
║                                                                                      ║
║    Revolutionizing research reproducibility through collaborative AI                ║
║    Same functionality as Web UI, optimized for terminal users                       ║
║                                                                                      ║
║    📋 KEY FEATURES:                                                                  ║
║    • Automated Code Reproduction      • Multi-Agent Architecture                   ║
║    • Extensible & Open-Source        • Growing Research Community                  ║
║    • Future: One Sentence → Codebase • Join Our Mission                           ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """主函数 / Main function"""
    print_banner()
    
    # 检查依赖 / Check dependencies
    if not check_dependencies():
        print("\n🚨 Please install missing dependencies and try again.")
        print("💡 For installation help, please check the project documentation.")
        sys.exit(1)
    
    # 获取当前脚本目录 / Get current script directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    cli_app_path = current_dir / "cli_app.py"
    
    # 检查cli_app.py是否存在 / Check if cli_app.py exists
    if not cli_app_path.exists():
        print(f"❌ CLI application file not found: {cli_app_path}")
        print("Please ensure the cli/cli_app.py file exists.")
        sys.exit(1)
    
    print(f"\n📁 Project root: {project_root}")
    print(f"📁 CLI App location: {cli_app_path}")
    print("🖥️  Starting DeepCode CLI interface...")
    print("🚀 Initializing command line application")
    print("=" * 90)
    print("💡 Tip: This CLI version provides the same functionality as the Web UI")
    print("🧬 Open-source code agent for revolutionizing research reproducibility")
    print("🔄 Progress tracking and results display optimized for terminal")
    print("🛑 Press Ctrl+C to exit at any time")
    print("=" * 90)
    
    # 启动CLI应用 / Launch CLI application
    try:
        # 确保项目根目录在Python路径中
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # 导入并运行CLI应用
        from cli.cli_app import main as cli_main
        
        print("\n🎯 Launching CLI application...")
        print("🎨 Loading enhanced terminal interface...")
        print("⚙️  Initializing AI research engine...")
        
        # 运行主CLI应用
        import asyncio
        asyncio.run(cli_main())
        
    except KeyboardInterrupt:
        print("\n\n🛑 DeepCode CLI stopped by user")
        print("Thank you for using DeepCode CLI! 🧬")
        print("🌟 Your research automation journey continues...")
    except ImportError as e:
        print(f"\n❌ Failed to import CLI application: {e}")
        print("Please check if all modules are properly installed.")
        print("💡 Try running the dependency check again or reinstalling the package.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your Python environment and try again.")
        print("💡 For support, please check the project documentation or GitHub issues.")
        sys.exit(1)
    finally:
        # 清理缓存文件 / Clean up cache files
        cleanup_cache()

if __name__ == "__main__":
    main() 