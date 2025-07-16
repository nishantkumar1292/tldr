#!/usr/bin/env python3
"""
Simple script to run the TLDR web app
"""

import os
import sys
import subprocess

def main():
    """Run the TLDR web app"""

    # Fix OpenMP library conflict on macOS
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Check if OPENAI_API_KEY is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable is not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Check if requirements are installed
    try:
        import fastapi
        import uvicorn
        import jinja2
    except ImportError:
        print("❌ Web app dependencies not installed")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

    # Check if TLDR library is available
    try:
        from tldr import YouTubeSummarizer
        print("✅ TLDR library found")
    except ImportError:
        print("❌ TLDR library not found")
        print("Please install it with: pip install -e ..")
        sys.exit(1)

    print("🚀 Starting TLDR web app...")
    print(f"📱 Open http://localhost:8000 in your browser")
    print("🛑 Press Ctrl+C to stop")

    # Run the app
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

if __name__ == "__main__":
    main()
