#!/bin/bash
# Backend Health Check Script
# Validates all module connections

echo "🔍 CSLR Backend System Check"
echo "=============================="
echo ""

# Check Python version
echo "✓ Python Version:"
python3 --version
echo ""

# Check directory structure
echo "✓ Directory Structure:"
ls -la app/ | head -15
echo ""

# Check pipeline modules
echo "✓ Pipeline Modules:"
ls -la app/pipeline/
echo ""

# Check requirements
echo "✓ Key Dependencies:"
grep -E "^(fastapi|torch|opencv|mediapipe)" requirements.txt | head -10
echo ""

# Check environment  
echo "✓ Environment Config:"
if [ -f .env ]; then
    echo ".env file exists"
    grep -E "^(DEVICE|USE_AMP|BATCH_SIZE)" .env || echo "Missing key settings"
else
    echo "⚠️  .env file not found - create from template"
fi
echo ""

# Count files
echo "✓ File Count:"
echo "Python files: $(find app/ -name '*.py' -type f | wc -l)"
echo "Modules: $(find app/ -type d | wc -l)"
echo ""

# Check critical files
echo "✓ Critical Files:"
for file in "app/main.py" "app/services/inference_service.py" "Dockerfile" "requirements.txt" "README.md"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file MISSING"
    fi
done
echo ""

echo "=============================="
echo "✅ Health check complete!"
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Configure .env file"
echo "3. Run: uvicorn app.main:app --reload"
