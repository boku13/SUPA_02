import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch is not installed")
except Exception as e:
    print(f"Error importing PyTorch: {e}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("Transformers is not installed")
except Exception as e:
    print(f"Error importing Transformers: {e}")

try:
    import peft
    print(f"PEFT version: {peft.__version__}")
except ImportError:
    print("PEFT is not installed")
except Exception as e:
    print(f"Error importing PEFT: {e}") 