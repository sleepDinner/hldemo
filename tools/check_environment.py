import subprocess
import sys


def run_command(command):
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return "command not found"
    output = (result.stdout + result.stderr).strip()
    return output if output else f"exit_code={result.returncode}"


def main():
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print("nvidia-smi:")
    print(run_command(["nvidia-smi"]))

    try:
        import torch
    except ImportError as exc:
        print(f"torch_import_error: {exc}")
        return

    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_build: {torch.version.cuda}")
    try:
        cuda_available = torch.cuda.is_available()
        print(f"torch_cuda_available: {cuda_available}")
        if cuda_available:
            print(f"cuda_device_count: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                print(f"cuda_device_{index}: {torch.cuda.get_device_name(index)}")
    except RuntimeError as exc:
        print("torch_cuda_runtime_error:")
        print(exc)
        print(
            "Likely cause: installed PyTorch CUDA wheel requires a newer NVIDIA driver. "
            "Install a CUDA 12.1 PyTorch build on this server, or upgrade the NVIDIA driver."
        )


if __name__ == "__main__":
    main()
