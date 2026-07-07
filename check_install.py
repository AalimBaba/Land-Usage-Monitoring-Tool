from __future__ import annotations

from pathlib import Path


def main() -> None:
    packages = ["numpy", "PIL", "streamlit", "tensorflow", "sklearn", "matplotlib"]
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "installed")
            print(f"{package}: {version}")
        except Exception as exc:
            print(f"{package}: missing or failed to import ({exc})")
    model_path = Path("unet_model.h5")
    print(f"model file: {'found' if model_path.exists() else 'missing'} ({model_path.resolve()})")


if __name__ == "__main__":
    main()
