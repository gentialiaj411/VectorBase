"""
Build Script for MiniVector C++ Core
=====================================
This script builds the minivector_core C++ extension module.
It automatically detects the build environment and uses the best available method.
Usage:
    python build_cpp.py              # Build using CMake
    python build_cpp.py --inplace    # Build and copy to minivector/
    python build_cpp.py --clean      # Clean build artifacts
Requirements:
    - C++17 compatible compiler
    - CMake >= 3.15
    - pybind11 >= 2.10.0
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
def check_requirements():
    """Check if build requirements are met."""
    print("Checking build requirements...")
    try:
        result = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
        cmake_version = result.stdout.split("\n")[0]
        print(f"  ✓ CMake: {cmake_version}")
    except FileNotFoundError:
        print("  ✗ CMake not found. Install from https://cmake.org/")
        return False
    try:
        import pybind11
        print(f"  ✓ pybind11: {pybind11.__version__}")
    except ImportError:
        print("  ✗ pybind11 not found. Install with: pip install pybind11")
        return False
    if platform.system() == "Windows":
        try:
            result = subprocess.run(["cl"], capture_output=True, text=True)
            print("  ✓ MSVC compiler available")
        except FileNotFoundError:
            print("  ! MSVC not in PATH - CMake will attempt to find it")
    else:
        try:
            result = subprocess.run(["g++", "--version"], capture_output=True, text=True)
            compiler = result.stdout.split("\n")[0]
            print(f"  ✓ Compiler: {compiler}")
        except FileNotFoundError:
            try:
                result = subprocess.run(["clang++", "--version"], capture_output=True, text=True)
                compiler = result.stdout.split("\n")[0]
                print(f"  ✓ Compiler: {compiler}")
            except FileNotFoundError:
                print("  ✗ No C++ compiler found. Install GCC or Clang.")
                return False
    return True
def clean_build():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    dirs_to_clean = [
        "build",
        "dist",
        "*.egg-info",
        "minivector_cpp/build",
    ]
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"  Removed: {path}")
    for ext in ["*.pyd", "*.so"]:
        for path in Path("minivector").glob(ext):
            path.unlink()
            print(f"  Removed: {path}")
    print("Clean complete.")
def build_with_cmake(inplace: bool = False):
    """Build using CMake."""
    print("\nBuilding with CMake...")
    src_dir = Path("minivector_cpp").absolute()
    build_dir = src_dir / "build"
    output_dir = Path("minivector").absolute() if inplace else build_dir
    build_dir.mkdir(exist_ok=True)
    import pybind11
    pybind11_cmake = pybind11.get_cmake_dir()
    cmake_args = [
        "cmake",
        str(src_dir),
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={pybind11_cmake}",
    ]
    if platform.system() == "Windows":
        cmake_args.extend(["-A", "x64"])
        cmake_args.append(f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={output_dir}")
        cmake_args.append(f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={output_dir}")
    else:
        cmake_args.append(f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}")
    print(f"  Source dir: {src_dir}")
    print(f"  Build dir: {build_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  CMake args: {' '.join(cmake_args[1:])}")
    print("\n[1/2] Configuring...")
    result = subprocess.run(cmake_args, cwd=build_dir)
    if result.returncode != 0:
        print("CMake configuration failed!")
        return False
    print("\n[2/2] Building...")
    build_args = ["cmake", "--build", ".", "--config", "Release"]
    if platform.system() != "Windows":
        build_args.extend(["--", "-j4"])
    result = subprocess.run(build_args, cwd=build_dir)
    if result.returncode != 0:
        print("Build failed!")
        return False
    print("\nBuild successful!")
    if platform.system() == "Windows":
        module_pattern = "minivector_core*.pyd"
    else:
        module_pattern = "minivector_core*.so"
    for path in output_dir.glob(module_pattern):
        print(f"  Built: {path}")
        if inplace and not path.parent.samefile(Path("minivector")):
            dest = Path("minivector") / path.name
            shutil.copy2(path, dest)
            print(f"  Copied to: {dest}")
    return True
def build_with_pip():
    """Build using pip install."""
    print("\nBuilding with pip...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-e", ".", "-v"
    ])
    return result.returncode == 0
def verify_build():
    """Verify the build by importing the module."""
    print("\nVerifying build...")
    try:
        sys.path.insert(0, str(Path(".").absolute()))
        from minivector import minivector_core as core
        print(f"  ✓ Module imported successfully")
        print(f"  ✓ Version: {core.get_version()}")
        print(f"  ✓ SIMD: {core.detect_simd().name}")
        print(f"  ✓ Build info:\n{core.get_build_info()}")
        import numpy as np
        query = np.random.randint(0, 256, size=48, dtype=np.uint8)
        database = np.random.randint(0, 256, size=(100, 48), dtype=np.uint8)
        indices, distances = core.batch_search(query, database, 5)
        print(f"  ✓ Functional test passed (searched 100 vectors)")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False
def main():
    parser = argparse.ArgumentParser(description="Build MiniVector C++ core")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--inplace", action="store_true", help="Copy module to minivector/")
    parser.add_argument("--pip", action="store_true", help="Use pip install instead of CMake")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    args = parser.parse_args()
    print("=" * 60)
    print("MiniVector C++ Core Build Script")
    print("=" * 60)
    if args.clean:
        clean_build()
        return 0
    if not check_requirements():
        print("\nBuild requirements not met. Please install missing dependencies.")
        return 1
    if args.pip:
        success = build_with_pip()
    else:
        success = build_with_cmake(inplace=args.inplace)
    if not success:
        print("\nBuild failed!")
        return 1
    if not args.no_verify:
        if not verify_build():
            print("\nBuild verification failed!")
            return 1
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print("\nTo use the C++ backend:")
    print("  from minivector.binary_engine import BinaryIndex, get_backend_info")
    print("  print(get_backend_info())")
    print("\nTo run benchmarks:")
    print("  python scripts/benchmark_simd.py")
    print("=" * 60)
    return 0
if __name__ == "__main__":
    sys.exit(main())
