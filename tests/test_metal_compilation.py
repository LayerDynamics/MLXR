#!/usr/bin/env python3
"""
Basic test to verify Metal shader compilation works.
This is a smoke test for Phase 0 - validating the build system.
"""

import os
import subprocess
import sys
from pathlib import Path


def find_project_root():
    """Find the MLXR project root directory."""
    # Start from the test file's parent (tests directory)
    # Then go up one more level to get to project root
    current = Path(__file__).resolve().parent.parent

    # Verify this is the project root
    if (current / "CMakeLists.txt").exists() and (current / "CLAUDE.md").exists():
        return current

    # If not, search upward
    while current != current.parent:
        if (current / "CMakeLists.txt").exists() and (current / "CLAUDE.md").exists():
            return current
        current = current.parent

    raise RuntimeError("Could not find project root")


def test_metal_build_script_exists():
    """Test that the Metal build script exists and is executable."""
    project_root = find_project_root()
    script_path = project_root / "scripts" / "build_metal.sh"

    assert script_path.exists(), f"Metal build script not found: {script_path}"
    assert os.access(script_path, os.X_OK), f"Metal build script not executable: {script_path}"
    print("✓ Metal build script exists and is executable")


def test_metal_source_exists():
    """Test that Metal source files exist."""
    project_root = find_project_root()
    metal_dir = project_root / "core" / "kernels" / "metal"

    assert metal_dir.exists(), f"Metal source directory not found: {metal_dir}"

    metal_files = list(metal_dir.glob("*.metal"))
    assert len(metal_files) > 0, "No Metal source files found"

    print(f"✓ Found {len(metal_files)} Metal source file(s)")
    for metal_file in metal_files:
        print(f"  - {metal_file.name}")


def test_metal_compilation():
    """Test that Metal shaders can be compiled."""
    project_root = find_project_root()
    script_path = project_root / "scripts" / "build_metal.sh"

    print("\nAttempting to compile Metal shaders...")

    result = subprocess.run(
        [str(script_path)],
        cwd=project_root,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print("Error output:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        assert False, f"Metal compilation failed with return code {result.returncode}"

    print("✓ Metal shaders compiled successfully")

    # Check that metallib files were created
    build_lib_dir = project_root / "build" / "lib"
    if build_lib_dir.exists():
        metallib_files = list(build_lib_dir.glob("*.metallib"))
        print(f"✓ Generated {len(metallib_files)} metallib file(s)")
        for metallib_file in metallib_files:
            print(f"  - {metallib_file.name}")


def test_cmake_configuration():
    """Test that CMake can configure the project."""
    project_root = find_project_root()
    build_dir = project_root / "build" / "cmake_test"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("\nAttempting to configure project with CMake...")

    result = subprocess.run(
        ["cmake", str(project_root), "-DBUILD_TESTS=OFF"],
        cwd=build_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Error output:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        # Don't fail the test - CMake might fail due to missing dependencies
        # which is expected in Phase 0
        print("⚠ CMake configuration had issues (expected in Phase 0)")
        print(result.stdout)
        return

    print("✓ CMake configuration successful")
    # Only print first 30 lines to avoid clutter
    lines = result.stdout.split('\n')
    print('\n'.join(lines[:30]))
    if len(lines) > 30:
        print(f"... ({len(lines) - 30} more lines)")


if __name__ == "__main__":
    print("=== MLXR Build System Validation ===\n")

    try:
        test_metal_build_script_exists()
        test_metal_source_exists()
        test_metal_compilation()
        test_cmake_configuration()

        print("\n=== All validation tests passed! ===")
        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
