#!/usr/bin/env python3
"""
Test Runner Script for Phase 1 Features.

Provides convenient commands for running different test suites.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ {description} - FAILED")
        return False
    else:
        print(f"\n✅ {description} - PASSED")
        return True


def main():
    """Main test runner entry point."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    # Base pytest command
    pytest_cmd = [sys.executable, "-m", "pytest", str(script_dir), "-v"]

    # Parse arguments
    args = sys.argv[1:]

    if not args or args[0] == "all":
        """Run all tests"""
        success = run_command(
            pytest_cmd
            + [
                "--cov=ai_inference_gateway",
                "--cov-report=term-missing",
                "--cov-report=html",
            ],
            "All Tests with Coverage",
        )
        return 0 if success else 1

    elif args[0] == "unit":
        """Run unit tests only (fast, isolated)"""
        success = run_command(pytest_cmd + ["-m", "unit", "--tb=short"], "Unit Tests")
        return 0 if success else 1

    elif args[0] == "integration":
        """Run integration tests"""
        success = run_command(pytest_cmd + ["-m", "integration"], "Integration Tests")
        return 0 if success else 1

    elif args[0] == "phase1":
        """Run Phase 1 feature tests only"""
        phase1_tests = [
            "test_response_format.py",
            "test_mcp_cache.py",
            "test_pii_redactor.py",
            "test_moderation.py",
        ]

        success = run_command(
            pytest_cmd
            + phase1_tests
            + [
                "--cov=ai_inference_gateway.response_format",
                "--cov=ai_inference_gateway.mcp_cache",
                "--cov=ai_inference_gateway.pii_redactor",
                "--cov=ai_inference_gateway.moderation",
                "--cov-report=term-missing",
            ],
            "Phase 1 Feature Tests",
        )
        return 0 if success else 1

    elif args[0] == "coverage":
        """Generate coverage report"""
        success = run_command(
            pytest_cmd
            + [
                "--cov=ai_inference_gateway",
                "--cov-report=html",
                "--cov-report=term",
                "--html=htmlcov/index.html",
            ],
            "Coverage Report",
        )

        if success:
            print(f"\n📊 Coverage report generated: {project_dir}/htmlcov/index.html")

        return 0 if success else 1

    elif args[0] == "fast":
        """Run fast tests only (exclude slow)"""
        success = run_command(
            pytest_cmd + ["-m", "not slow"], "Fast Tests (Excluding Slow)"
        )
        return 0 if success else 1

    elif args[0] == "watch":
        """Run tests in watch mode (requires pytest-xdist)"""
        print("Running tests in watch mode...")
        print("Press Ctrl+C to stop")

        try:
            subprocess.run(pytest_cmd + ["-f", "--tb=short"], cwd=script_dir)
        except KeyboardInterrupt:
            print("\n⏹️  Watch mode stopped")
            return 0

    else:
        """Run specific test file or pattern"""
        success = run_command(pytest_cmd + args, f"Custom Tests: {' '.join(args)}")
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
