#!/bin/bash
# LOCKED BY Worker C
# Developer hooks script for linting and testing.
# 
# This script runs the following in sequence:
# 1. ruff format . - Formats all Python files
# 2. ruff check . - Lints the code and fixes issues
# 3. pytest - Runs the test suite
#
# Usage: ./run_tests.sh

set -e  # Exit on any error

echo "=== Running Developer Hooks ==="

echo "1. Formatting code with ruff format..."
ruff format .

if [ $? -eq 0 ]; then
    echo "✓ Code formatting successful"
else
    echo "✗ Code formatting failed"
    exit 1
fi

echo "2. Linting code with ruff check..."
ruff check . --fix

if [ $? -eq 0 ]; then
    echo "✓ Code linting successful"
else
    echo "✗ Code linting failed"
    exit 1
fi

echo "3. Running tests with pytest..."
pytest tests/ -v

if [ $? -eq 0 ]; then
    echo "✓ Tests passed"
else
    echo "✗ Tests failed"
    exit 1
fi

echo "=== All developer hooks completed successfully ==="
