{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  buildInputs = with pkgs.python313Packages; [
    pytest
    pytest-asyncio
    pytest-cov
    pytest-mock
    pydantic
    pydantic-settings
    fastapi
    httpx
    requests
    starlette
    sentence-transformers
    qdrant-client
    openai
    prometheus-client
  ];

  shellHook = ''
    echo "Knowledge Fabric Test Environment"
    echo "==============================="
    echo ""
    echo "Available commands:"
    echo "  pytest                          - Run all tests"
    echo "  pytest tests/                   - Run tests in directory"
    echo "  pytest tests/test_knowledge_fabric.py -v -s  Run specific test file"
    echo "  pytest --cov=ai_inference_gateway.middleware.knowledge_fabric --cov-report=term-missing -v"
    echo ""
  '';
}
