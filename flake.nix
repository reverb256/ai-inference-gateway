{
  description = "AI Inference Gateway - OpenAI-compatible API gateway with intelligent routing, circuit breaker, RAG, and MCP";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      python3 = pkgs.python313;

      gatewayDeps = with python3.pkgs; [
        fastapi
        uvicorn
        httpx
        openai
        pydantic
        pydantic-settings
        prometheus-client
        python-multipart
        qdrant-client
        sentence-transformers
        rank-bm25
        numpy
        beautifulsoup4
        redis
        mcp
        aiohttp
        psutil
        pyjwt
        cryptography
        pillow
        scikit-learn
        lxml
        einops
        sentry-sdk
        anthropic
        huggingface-hub
        transformers
        torch
        torchaudio
        accelerate
        soundfile
        librosa
        pydub
        uvloop
        httptools
        feedgen
        datasets
      ];

      # Install the gateway source as a Python package
      gatewaySrcPkg = python3.pkgs.buildPythonPackage {
        pname = "ai-inference-gateway";
        version = "2.0.0";
        src = ./src;
        pyproject = false;
        propagatedBuildInputs = gatewayDeps;
        # No tests in the package build (run separately via devShell)
        doCheck = false;
      };

      # Full Python environment with gateway + all deps
      gatewayPython = python3.withPackages (ps: gatewayDeps ++ [ gatewaySrcPkg ]);

      # Container image
      gatewayContainerImage = pkgs.dockerTools.buildLayeredImage {
        name = "ai-inference-gateway";
        tag = "latest";
        extraCommands = ''
          mkdir -p home/ai-gateway
          chown 1000:1000 home/ai-gateway
          mkdir -p var/cache/ai-inference
          chown 1000:1000 var/cache/ai-inference
          mkdir -p tmp
          chown 1000:1000 tmp
          mkdir -p run/ai-inference
          chown 1000:1000 run/ai-inference
        '';
        contents = [
          gatewayPython
          pkgs.bash
          pkgs.coreutils
          pkgs.cacert
        ];
        config = {
          User = "1000:1000";
          Cmd = [
            "${gatewayPython}/bin/python"
            "-m"
            "uvicorn"
            "ai_inference_gateway.main:app"
            "--host"
            "0.0.0.0"
            "--port"
            "8080"
            "--workers"
            "4"
          ];
          ExposedPorts = {
            "8080/tcp" = { };
          };
          Env = [
            "PYTHONPATH=/app:${gatewayPython}/lib/python3.13/site-packages"
            "PATH=${gatewayPython}/bin:/usr/bin:/bin"
            "HOME=/home/ai-gateway"
            "USER=ai-gateway"
            "TRANSFORMERS_CACHE=/var/cache/ai-inference"
            "HF_HOME=/var/cache/ai-inference"
          ];
          WorkingDir = "/app";
        };
      };
    in
    {
      packages.${system} = {
        default = gatewaySrcPkg;
        ai-inference-gateway = gatewaySrcPkg;
        container = gatewayContainerImage;
      };

      nixosModules.default = import ./nix;

      kubernetesModules.default = import ./kubernetes/module.nix;

      devShells.${system}.default = pkgs.mkShell {
        buildInputs =
          with python3.pkgs;
          [
            # Runtime deps
            fastapi
            uvicorn
            httpx
            openai
            pydantic
            pydantic-settings
            prometheus-client
            python-multipart
            qdrant-client
            sentence-transformers
            rank-bm25
            numpy
            beautifulsoup4
            redis
            mcp
            aiohttp
            psutil
            pyjwt
            cryptography
            anthropic
            sentry-sdk
            uvloop
            httptools

            # Dev tools
            pytest
            pytest-asyncio
            pytest-cov
            pytest-mock
            ruff
          ]
          ++ [
            pkgs.qdrant
            pkgs.redis
          ];

        shellHook = ''
          export PYTHONPATH=$(pwd)/src:$PYTHONPATH
          echo "AI Inference Gateway dev shell"
          echo "=============================="
          echo ""
          echo "Commands:"
          echo "  pytest                    - Run all tests"
          echo "  pytest tests/test_main.py - Run specific test"
          echo "  ruff check src/           - Lint source"
          echo ""
          echo "Environment:"
          echo "  PYTHONPATH=$(pwd)/src"
        '';
      };

      # For hydra jobs / CI
      checks.${system} = {
        build = gatewaySrcPkg;
      };
    };
}
