{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.ai-inference;
  inherit (lib) mkIf;

  gatewaySrc = ../src;

  # CRITICAL: runCommand with { src = ... } does NOT make src a real nix dependency.
  # Nix caches the derivation even when source files change, serving STALE CODE.
  # Fix: use builtins.path to compute a hash of the source tree, then reference
  # that hash in the derivation so any source change forces a rebuild.
  gatewaySrcHash = builtins.hashString "sha256" (
    builtins.concatStringsSep ""
      (map (f: builtins.hashFile "sha256" "${gatewaySrc}/${f}")
        (builtins.filter (f: lib.hasSuffix ".py" f)
          (builtins.attrNames (builtins.readDir gatewaySrc))))
  );

  modularGatewayPkgBase =
    pkgs.runCommand "ai-inference-gateway-modular-pkg-base-${lib.strings.substring 0 8 gatewaySrcHash}"
      {
        preferLocalBuild = true;
      }
      ''
        mkdir -p $out/ai_inference_gateway
        cp -r ${gatewaySrc}/. $out/ai_inference_gateway/
        chmod -R u+w $out/ai_inference_gateway
        find $out -name "*.pyc" -delete
        find $out -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
      '';

  modularGatewayPkgPython =
    pkgs.runCommand "ai-inference-gateway-modular-pkg-python-${lib.strings.substring 0 8 gatewaySrcHash}"
      {
        preferLocalBuild = true;
      }
      ''
        mkdir -p $out/lib/python3.13/site-packages
        cp -r ${gatewaySrc}/. $out/lib/python3.13/site-packages/ai_inference_gateway
        chmod -R u+w $out/lib/python3.13/site-packages/ai_inference_gateway
        find $out -name "*.pyc" -delete
        find $out -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
      '';

  gatewayPython = pkgs.python3.withPackages (
    ps:
    [
      ps.fastapi
      ps.uvicorn
      ps.httpx
      ps.openai
      ps.anthropic
      ps.prometheus-client
      ps.pyjwt
      ps.cryptography
      ps.python-multipart
      ps.uvloop
      ps.httptools
      ps.aiohttp
      ps.psutil
      ps.qdrant-client
      ps.sentence-transformers
      ps.rank-bm25
      ps.numpy
      ps.beautifulsoup4
      ps.redis
      ps.pydantic
      ps.pydantic-settings
      ps.sentry-sdk
      ps.mcp
      ps.huggingface-hub
      ps.qwen-tts
      ps.transformers
      ps.torch
      ps.torchaudio
      ps.accelerate
      ps.datasets
      ps.pydub
      ps.soundfile
      ps.librosa
      ps.einops
      ps.pillow
      ps.onnxruntime
      ps.scikit-learn
      ps.lxml
      ps.feedgen
    ]
    ++ [ modularGatewayPkgPython ]
  );

  modularGatewayPkg = pkgs.symlinkJoin {
    name = "ai-inference-gateway-modular-pkg-v15";
    paths = [
      modularGatewayPkgBase
      gatewayPython
    ];
  };

  gatewayPkg = modularGatewayPkg;

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
      modularGatewayPkgBase
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
        "TORCHINDUCTOR_CACHE_DIR=/var/cache/ai-inference/torch-cache"
      ];
      WorkingDir = "/app";
    };
  };

  opencodeSearxngMcpWrapper = pkgs.writeShellApplication {
    name = "opencode-searxng-mcp";
    runtimeInputs = with pkgs; [
      coreutils
      findutils
      gnugrep
      gnused
    ];
    text = builtins.readFile ./bin/opencode-searxng-mcp;
  };
in
{
  config = mkIf (cfg.enable && cfg.gateway.enable) {
    services.ai-inference.gateway.python = gatewayPython;

    environment.systemPackages = [ opencodeSearxngMcpWrapper ];

    systemd.services.ai-inference-gateway = {
      description = "AI Inference Gateway";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];
      path = [ pkgs.k3s ];

      environment = {
        KUBECONFIG = "/etc/rancher/k3s/k3s.yaml";
        PYTHONPATH = "${modularGatewayPkgBase}:${gatewayPython}/lib/python3.13/site-packages";
        BACKEND_URL = cfg.backend.url;
        BACKEND_TYPE = cfg.backend.type;
        GATEWAY_HOST = cfg.gateway.host;
        GATEWAY_PORT = toString cfg.gateway.port;
        LOCAL_BACKEND_URL = cfg.backend.local.url;
        LOCAL_BACKEND_MODEL = cfg.backend.local.model;
        # RAG service
        RAG_ENABLED = if cfg.rag.enable then "true" else "false";
        QDRANT_URL = cfg.rag.qdrantUrl;
        EMBEDDING_MODEL = cfg.rag.embeddingModel;
        EMBEDDING_DEVICE = cfg.rag.embeddingDevice;
        RERANKER_ENABLED = if cfg.rag.reranker.enable then "true" else "false";
        RERANKER_MODEL = cfg.rag.reranker.model;
        # Query expansion
        QUERY_EXPANSION_ENABLED = if cfg.queryExpansion.enable then "true" else "false";
        QUERY_EXPANSION_MODEL = cfg.queryExpansion.model;
        # Semantic cache
        SEMANTIC_CACHE_ENABLED = if cfg.semanticCache.enable then "true" else "false";
        SEMANTIC_CACHE_TTL_SECONDS = toString cfg.semanticCache.ttlSeconds;
        SEMANTIC_CACHE_SIMILARITY_THRESHOLD = toString cfg.semanticCache.similarityThreshold;
        HF_HOME = "/var/cache/ai-inference";
        TRANSFORMERS_CACHE = "/var/cache/ai-inference";
      }
      // lib.optionalAttrs cfg.backend.zai.enable {
        ZAI_API_KEY_FILE =
          if cfg.backend.zai.apiKeyFile != null then toString cfg.backend.zai.apiKeyFile else "";
        ZAI_BASE_URL = cfg.backend.zai.baseUrl;
        BACKEND_FALLBACK_URLS = cfg.backend.zai.baseUrl;
      }
      // lib.optionalAttrs cfg.backend.nvidia-nim.enable {
        NVIDIA_NIM_API_KEY_FILE =
          if cfg.backend.nvidia-nim.apiKeyFile != null then
            toString cfg.backend.nvidia-nim.apiKeyFile
          else
            "";
        NVIDIA_NIM_BASE_URL = cfg.backend.nvidia-nim.baseUrl;
      }
      // lib.optionalAttrs cfg.backend.pollinations.enable {
        POLLINATIONS_API_KEY_FILE =
          if cfg.backend.pollinations.apiKeyFile != null then
            toString cfg.backend.pollinations.apiKeyFile
          else
            "";
      }
      // lib.optionalAttrs cfg.gateway.middleware.redis.enable {
        REDIS_URL = "redis://${cfg.gateway.middleware.redis.host}:${toString cfg.gateway.middleware.redis.port}";
      }
      // lib.optionalAttrs cfg.gateway.middleware.knowledgeFabric.enable {
        MIDDLEWARE__KNOWLEDGE_FABRIC__ENABLED = "true";
        MIDDLEWARE__KNOWLEDGE_FABRIC__RRF_K = toString cfg.gateway.middleware.knowledgeFabric.rrf_k;
        MIDDLEWARE__KNOWLEDGE_FABRIC__RAG_ENABLED =
          if cfg.gateway.middleware.knowledgeFabric.rag_enabled then "true" else "false";
        MIDDLEWARE__KNOWLEDGE_FABRIC__SEARXNG_ENABLED =
          if cfg.gateway.middleware.knowledgeFabric.searxng_enabled then "true" else "false";
        MIDDLEWARE__KNOWLEDGE_FABRIC__SEARXNG_URL = cfg.gateway.middleware.knowledgeFabric.searxng_url;
        MIDDLEWARE__KNOWLEDGE_FABRIC__CODE_SEARCH_ENABLED =
          if cfg.gateway.middleware.knowledgeFabric.code_search_enabled then "true" else "false";
        MIDDLEWARE__KNOWLEDGE_FABRIC__CODE_SEARCH_PATHS = builtins.toJSON cfg.gateway.middleware.knowledgeFabric.code_search_paths;
        MIDDLEWARE__KNOWLEDGE_FABRIC__WEB_SEARCH_ENABLED =
          if cfg.gateway.middleware.knowledgeFabric.web_search_enabled then "true" else "false";
        MIDDLEWARE__KNOWLEDGE_FABRIC__WEB_MAX_RESULTS = toString cfg.gateway.middleware.knowledgeFabric.web_max_results;
        MIDDLEWARE__KNOWLEDGE_FABRIC__BRAIN_WIKI_ENABLED =
          if cfg.gateway.middleware.knowledgeFabric.brain_wiki_enabled then "true" else "false";
        MIDDLEWARE__KNOWLEDGE_FABRIC__BRAIN_WIKI_PATH =
          cfg.gateway.middleware.knowledgeFabric.brain_wiki_path;
      };

      serviceConfig = {
        Type = "simple";
        ExecStart =
          let
            args = [
              "--host"
              cfg.gateway.host
              "--port"
              (toString cfg.gateway.port)
              "--workers"
              (toString cfg.gateway.workers)
            ];
          in
          "${gatewayPython}/bin/python -m uvicorn ai_inference_gateway.main:app ${lib.concatStringsSep " " args}";

        ReadOnlyPaths =
          [ "/etc/rancher/k3s/k3s.yaml" ]
          ++ lib.optionals (cfg.backend.zai.apiKeyFile != null) [ cfg.backend.zai.apiKeyFile ]
          ++ lib.optionals (cfg.backend.pollinations.apiKeyFile != null) [
            cfg.backend.pollinations.apiKeyFile
          ];

        RuntimeDirectory = "ai-inference";
        RuntimeDirectoryMode = "0755";
        CacheDirectory = "ai-inference";
        CacheDirectoryMode = "0755";
        LogsDirectory = "ai-inference";

        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = "read-only";
        ReadWritePaths = [
          "/var/cache/ai-inference"
          "/run/ai-inference"
          "/tmp"
        ];

        Restart = "on-failure";
        RestartSec = 5;
        LimitNOFILE = 65536;
        TimeoutStartSec = "300";

        Environment = [ "PATH=${gatewayPython}/bin:/run/current-system/sw/bin:/usr/bin:/bin" ];
      };
    };
  };
}
