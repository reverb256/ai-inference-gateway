{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.ai-inference;
  inherit (lib)
    mkEnableOption
    mkOption
    types
    literalExpression
    ;

  gatewayEnv = pkgs.python3.withPackages (ps: [
    ps.fastapi
    ps.uvicorn
    ps.httpx
    ps.prometheus-client
    ps.pyjwt
    ps.cryptography
    ps.python-multipart
    ps.uvloop
    ps.httptools
    ps.qdrant-client
    ps.sentence-transformers
    ps.rank-bm25
    ps.numpy
  ]);
in
{
  imports = [
    ./gateway.nix
    ./router.nix
    ./monitor.nix
    ./health-monitor.nix
    ./auth
    ./qdrant.nix
  ];

  options.services.ai-inference = {
    enable = mkEnableOption "AI Inference Service (integrates with LM Studio)";

    package = mkOption {
      type = types.package;
      default = gatewayEnv;
      defaultText = literalExpression "pkgs.python3.withPackages (ps: [ps.fastapi ps.uvicorn ...])";
      description = "Python environment with gateway dependencies";
      readOnly = true;
    };

    backend = {
      url = mkOption {
        type = types.str;
        default = "http://127.0.0.1:1234";
        description = "Backend API URL (LM Studio, vLLM, Ollama, or ZAI)";
      };

      type = mkOption {
        type = types.enum [
          "vllm"
          "llama-cpp"
          "sglang"
          "zai"
          "pollinations"
        ];
        default = "llama-cpp";
        description = "Backend inference engine type";
      };

      zai = {
        enable = mkOption {
          type = types.bool;
          default = false;
          description = "Enable ZAI coding plan endpoint";
        };

        apiKey = mkOption {
          type = types.str;
          default = "";
          description = "ZAI API key for coding plan";
        };

        apiKeyFile = mkOption {
          type = types.nullOr types.path;
          default = null;
          example = literalExpression "/run/agenix/zai-api-key";
          description = "Path to file containing ZAI API key (takes precedence over apiKey)";
        };

        baseUrl = mkOption {
          type = types.str;
          default = "https://api.z.ai/api/coding/paas/v4";
          description = "ZAI API base URL (matches OpenCode configuration)";
        };

        maxRetries = mkOption {
          type = types.int;
          default = 3;
          description = "Maximum retry attempts for ZAI requests";
        };

        retryDelay = mkOption {
          type = types.float;
          default = 1.0;
          description = "Initial retry delay in seconds (exponential backoff)";
        };

        timeout = mkOption {
          type = types.float;
          default = 300.0;
          description = "Request timeout in seconds";
        };

        enableRetry = mkOption {
          type = types.bool;
          default = true;
          description = "Enable automatic retry with exponential backoff for ZAI requests";
        };

        models = mkOption {
          type = types.attrs;
          default = {
            "glm-5.1" = {
              name = "GLM-5.1 (200K)";
            };
            "glm-5" = {
              name = "GLM-5 (200K)";
            };
            "glm-5-turbo" = {
              name = "GLM-5 Turbo (200K, Agentic)";
            };
            "glm-4.7" = {
              name = "GLM-4.7 (200K)";
            };
            "glm-4.7-flash" = {
              name = "GLM-4.7 Flash (128K, Vision)";
            };
            "glm-4.6" = {
              name = "GLM-4.6 (256K)";
            };
            "glm-4.5-air" = {
              name = "GLM-4.5 Air (128K)";
            };
          };
          description = "Available ZAI models";
        };
      };

      nvidia-nim = {
        enable = mkOption {
          type = types.bool;
          default = false;
          description = "Enable NVIDIA NIM inference endpoints";
        };

        baseUrl = mkOption {
          type = types.str;
          default = "https://integrate.api.nvidia.com/v1";
          description = "NVIDIA NIM API base URL";
        };

        apiKeyFile = mkOption {
          type = types.nullOr types.path;
          default = null;
          description = "Path to NVIDIA NIM API key file";
        };

        models = mkOption {
          type = types.attrs;
          default = {
            "nvidia/nemotron-3-super-120b-a12b" = {
              name = "Nemotron 3 Super 120B (1M ctx)";
            };
            "deepseek-ai/deepseek-v3.1" = {
              name = "DeepSeek V3.1 (131K ctx)";
            };
            "moonshotai/kimi-k2.5" = {
              name = "Kimi K2.5 1T (262K ctx)";
            };
            "minimaxai/minimax-m2.5" = {
              name = "MiniMax M2.5 230B (1M ctx)";
            };
            "z-ai/glm5" = {
              name = "GLM-5 744B (205K ctx)";
            };
            "openai/gpt-oss-120b" = {
              name = "GPT-OSS 120B (131K ctx)";
            };
            "qwen/qwen3-coder-480b-a35b-instruct" = {
              name = "Qwen3 Coder 480B (1M ctx)";
            };
          };
          description = "Available NVIDIA NIM models";
        };
      };

      local = {
        url = mkOption {
          type = types.str;
          default = "http://127.0.0.1:1235";
          description = "Local llama-cpp server URL";
        };
        model = mkOption {
          type = types.str;
          default = "gemma-4-e2b-it";
          description = "Default model on local llama-cpp server";
        };
      };

      pollinations = {
        enable = mkOption {
          type = types.bool;
          default = false;
          description = "Enable Pollinations AI service (free text, image, TTS)";
        };

        apiKey = mkOption {
          type = types.str;
          default = "";
          description = "Pollinations API key";
        };

        apiKeyFile = mkOption {
          type = types.nullOr types.path;
          default = null;
          example = literalExpression "/run/agenix/pollinations-api-key";
          description = "Path to file containing Pollinations API key (takes precedence over apiKey)";
        };

        baseUrl = mkOption {
          type = types.str;
          default = "https://text.pollinations.ai";
          description = "Pollinations API base URL";
        };

        models = mkOption {
          type = types.attrs;
          default = {
            "openai" = {
              name = "OpenAI-compatible (GPT-4, GPT-4.1, GPT-4o, o1)";
            };
            "anthropic" = {
              name = "Anthropic-compatible (Claude Sonnet, Opus, Haiku)";
            };
            "qwen" = {
              name = "Qwen2.5 72B, 7B";
            };
            "flux" = {
              name = "Flux image generation";
            };
            "turbo" = {
              name = "Fast generation";
            };
          };
          description = "Available Pollinations models";
        };
      };
    };

    searxngUrl = mkOption {
      type = types.str;
      default = "http://10.0.0.102:8080";
      description = "SearXNG URL for knowledge fabric integration";
    };

    gateway = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable API gateway (routing, auth, metrics)";
      };

      host = mkOption {
        type = types.str;
        default = "127.0.0.1";
        description = "Gateway listen address";
      };

      port = mkOption {
        type = types.port;
        default = 8080;
        description = "Gateway listen port";
      };

      workers = mkOption {
        type = types.int;
        default = 4;
        description = "Number of uvicorn workers";
      };

      python = mkOption {
        type = types.nullOr types.package;
        default = null;
        description = "Gateway Python environment (set by gateway.nix)";
        internal = true;
      };

      middleware = {
        redis = {
          enable = mkOption {
            type = types.bool;
            default = false;
            description = "Enable Redis for middleware features";
          };
          host = mkOption {
            type = types.str;
            default = "127.0.0.1";
            description = "Redis host";
          };
          port = mkOption {
            type = types.int;
            default = 6379;
            description = "Redis port";
          };
        };
        knowledgeFabric = {
          enable = mkOption {
            type = types.bool;
            default = false;
            description = "Enable Knowledge Fabric middleware for synthesized search";
          };
          rrf_k = mkOption {
            type = types.int;
            default = 60;
            description = "RRF constant for Reciprocal Rank Fusion";
          };
          rag_enabled = mkOption {
            type = types.bool;
            default = false;
            description = "Enable RAG knowledge source";
          };
          code_search_enabled = mkOption {
            type = types.bool;
            default = true;
            description = "Enable code search source";
          };
          searxng_enabled = mkOption {
            type = types.bool;
            default = false;
            description = "Enable SearXNG knowledge source";
          };
          web_search_enabled = mkOption {
            type = types.bool;
            default = false;
            description = "Enable MCP web search source";
          };
          code_search_paths = mkOption {
            type = types.listOf types.str;
            default = [ "/etc/nixos" ];
            description = "Paths to search for code";
          };
          rag_top_k = mkOption {
            type = types.int;
            default = 5;
            description = "RAG top-K results";
          };
          searxng_url = mkOption {
            type = types.str;
            default = "http://searxng.search.svc.cluster.local:8080";
            description = "SearXNG URL";
          };
          searxng_max_results = mkOption {
            type = types.int;
            default = 5;
            description = "SearXNG max results";
          };
          code_max_results = mkOption {
            type = types.int;
            default = 5;
            description = "Code search max results";
          };
          web_max_results = mkOption {
            type = types.int;
            default = 5;
            description = "Web search max results";
          };
          brain_wiki_enabled = mkOption {
            type = types.bool;
            default = true;
            description = "Enable brain wiki knowledge source";
          };
          brain_wiki_path = mkOption {
            type = types.str;
            default = "/home/j_kro/brain/wiki";
            description = "Path to brain wiki directory";
          };
          brain_wiki_max_results = mkOption {
            type = types.int;
            default = 5;
            description = "Brain wiki max results";
          };
          brain_wiki_max_chunk_chars = mkOption {
            type = types.int;
            default = 2000;
            description = "Max chars per brain wiki chunk";
          };
        };
      };
    };

    routing = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable intelligent model routing by context size";
      };

      fallbackChain = mkOption {
        type = types.listOf types.str;
        default = [
          "vllm"
          "zai"
          "pollinations"
        ];
        description = "Order of backend fallback on failure";
      };

      rules = mkOption {
        type = types.listOf (
          types.submodule {
            options = {
              minTokens = mkOption {
                type = types.int;
                description = "Minimum token count for this rule";
              };
              maxTokens = mkOption {
                type = types.int;
                default = 999999;
                description = "Maximum token count for this rule";
              };
              model = mkOption {
                type = types.str;
                description = "Model to use for this range";
              };
              priority = mkOption {
                type = types.int;
                default = 0;
                description = "Priority (higher = preferred)";
              };
              contextLength = mkOption {
                type = types.int;
                default = 262144;
                description = "Context window size in tokens";
              };
            };
          }
        );
        default = [
          {
            minTokens = 0;
            maxTokens = 131072;
            model = "qwen3.5-35b-a3b";
            priority = 10;
            contextLength = 262144;
          }
          {
            minTokens = 131073;
            maxTokens = 999999;
            model = "qwen3.5-27b";
            priority = 20;
            contextLength = 262144;
          }
        ];
        description = "Model routing rules by token count";
      };

      defaultModel = mkOption {
        type = types.str;
        default = "qwen3.5-35b-a3b";
        description = "Default model when routing is disabled or no rule matches";
      };
    };

    auth = {
      mode = mkOption {
        type = types.enum [
          "none"
          "tailscale"
          "api-key"
          "web3"
        ];
        default = "none";
        description = "Authentication mode";
      };

      tailscale = {
        aclTags = mkOption {
          type = types.listOf types.str;
          default = [ ];
          example = [
            "tag:ai-inference"
            "tag:trusted"
          ];
          description = "Allowed Tailscale ACL tags";
        };
      };

      apiKeyFile = mkOption {
        type = types.nullOr types.path;
        default = null;
        example = literalExpression "/etc/ai-inference/api-keys.txt";
        description = "Path to API keys file";
      };
    };

    monitoring = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Enable Prometheus metrics export";
      };

      port = mkOption {
        type = types.port;
        default = 9190;
        description = "Metrics endpoint port (separate from gateway)";
      };
    };

    rateLimit = {
      enable = mkOption {
        type = types.bool;
        default = false;
        description = "Enable rate limiting";
      };

      requestsPerMinute = mkOption {
        type = types.int;
        default = 60;
        description = "Requests per minute per client";
      };
    };

    systemPrompts = {
      enable = mkEnableOption "Custom system prompts for different request types";

      default = mkOption {
        type = types.str;
        default = "";
        description = "Default system prompt applied to all requests";
      };

      coding = mkOption {
        type = types.str;
        default = "You are an expert coding assistant. Write clean, efficient, and well-documented code.";
        description = "System prompt for coding-related requests";
      };

      reasoning = mkOption {
        type = types.str;
        default = "You are an expert reasoning assistant. Think step-by-step and provide clear explanations.";
        description = "System prompt for reasoning-related requests";
      };

      analysis = mkOption {
        type = types.str;
        default = "You are an expert analysis assistant. Provide thorough and structured analysis.";
        description = "System prompt for analysis-related requests";
      };

      agentic = mkOption {
        type = types.str;
        default = "You are an autonomous agent capable of multi-step planning and execution.";
        description = "System prompt for agentic/workflow requests";
      };

      fast = mkOption {
        type = types.str;
        default = "You are a fast and efficient assistant. Provide concise, direct answers.";
        description = "System prompt for fast response requests";
      };

      custom = mkOption {
        type = types.attrsOf types.str;
        default = { };
        example = literalExpression ''
          {
            nixos = "You are a NixOS configuration expert.";
            kubernetes = "You are a Kubernetes expert.";
          }
        '';
        description = "Custom system prompts by name";
      };
    };

    mcp = {
      enable = mkEnableOption "MCP broker for aggregating tools from multiple MCP servers";

      servers = mkOption {
        type = types.attrsOf (
          types.submodule (
            { config, ... }:
            {
              options = {
                type = mkOption {
                  type = types.enum [
                    "local"
                    "remote"
                  ];
                  default = "remote";
                  description = "MCP server type";
                };

                url = mkOption {
                  type = types.nullOr types.str;
                  default = null;
                  description = "MCP server URL (required for remote type)";
                };

                command = mkOption {
                  type = types.nullOr (types.listOf types.str);
                  default = null;
                  description = "Command to run for local MCP servers";
                };

                environment = mkOption {
                  type = types.attrsOf types.str;
                  default = { };
                  description = "Environment variables for local MCP servers";
                };

                headers = mkOption {
                  type = types.attrsOf types.str;
                  default = { };
                  description = "HTTP headers for authentication (remote type only)";
                };

                enabled = mkOption {
                  type = types.bool;
                  default = true;
                  description = "Whether this server is enabled";
                };
              };
            }
          )
        );
        default = {
          searxng = {
            type = "local";
            command = [
              "${pkgs.python3}/bin/python3"
              "-m"
              "ai_inference_gateway.mcp_servers.searxng_server"
            ];
            environment = {
              SEARXNG_URL = "https://search.reverb256.ca";
              SEARXNG_CACHE_TTL = "300";
            };
          };
        };
        description = "MCP servers to connect to";
      };
    };

    security = {
      maxRequestSize = mkOption {
        type = types.int;
        default = 10485760;
        description = "Maximum request size in bytes";
      };

      enableProxy = mkOption {
        type = types.bool;
        default = false;
        description = "Enable security proxy";
      };
    };

    rag = {
      enable = mkEnableOption "RAG (Retrieval Augmented Generation) with hybrid search";

      qdrant = {
        enable = mkEnableOption "Qdrant vector database service for RAG";

        package = mkOption {
          type = types.package;
          default = pkgs.qdrant;
          description = "Qdrant package to use";
        };

        host = mkOption {
          type = types.str;
          default = "127.0.0.1";
          description = "Qdrant listen address";
        };

        port = mkOption {
          type = types.port;
          default = 6333;
          description = "Qdrant HTTP port";
        };

        grpcPort = mkOption {
          type = types.port;
          default = 6334;
          description = "Qdrant gRPC port";
        };

        storagePath = mkOption {
          type = types.str;
          default = "/var/lib/qdrant";
          description = "Qdrant storage path";
        };

        memoryLimit = mkOption {
          type = types.str;
          default = "2G";
          description = "Memory limit for Qdrant service";
        };
      };

      qdrantUrl = mkOption {
        type = types.str;
        default = "http://127.0.0.1:6333";
        description = "Qdrant vector database URL";
      };

      embeddingModel = mkOption {
        type = types.str;
        default = "BAAI/bge-m3";
        description = "Embedding model for document chunking (BGE-M3: 1024d, multilingual, 8192 ctx)";
      };

      embeddingDevice = mkOption {
        type = types.enum [ "cpu" "cuda" ];
        default = "cpu";
        description = "Device for embedding inference (cpu recommended when GPU is mining)";
      };

      chunkSize = mkOption {
        type = types.int;
        default = 512;
        description = "Chunk size for document splitting (characters)";
      };

      chunkOverlap = mkOption {
        type = types.int;
        default = 50;
        description = "Overlap between chunks";
      };

      topK = mkOption {
        type = types.int;
        default = 5;
        description = "Number of documents to retrieve";
      };

      hybridSearch = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = "Enable hybrid search (vector + BM25)";
        };

        vectorWeight = mkOption {
          type = types.float;
          default = 0.7;
          description = "Weight for vector search (0-1)";
        };

        bm25Weight = mkOption {
          type = types.float;
          default = 0.3;
          description = "Weight for BM25 keyword search (0-1)";
        };
      };

      autoRag = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = "Automatically detect when to use RAG";
        };

        threshold = mkOption {
          type = types.float;
          default = 0.3;
          description = "Confidence threshold below which RAG is triggered";
        };

        keywords = mkOption {
          type = types.listOf types.str;
          default = [
            "what"
            "how"
            "explain"
            "describe"
            "tell me about"
            "find"
            "search"
            "lookup"
            "who"
            "when"
            "where"
            "why"
          ];
          description = "Keywords that trigger RAG retrieval";
        };
      };

      tokenScopedCollections = mkOption {
        type = types.bool;
        default = true;
        description = "Scope knowledge bases by API token (multi-tenancy)";
      };

      reranker = {
        enable = mkOption {
          type = types.bool;
          default = true;
          description = "Enable cross-encoder reranking for better RAG precision";
        };

        model = mkOption {
          type = types.str;
          default = "BAAI/bge-reranker-v2-m3";
          description = "Reranker model name";
        };
      };
    };

    sentry = {
      enable = mkOption {
        type = types.bool;
        default = false;
        description = "Enable Sentry error tracking";
      };

      dsn = mkOption {
        type = types.str;
        default = "";
        description = "Sentry DSN (Data Source Name)";
      };

      dsnFile = mkOption {
        type = types.nullOr types.path;
        default = null;
        example = literalExpression "/run/agenix/sentry-dsn";
        description = "Path to file containing Sentry DSN";
      };

      environment = mkOption {
        type = types.enum [
          "development"
          "staging"
          "production"
        ];
        default = "production";
        description = "Sentry environment name";
      };

      tracesSampleRate = mkOption {
        type = types.float;
        default = 0.1;
        description = "Sample rate for performance tracing (0.0 to 1.0)";
      };
    };
  };
}
