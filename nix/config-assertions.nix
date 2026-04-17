{ config, lib, pkgs, ... }:
let
  cfg = config.services.ai-inference;
in
{
  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.backend.url != "";
        message = ''
          AI Inference Service requires a backend URL to be configured.

          Current configuration:
            services.ai-inference.backend.url = "${cfg.backend.url}"

          Configure a backend URL in one of these ways:
            services.ai-inference.backend.url = "http://127.0.0.1:1234";
            services.ai-inference.backend.url = "http://127.0.0.1:8080";
        '';
      }
      {
        assertion =
          cfg.backend.zai.enable -> (cfg.backend.zai.apiKey != "" || cfg.backend.zai.apiKeyFile != null);
        message = ''
          ZAI backend is enabled but no API key is configured.

          When services.ai-inference.backend.zai.enable is true, you must configure:
            services.ai-inference.backend.zai.apiKey = "your-api-key";
            services.ai-inference.backend.zai.apiKeyFile = /run/agenix/zai-api-key;

          Current configuration:
            zai.enable = ${toString cfg.backend.zai.enable}
            zai.apiKey = ${if cfg.backend.zai.apiKey != "" then "***" else "(not set)"}
            zai.apiKeyFile = ${
              if cfg.backend.zai.apiKeyFile != null then toString cfg.backend.zai.apiKeyFile else "(not set)"
            }
        '';
      }
      {
        assertion =
          cfg.backend.type == "zai" -> (cfg.backend.zai.apiKey != "" || cfg.backend.zai.apiKeyFile != null);
        message = ''
          Backend type is "zai" but no ZAI API key is configured.

          When using ZAI backend, configure an API key:
            services.ai-inference.backend.zai.apiKey = "your-zai-api-key";
            services.ai-inference.backend.zai.apiKeyFile = /run/agenix/zai-api-key;

          Current configuration:
            backend.type = "${cfg.backend.type}"
            zai.apiKey = ${if cfg.backend.zai.apiKey != "" then "***" else "(not set)"}
            zai.apiKeyFile = ${
              if cfg.backend.zai.apiKeyFile != null then toString cfg.backend.zai.apiKeyFile else "(not set)"
            }

          Or change backend type to: vllm, llama-cpp, sglang, pollinations
        '';
      }
      {
        assertion = cfg.rag.enable -> cfg.rag.qdrant.enable;
        message = ''
          RAG is enabled but Qdrant vector database is not enabled.

          When services.ai-inference.rag.enable is true, you must also enable Qdrant:
            services.ai-inference.rag.qdrant.enable = true;

          Current configuration:
            rag.enable = ${toString cfg.rag.enable}
            rag.qdrant.enable = ${toString cfg.rag.qdrant.enable}
        '';
      }
      {
        assertion = cfg.mcp.enable -> (builtins.length (lib.attrValues cfg.mcp.servers)) > 0;
        message = ''
          MCP broker is enabled but no MCP servers are configured.

          Add MCP servers to:
            services.ai-inference.mcp.servers.<name> = { ... };

          Example:
            services.ai-inference.mcp.servers.searxng = {
              type = "local";
              command = [ "${pkgs.python3}/bin/python3" "-m" "searxng_server" ];
            };

          Current configuration:
            mcp.enable = ${toString cfg.mcp.enable}
            mcp.servers (count) = ${toString (builtins.length (lib.attrValues cfg.mcp.servers))}
        '';
      }
      {
        assertion = cfg.security.maxRequestSize > 0;
        message = ''
          Invalid security.maxRequestSize: must be greater than 0.

          Current value: ${toString cfg.security.maxRequestSize}

          Recommended minimum: 1048576 (1MB)
          Current default: 10485760 (10MB)
        '';
      }
    ];
  };
}
