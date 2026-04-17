{
  config,
  lib,
  pkgs,
  inputs,
  ...
}:
let
  cfg = config.services.ai-inference;
in
{
  config = lib.mkIf cfg.enable {
    environment.systemPackages = with pkgs; [
      config.services.ai-inference.package
      inputs.claude-native.packages.x86_64-linux.claude
      ffmpeg
      (pkgs.writeShellScriptBin "ai-inference-status" ''
        #!/bin/bash
        echo "=== AI Inference Service Status ==="
        echo "Backend: ${cfg.backend.type}"
        echo "Backend URL: ${cfg.backend.url}"
        echo ""
        echo "=== Backend Models ==="
        ${pkgs.curl}/bin/curl -s ${cfg.backend.url}/v1/models | ${pkgs.jq}/bin/jq -r '.data[].id' || echo "Backend unavailable"
        echo ""
        echo "=== K8s Gateway Health ==="
        ${pkgs.curl}/bin/curl -s http://ai-inference-gateway.ai-inference.svc.cluster.local:8080/health || echo "K8s gateway unavailable"
      '')
    ];

    services = {
      prometheus.scrapeConfigs = lib.mkIf cfg.monitoring.enable [
        {
          job_name = "ai-inference-gateway";
          static_configs = [
            {
              targets = [ "ai-inference-gateway.ai-inference.svc.cluster.local:${toString cfg.monitoring.port}" ];
              labels = {
                instance = "ai-inference-gateway";
                backend = cfg.backend.type;
              };
            }
          ];
        }
      ];

      redis.servers.ai-gateway = {
        inherit (cfg.gateway.middleware.redis) enable;
        bind = "127.0.0.1";
        port = 6380;
      };
    };
  };
}
