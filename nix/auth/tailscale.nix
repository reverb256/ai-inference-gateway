{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.ai-inference;
  authCfg = config.services.ai-inference.auth;
  inherit (lib) mkIf mkDefault;
in
{
  config = mkIf (cfg.enable && authCfg.mode == "tailscale") {
    services = {
      tailscale = {
        enable = true;
        extraUpFlags = [ "--accept-routes" ];
      };

      ai-inference.gateway.host = mkDefault "100.64.0.1";

      nginx = mkIf (builtins.length authCfg.tailscale.aclTags > 0) {
        enable = false;
        virtualHosts."ai-inference-tailscale" = {
          listen = [
            {
              addr = "100.64.0.1";
              port = 8080;
            }
          ];
          locations."/" = {
            proxyPass = "http://127.0.0.1:${toString cfg.gateway.port}";
            extraConfig = ''
              allow 100.64.0.0/10;
              allow fd7a:115c:a1e0::/48;
              allow 127.0.0.1;
              deny all;

              proxy_set_header X-Real-IP $remote_addr;
              proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
              proxy_set_header Host $host;

              proxy_http_version 1.1;
              proxy_set_header Upgrade $http_upgrade;
              proxy_set_header Connection "upgrade";
            '';
          };
        };
      };
    };

    networking.firewall = {
      allowedTCPPorts = lib.mkOptionDefault [ 41641 ];
      trustedInterfaces = [ "tailscale0" ];
    };

    systemd.services.ai-inference-tailscale-ip = {
      description = "Detect and configure Tailscale IP for AI inference";
      after = [
        "tailscale.service"
        "network-online.target"
      ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "oneshot";
        ExecStart = pkgs.writeScript "detect-tailscale-ip" ''
          #!/bin/bash
          IP=$(ip -4 addr show tailscale0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
          if [ -n "$IP" ]; then
            echo "Tailscale IP detected: $IP"
            echo "Update your configuration with:"
            echo "  services.ai-inference.gateway.host = \"$IP\";"
            if [ "${cfg.gateway.host}" = "100.64.0.1" ]; then
              echo "WARNING: Using default Tailscale IP. Update your NixOS config."
            fi
          else
            echo "No Tailscale IP found. Ensure Tailscale is running."
          fi
        '';
      };
    };
  };
}
