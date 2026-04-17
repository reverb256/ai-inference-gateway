{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.ai-inference.auth;
  inherit (lib) mkIf;
in {
  imports = [
    ./tailscale.nix
  ];

  config = mkIf config.services.ai-inference.enable {

    systemd.services.ai-inference-validate-keys = mkIf (cfg.mode == "api-key" && cfg.apiKeyFile != null) {
      description = "Validate AI inference API keys";
      wantedBy = ["multi-user.target"];

      serviceConfig = {
        Type = "oneshot";
        ExecStart = pkgs.writeShellScript "ai-inference-validate-keys" ''
          if [ ! -f ${cfg.apiKeyFile} ]; then
            echo "API key file not found: ${cfg.apiKeyFile}"
            exit 1
          fi
          echo "API key file found"
        '';
        User = "root";
      };
    };
  };
}
