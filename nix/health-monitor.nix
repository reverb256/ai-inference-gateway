{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.ai-inference.health-monitor;
  inherit (lib) mkEnableOption mkIf types;
in {
  options.services.ai-inference.health-monitor = {
    enable = mkEnableOption "AI Inference Gateway health monitoring for OpenCode";

    interval = lib.mkOption {
      type = types.str;
      default = "5min";
      description = "Systemd timer interval for health checks";
    };
  };

  config = mkIf (config.services.ai-inference.enable && cfg.enable) {
    environment.systemPackages = with pkgs; [
      (pkgs.writeShellScriptBin "opencode-gateway-health" ''
        #!${pkgs.bash}/bin/bash
        exec ${pkgs.bash}/bin/bash ${./scripts/ensure-opencode-gateway.sh} "$@"
      '')
    ];

    systemd = {
      services.opencode-gateway-health = {
        description = "OpenCode Gateway Health Monitor";
        serviceConfig = {
          Type = "oneshot";
          ExecStart = lib.getExe pkgs.bash + " ${./scripts/ensure-opencode-gateway.sh} repair";
          User = "root";
          Group = "root";
        };
      };

      timers.opencode-gateway-health = {
        description = "OpenCode Gateway Health Check Timer";
        wantedBy = ["timers.target"];
        timerConfig = {
          OnBootSec = "1min";
          OnUnitActiveSec = cfg.interval;
          AccuracySec = "1s";
        };
      };

      tmpfiles.settings."ai-inference-health" = {
        "/var/log/ai-inference" = {
          d = {
            group = "ai-inference";
            mode = "0755";
            user = "ai-inference";
          };
        };
      };
    };

    system.activationScripts.updateOpenCodeConfig =
      lib.stringAfter ["users"]
      ''
        if [ -f /home/j_kro/.config/opencode/opencode.json ]; then
          echo "OpenCode configuration exists at /home/j_kro/.config/opencode/opencode.json"
          echo "Run 'opencode-gateway-health check' to verify gateway status"
        fi
      '';
  };
}
