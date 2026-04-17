{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.ai-inference.rag;
  inherit (lib) mkIf optional;

  qdrantConfig = pkgs.writeText "qdrant-config.yaml" ''
    service:
      host: ${cfg.qdrant.host}
      http_port: ${toString cfg.qdrant.port}
      grpc_port: ${toString cfg.qdrant.grpcPort}

    storage:
      storage_path: ${cfg.qdrant.storagePath}

    telemetry:
      disable: true
  '';
in
{
  config = mkIf (cfg.enable && cfg.qdrant.enable) {
    systemd.services.qdrant = {
      description = "Qdrant Vector Database";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        ExecStart = "${cfg.qdrant.package}/bin/qdrant --config-path ${qdrantConfig}";
        Restart = "on-failure";
        RestartSec = "10s";
        User = "qdrant";
        Group = "qdrant";
        WorkingDirectory = cfg.qdrant.storagePath;
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [
          cfg.qdrant.storagePath
          "${cfg.qdrant.storagePath}/snapshots"
          "/tmp"
        ];
        MemoryMax = cfg.qdrant.memoryLimit;
        StandardOutput = "journal";
        StandardError = "journal";
        SyslogIdentifier = "qdrant";
      };
    };

    users.users.qdrant = {
      isSystemUser = true;
      group = "qdrant";
      description = "Qdrant Vector Database";
    };
    users.groups.qdrant = { };

    systemd.tmpfiles.rules = [
      "d ${cfg.qdrant.storagePath} 0750 qdrant qdrant - -"
    ];

    networking.firewall.allowedTCPPorts = lib.mkOptionDefault (
      lib.optional (cfg.qdrant.host != "127.0.0.1") cfg.qdrant.port
    );
  };
}
