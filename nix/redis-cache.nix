{
  config,
  lib,
  ...
}:
with lib; let
  cfg = config.services.redis.servers.searxng;
in {
  options.services.redis.servers.searxng = {
    enable = mkEnableOption "Redis server for SearXNG caching";

    bind = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = "IP address to bind Redis server";
    };

    port = mkOption {
      type = types.port;
      default = 6379;
      description = "Port to listen on";
    };

    maxmemory = mkOption {
      type = types.str;
      default = "256mb";
      description = "Maximum memory to use for caching";
    };

    maxmemoryPolicy = mkOption {
      type = types.str;
      default = "allkeys-lru";
      description = "Eviction policy when maxmemory is reached";
    };
  };

  config = mkIf cfg.enable {
    services.redis.servers.searxng = {
      inherit (cfg) enable bind port maxmemory maxmemoryPolicy;

      databases = 1;
      save = [];
      appendonly = false;

      requirePass = false;
      unixSocket = "/run/redis-searxng/redis.sock";

      logLevel = "warning";

      settings = {
        maxmemory-policy = cfg.maxmemoryPolicy;

        tcp-backlog = 511;
        timeout = 0;
        tcp-keepalive = 300;

        hash-max-ziplist-entries = 512;
        hash-max-ziplist-value = 64;

        save = "";

        maxclients = 10000;

        slowlog-log-slower-than = 10000;
        slowlog-max-len = 128;

        latency-monitor-threshold = 100;
      };
    };

    systemd.tmpfiles.rules = [
      "d /run/redis-searxng 0750 redis redis -"
    ];

  };
}
