{ config, lib, ... }:
let
  cfg = config.services.ai-inference;
in
{
  config = lib.mkIf cfg.enable {
    networking.firewall.allowedTCPPorts = lib.mkOptionDefault (
      lib.optional (
        cfg.rag.enable && cfg.rag.qdrant.enable && cfg.rag.qdrant.host != "127.0.0.1"
      ) cfg.rag.qdrant.port
    );
  };
}
