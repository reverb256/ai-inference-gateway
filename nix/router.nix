{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.ai-inference;
  inherit (lib) mkIf;


  tokenEstimator = pkgs.symlinkJoin {
    name = "token-estimator";
    paths = [
      (pkgs.writeScriptBin "token-estimator" ''
        #!${pkgs.python3}/bin/python3
        """
        Token estimation utilities
        Approximates token count for routing decisions
        """
        import sys
        import json

        CHARS_PER_TOKEN = 4
        CHARS_PER_TOKEN_CODE = 6

        def estimate_tokens(text: str, is_code: bool = False) -> int:
            if not text:
                return 0
            divisor = CHARS_PER_TOKEN_CODE if is_code else CHARS_PER_TOKEN
            return max(1, len(text) // divisor)

        def recommend_model(token_count: int) -> str:
            """Recommend model based on estimated token count (Qwen3.5 supports 256K)."""
            if token_count <= 16384:
                return "qwen3.5-2b"
            elif token_count <= 65536:
                return "qwen3.5-4b"
            elif token_count <= 131072:
                return "qwen/qwen3.5-9b"
            else:
                return "magnum-opus-35b-a3b-i1"

        if len(sys.argv) > 1:
            text = sys.argv[1]
            print(f"Estimated tokens: {estimate_tokens(text)}")
        else:
            data = json.load(sys.stdin)
            if "messages" in data:
                total = sum(estimate_tokens(msg.get("content", "")) for msg in data["messages"])
                print(json.dumps({
                    "estimated": total,
                    "recommended_model": recommend_model(total)
                }, indent=2))
      '')
    ];
  };
in {
  config = mkIf (cfg.enable && cfg.routing.enable) {
    environment.systemPackages = [tokenEstimator];

  };
}
