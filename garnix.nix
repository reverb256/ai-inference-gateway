{
  description = "AI Inference Gateway - Garnix CI";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystemMap (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        # CI checks
        checks = {
          # Flake evaluation
          flake-check = pkgs.runCommand "flake-check" {
            nativeBuildInputs = [ pkgs.nix ];
          } ''
            nix flake check ${self} --no-build
            touch $out
          '';

          # Format check (nixfmt)
          format-check = pkgs.runCommand "format-check" {
            nativeBuildInputs = [ pkgs.nixfmt ];
          } ''
            nixfmt --check ${self}
            touch $out
          '';
        };
      }
    );
}
