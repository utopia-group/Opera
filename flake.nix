{
  description = "Application packaged using poetry2nix";

  nixConfig = {
    extra-substituters = ["https://mistzzt.cachix.org"];
    extra-trusted-public-keys = ["mistzzt.cachix.org-1:Ie2vJ/2OCl4D/ifadJLqqd6X3Uj7J2bDqNmw8n1hAJc="];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-23.11";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    progsyn = {
      url = "github:mistzzt/program-synthesis-nur";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    poetry2nix,
    progsyn,
  }: let
    systems = [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ];

    forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
  in {
    packages = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      synPkgs = progsyn.packages.${system};
      inherit (poetry2nix.lib.mkPoetry2Nix {inherit pkgs;}) mkPoetryApplication;

      opera = mkPoetryApplication {
        projectDir = ./.;
        preferWheels = true;
        buildInputs = [synPkgs.reduce-algebra];
      };
    in {
      opera = opera.overrideAttrs (final: prev: {
        preConfigure = ''
          echo "REDUCE_PATH=${synPkgs.reduce-algebra}/bin/redcsl" > b2s/.env
        '';
      });
      default = self.packages.${system}.opera;
    });

    devShells = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      synPkgs = progsyn.packages.${system};
    in {
      default = pkgs.mkShell {
        inputsFrom = [self.packages.${system}.opera];
        packages =
          (with pkgs; [poetry cmake cvc5])
          ++ (with synPkgs; [sketch]);
      };
    });
  };
}
