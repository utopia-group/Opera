FROM nixos/nix:2.18.1

RUN echo "experimental-features = flakes nix-command" >> /etc/nix/nix.conf
RUN nix run nixpkgs#gnused -- -i 's/sandbox = false/sandbox = true/' /etc/nix/nix.conf

WORKDIR /workspace
ENTRYPOINT ["nix", "develop"]
