# Lightweight shell for the weight-only phases (1-3): numpy + hub client, no torch/CUDA.
#   nix-shell nix/cpu-shell.nix --run 'python -m probe.survey'
{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [ (pkgs.python3.withPackages (ps: [ ps.numpy ps.huggingface-hub ps.umap-learn ps.scikit-learn ps.rich ])) ];
  shellHook = ''
    export HF_HOME=''${HF_HOME:-$HOME/.cache/huggingface}
    export PYTHONPATH=$PWD/src''${PYTHONPATH:+:$PYTHONPATH}
  '';
}
