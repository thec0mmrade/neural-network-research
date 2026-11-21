# Environment for qwen-weight-probe. Enter with `nix-shell`, or run one command with `nix-shell --run '...'`.
# cudaSupport = true gives a CUDA torch AND matches the package hashes that cache.nixos-cuda.org
# builds, so CUDA libraries (NCCL, libnvshmem, torch itself) download instead of compiling locally.
# Without it, those hashes miss the cache and nix compiles them on all cores (this crashed the machine once).
{ pkgs ? import <nixpkgs> { config = { allowUnfree = true; cudaSupport = true; }; } }:
let
  python = pkgs.python3;
  math-verify = import ./nix/math-verify.nix { python3Packages = python.pkgs; };
  pyEnv = python.withPackages (ps: with ps; [
    torch
    transformers
    accelerate
    safetensors
    huggingface-hub
    numpy
    sympy
    math-verify
  ]);
in
pkgs.mkShell {
  packages = [ pyEnv pkgs.git ];
  shellHook = ''
    # NixOS exposes the host NVIDIA driver (libcuda) here.
    export LD_LIBRARY_PATH=/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export HF_HOME=''${HF_HOME:-$HOME/.cache/huggingface}
    export PYTHONPATH=$PWD/src''${PYTHONPATH:+:$PYTHONPATH}
    export NIXPKGS_VERSION=${pkgs.lib.version}
  '';
}
