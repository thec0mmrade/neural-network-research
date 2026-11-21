# math-verify and its parser dependency, which nixpkgs does not ship.
# Both are pure-Python wheels from PyPI, pinned by sha256.
{ python3Packages }:
let
  latex2sympy2-extended = python3Packages.buildPythonPackage rec {
    pname = "latex2sympy2_extended";
    version = "1.11.0";
    format = "wheel";
    src = python3Packages.fetchPypi {
      inherit pname version format;
      dist = "py3";
      python = "py3";
      sha256 = "aebb77d52ce269e25028e4bea89ddb14d242ba36bcf7b636496fb5fd9728d234";
    };
    dependencies = with python3Packages; [ sympy antlr4-python3-runtime ];
    pythonImportsCheck = [ "latex2sympy2_extended" ];
  };
in
python3Packages.buildPythonPackage rec {
  pname = "math_verify";
  version = "0.9.0";
  format = "wheel";
  src = python3Packages.fetchPypi {
    inherit pname version format;
    dist = "py3";
    python = "py3";
    sha256 = "3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194";
  };
  dependencies = [ latex2sympy2-extended ];
  pythonImportsCheck = [ "math_verify" ];
}
