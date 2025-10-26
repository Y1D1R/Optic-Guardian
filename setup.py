# setup

from pathlib import Path
from setuptools import setup, find_packages # type: ignore

# --- Metadata ---
PACKAGE_NAME = "RetinoNet"
VERSION = "0.1.0"
AUTHOR = "Y1D1R"
REPO_URL = "https://github.com/Y1D1R/Optic-Guardian.git"
DESCRIPTION = "Optic-Guardian est un projet de Deep Learning end-to-end dédié à la détection automatisée de la rétinopathie diabétique à partir d'images du fond d'œil."

README = Path("README.md").read_text(encoding="utf-8")

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    url=REPO_URL,
    project_urls={
        "Issues": f"{REPO_URL}/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
)