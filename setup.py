from skbuild import setup


setup(
    cmake_source_dir=".",
    name="rmsh",
    version="0.0.1",
    cmake_languages=["C"],
    packages=["rmsh"],
)
