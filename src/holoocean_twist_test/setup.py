import os
from glob import glob
from setuptools import find_packages, setup

package_name = "holoocean_twist_test"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fbartelt",
    maintainer_email="fbartelt@ufmg.br",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "twist_publisher = holoocean_twist_test.twist_publisher:main",
            "twist_to_agent = holoocean_twist_test.twist_to_agent:main",
            "path_controller = holoocean_twist_test.path_controller:main",
            "path_visualizer  = holoocean_twist_test.path_visualizer:main",
        ],
    },
)
