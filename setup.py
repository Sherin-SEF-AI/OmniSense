"""
OMNISENSE Multi-Camera Spatial Intelligence Platform
Setup configuration
"""

from setuptools import setup, find_packages
import os

# Read the long description from README
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "OMNISENSE Multi-Camera Spatial Intelligence Platform"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith('#')]

setup(
    name="omnisense",
    version="1.0.0",
    author="OmniSense Development Team",
    author_email="dev@omnisense.ai",
    description="Production-ready multi-camera spatial intelligence platform with real-time perception",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/Sherin-SEF-AI/OmniSense",
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'scripts']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.0.0',
            'mypy>=1.4.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'omnisense=omnisense.main:main',
            'omnisense-calibrate=omnisense.camera.calibration:calibrate_cli',
            'omnisense-download-models=scripts.download_models:main',
        ],
    },
    package_data={
        'omnisense': [
            'configs/*.yaml',
            'configs/**/*.yaml',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
