from setuptools import setup, find_packages

setup(
    name="medical-image-restoration",
    version="1.0.0",
    description="Medical Image Restoration and Enhancement System Using Deep Learning",
    author="MedRestore",
    packages=find_packages(),
    package_data={"web": ["static/*"]},
    python_requires=">=3.10",
    install_requires=[line.strip() for line in open("requirements.txt") if line.strip()],
    extras_require={"dev": ["pytest>=8"]},
    entry_points={
        "console_scripts": [
            "medrestore-ui=web.server:main",
            "medrestore-train=scripts.train:main",
            "medrestore-evaluate=scripts.evaluate:main",
            "medrestore-restore=scripts.restore_image:main",
            "medrestore-benchmark=scripts.benchmark:main",
        ],
    },
)
