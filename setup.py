from setuptools import setup, find_packages

setup(
    name="mandarin_visual_speech_representation_tool", 
    version="1.1.0",  
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    python_requires=">=3.6", 
    author="John Henry Cruz",
    author_email="johnhenrycruz000@gmail.com",
    description="A tool for Mandarin Visual Speech Representation",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kid-codei/mandarin_visual_speech_representation_tool",
)
