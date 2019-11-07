import setuptools

# https://dzone.com/articles/executable-package-pip-install
# python3 setup.py bdist_wheel
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='myai',
     version='1.5',
     scripts=['myai'] ,
     author="zhangxk",
     author_email="mckj_zhangxk@163@com",
     description="A package Test package",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/mckjzhangxk/deepAI.git",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     package_data={'cfai': ['README.md']},
 )