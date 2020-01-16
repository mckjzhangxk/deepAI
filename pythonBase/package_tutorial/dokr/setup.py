import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='myfirstzxk',  
     version='0.1',
     scripts=['dokr'] ,
     author="zhangxk",
     author_email="mckj_zhangxk@163@com",
     description="A package Test package",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/javatechy/dokr",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     package_data={'myzxk': ['README.md']},
 )