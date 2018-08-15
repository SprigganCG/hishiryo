from setuptools import setup

setup(name='hishiryo',
      version='0.1',
      description='Render a csv dataset into a picture',
      url='http://github.com/spriggancg/hishiryo',
      author='SprigganCG',
      author_email='spriggancg@gmail.com',
      license='MIT',
      packages=['hishiryo'],
      install_requires=['pandas','Pillow','numpy','opencv-python','svgwrite'],
      zip_safe=False)