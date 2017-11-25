
try:
  import cb
except ImportError:
  print "downloading complicated build..."
  import urllib2
  response = urllib2.urlopen('https://raw.github.com/joe-jordan/complicated_build/master/cb/__init__.py')
  content = response.read()
  f = open('cb.py', 'w')
  f.write(content)
  f.close()
  import cb
  print "done!"
  
import numpy as np


global_includes = [np.get_include()]
global_macros = [("__FORCE_CPP__",)]

extensions = [
  {'name' : 'invsolver',
    'sources' : [
      'invsolver.pyx',
      'calcul.f',
      'fast_surf.f',
      'flat1.f',
      'init.f',
      'mchdepsun.f',
      'surfa.f'
      
  ]},
  
]

import datetime
cb.setup(extensions, global_macros = global_macros, global_includes = global_includes)(
  name="invsolver",
  version=datetime.date.today().strftime("%d%b%Y"),

  url="TBA",
  packages=["insolver"]
)

