import os
import sys

import better

from mock import Mock as MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
            return Mock()

sys.modules.update((mod, Mock()) for mod in 'ode'.split())


# ----8<---- from pyglet documentation conf.py ----8<----
implementations = ["carbon", "cocoa", "win32", "xlib"]
skip_modules = dict(pyglet={
    "pyglet.com": None,
    "pyglet.compat": None,
    "pyglet.lib": None,
    "pyglet.libs": None,
    "pyglet.app": implementations,
    "pyglet.canvas": implementations + ["xlib_vidmoderestore"],
    "pyglet.font": [
        "carbon", "quartz", "win32", "freetype", "freetype_lib", "win32query"],
    "pyglet.input": [
        "carbon_hid", "carbon_tablet", "darwin_hid", "directinput", "evdev",
        "wintab", "x11_xinput", "x11_xinput_tablet"],
    "pyglet.image.codecs": [
        "gdiplus", "gdkpixbuf2", "pil", "quartz", "quicktime"],
    "pyglet.gl": implementations + [
        "agl", "glext_arb", "glext_nv", "glx", "glx_info", "glxext_arb",
        "glxext_mesa", "glxext_nv", "lib_agl", "lib_glx", "lib_wgl", "wgl",
        "wgl_info", "wglext_arb", "wglext_nv"],
    "pyglet.media.drivers": ["directsound", "openal", "pulse"],
    "pyglet.media.sources": ["avbin"],
    "pyglet.window": implementations,
})
def skip_member(member, obj):
    module = obj.__name__
    if module=="tests.test": return True
    if ".win32" in module: return True
    if ".carbon" in module: return True
    if ".cocoa" in module: return True
    if ".xlib" in module: return True
    if module=="pyglet.input.evdev_constants": return True
    if module=="pyglet.window.key":
        if member==member.upper(): return True
    if module=="pyglet.gl.glu": return True
    if module.startswith("pyglet.gl.glext_"): return True
    if module.startswith("pyglet.gl.gl_ext_"): return True
    if module.startswith("pyglet.gl.glxext_"): return True
    if module.startswith("pyglet.image.codecs."): return True
    if module!="pyglet.gl.gl":
        if member in ["DEFAULT_MODE", "current_context"]:
            return True
    if member.startswith("PFN"): return True
    if member.startswith("GL_"): return True
    if member.startswith("GLU_"): return True
    if member.startswith("RTLD_"): return True
    if member=="GLvoid": return True
    if len(member)>4:
        if member.startswith("gl") and member[2]==member[2].upper():
            return True
        if member.startswith("glu") and member[3]==member[3].upper():
            return True
    return False
sys.skip_member = skip_member
sys.all_submodules = find_all_modules(document_modules, skip_modules)
# ----8<---- end from pyglet documentation conf.py ----8<----

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    #'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    #'sphinx.ext.pngmath',
    'sphinx.ext.viewcode',
    'numpydoc',
    ]
autosummary_generate = True
autodoc_default_flags = ['members']
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = True
source_suffix = '.rst'
source_encoding = 'utf-8-sig'
master_doc = 'index'
project = u'Pagoda'
copyright = u'2015, Leif Johnson'
version = '0.1'
release = '0.1.0'
exclude_patterns = ['_build']
templates_path = ['_templates']
pygments_style = 'tango'

html_theme = 'better'
html_theme_path = [better.better_theme_path]
html_theme_options = dict(
  rightsidebar=False,
  inlinecss='',
  cssfiles=['_static/style-tweaks.css'],
  showheader=True,
  showrelbartop=True,
  showrelbarbottom=True,
  linktotheme=True,
  sidebarwidth='15rem',
  textcolor='#111',
  headtextcolor='#333',
  footertextcolor='#333',
  ga_ua='',
  ga_domain='',
)
html_short_title = 'Home'
html_static_path = ['_static']

def h(xs):
    return ['{}.html'.format(x) for x in xs.split()]
html_sidebars = {
    'index': h('gitwidgets globaltoc sourcelink searchbox'),
    '**': h('gitwidgets localtoc sourcelink searchbox'),
}

intersphinx_mapping = {
    'python': ('http://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
}
