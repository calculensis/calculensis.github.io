AUTHOR = 'Kayla Lewis'
SITENAME = 'the decision blog'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'America/New_York'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = 'feeds/all.atom.xml'
FEED_ALL_RSS = 'feeds/all.rss.xml'
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = 'feeds/%s.rss.xml'
RSS_FEED_SUMMARY_ONLY = False

# Blogroll
LINKS = (('Pelican', 'https://getpelican.com/'),
         ('Python.org', 'https://www.python.org/'),
         ('Jinja2', 'https://palletsprojects.com/p/jinja/'),)

# Social widget
SOCIAL = (('follow me @Estimatrix', 'https://twitter.com/Estimatrix'),
          ('or on Facebook','https://www.facebook.com/caela.calculensis/'),
          ('or at LinkedIn','https://www.linkedin.com/in/kayla-lewis-79ba9ba1/'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
STATIC_PATHS = [
    'images','../CNAME','category'
    ]
EXTRA_PATH_METADATA = {
    '../CNAME': {'path': 'CNAME'},
    }
PLUGIN_PATHS=['./plugins']
PLUGINS = ['render_math']
THEME = 'Pelican-Cid'
