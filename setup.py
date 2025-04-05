from setuptools import setup

APP = ['dashboard_server.py']
DATA_FILES = ['dashboard.html', 'trade_log.csv']
OPTIONS = {
    'argv_emulation': True,
    'packages': [],
    'iconfile': None,
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
