# Configuration file for jupyter-notebook.

#------------------------------------------------------------------------------
# Application(SingletonConfigurable) configuration
#------------------------------------------------------------------------------

## Set the log level by value or name.
c.Application.log_level = 'INFO'

#------------------------------------------------------------------------------
# NotebookApp(JupyterApp) configuration
#------------------------------------------------------------------------------

## The IP address the notebook server will listen on.
c.NotebookApp.ip = '*'

# Password to access the server
c.NotebookApp.password = ''
c.NotebookApp.allow_password_change = False

## The directory to use for notebooks and kernels.
c.NotebookApp.notebook_dir = '/notebooks'

## Whether to open in a browser after starting. The specific browser used is
#  platform dependent and determined by the python standard library `webbrowser`
#  module, unless it is overridden using the --browser (NotebookApp.browser)
#  configuration option.
c.NotebookApp.open_browser = False

## The port the notebook server will listen on.
c.NotebookApp.port = 8888

## Allow running as root
c.NotebookApp.allow_root = True

## Disable the "Quit" button on the Web UI (shuts down the server)
c.NotebookApp.quit_button = False
