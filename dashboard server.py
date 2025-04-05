import http.server
import socketserver
import os

PORT = 8080
DIRECTORY = os.path.abspath('.')

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == '__main__':
    print(f"Serving dashboard at http://localhost:{PORT}/dashboard.html")
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        httpd.serve_forever()
