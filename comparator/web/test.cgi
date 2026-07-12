#!/bin/bash
# Minimal probe to check whether CGI execution is enabled for this directory.
#
# Deploy it next to index.html (chmod +x test.cgi), then from anywhere run:
#     curl -s http://<your-intranet-host>/<path>/test.cgi
#
#   * Prints "CGI OK ..."      -> CGI works; deploy build.cgi.
#   * Prints this script's TEXT -> CGI is NOT enabled; the server is serving the
#                                  file statically. Use `compare-web serve`
#                                  (SSH tunnel) or ask IT to enable ExecCGI /
#                                  reverse-proxy instead. See docs/DEPLOYMENT.md.
echo "Content-Type: text/plain"
echo ""
echo "CGI OK — this server executes scripts in this directory."
echo "python: $(command -v python3 || command -v python || echo 'not on PATH')"
