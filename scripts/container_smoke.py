"""Smoke a running preview container; no trained-model claim or fake scores."""
import argparse
import http.client
import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def request(base, path, expected, data=None, headers=None):
    req = urllib.request.Request(base + path, data=data, headers=headers or {})
    try:
        response = urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        body = response.read().decode()
        assert response.code == expected, (path, response.code, body)
        return json.loads(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    for attempt in range(60):
        try:
            request(args.url, '/healthz', 200)
            break
        except (OSError, AssertionError):
            if attempt == 59:
                raise
            time.sleep(1)
    status = request(args.url, '/api/status', 200)
    assert status['mode'] == 'preview' and status['ready'] is False
    request(args.url, '/readyz', 503)
    body = b'--test\r\nContent-Disposition: form-data; name="file"; filename="invalid.mp4"\r\nContent-Type: video/mp4\r\n\r\ninvalid\r\n--test--\r\n'
    request(args.url, '/api/analyze', 503, body,
            {'Content-Type': 'multipart/form-data; boundary=test'})
    # The middleware rejects excessive Content-Length before reading the body.
    # Read that early response instead of racing a 33 MiB send against a closed
    # socket. Chunked-body enforcement is covered by the integration suite.
    target = urlsplit(args.url)
    connection_type = http.client.HTTPSConnection if target.scheme == 'https' else http.client.HTTPConnection
    connection = connection_type(target.hostname, target.port, timeout=5)
    try:
        connection.putrequest('POST', '/api/analyze')
        connection.putheader('Content-Length', str(33 * 1024 * 1024))
        connection.putheader('Content-Type', 'application/octet-stream')
        connection.endheaders()
        response = connection.getresponse()
        body = response.read().decode()
        assert response.status == 413, (response.status, body)
    finally:
        connection.close()
    print(json.dumps({'result': 'Passed', 'liveness': 200, 'readiness': 503,
                      'mode': 'preview', 'analysis_without_model': 503,
                      'oversize_upload': 413}))


if __name__ == '__main__':
    main()
