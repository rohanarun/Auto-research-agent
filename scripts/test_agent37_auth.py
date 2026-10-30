"""Cloud-only credential transfer: real one-use listener, no API/model calls."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('agent37_bridge', ROOT / 'Resources/DaytonaWorker.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class Agent37AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=ROOT / 'tmp', prefix='agent37-auth-')
        self.root = Path(self.folder.name)
        self.original = bridge.R
        bridge.R = self.root

    def tearDown(self):
        bridge.R = self.original
        self.folder.cleanup()

    def test_listener_rejects_wrong_token_and_serves_auth_only_once(self):
        auth = self.root / 'connection/auth.json'
        auth.parent.mkdir()
        raw = b'{"tokens":{"fixture":"cloud-only"}}'
        auth.write_bytes(raw)
        ready = threading.Event()
        port = []
        def announce(value, **kwargs):
            port.append(value)
            ready.set()
        with mock.patch.object(bridge, 'print', announce, create=True):
            thread = threading.Thread(target=bridge.serve_auth, args=('single-use-token',), daemon=True)
            thread.start()
            self.assertTrue(ready.wait(3))
            url = f'http://127.0.0.1:{port[0]}/'
            with self.assertRaises(urllib.error.HTTPError) as failed:
                urllib.request.urlopen(url, timeout=2)
            self.assertEqual(failed.exception.code, 403)
            request = urllib.request.Request(url, headers={'X-Research-Transfer': 'single-use-token'})
            with urllib.request.urlopen(request, timeout=2) as response:
                self.assertEqual(response.headers.get("Cache-Control"), "no-store")
                self.assertEqual(response.read(), raw)
            thread.join(3)
            self.assertFalse(thread.is_alive(), 'Listener must close after one successful transfer')
            with self.assertRaises(urllib.error.URLError):
                urllib.request.urlopen(request, timeout=1)

    def test_receive_stores_only_auth_privately_and_refuses_redirects(self):
        raw = b'{"tokens":{"fixture":"cloud-only"}}'
        response = io.BytesIO(raw)
        opener = mock.Mock()
        opener.open.return_value = contextlib.closing(response)
        with mock.patch.object(bridge.urllib.request, 'build_opener', return_value=opener) as factory:
            receipt = bridge.receive_auth_transfer({'url': 'https://ab12cd34ef-12345.agent37.app/?a37_token=temporary', 'token': 'one-use'})
            handler = factory.call_args.args[0]
            with self.assertRaises(RuntimeError):
                handler.redirect_request(None, None, None, None, None, None)
        destination = self.root / 'auth/auth.json'
        self.assertEqual(destination.read_bytes(), raw)
        self.assertEqual(destination.stat().st_mode & 0o777, 0o600)
        self.assertEqual(receipt, {'copied': True, 'host_auth_read': False, 'provider_history_copied': False})
        request = opener.open.call_args.args[0]
        self.assertEqual(dict(request.header_items()), {'X-research-transfer': 'one-use'})
        self.assertNotIn('cloud-only', json.dumps(receipt))
        for bad in ['http://ab12cd34ef-12345.agent37.app', 'https://evil.example/', 'https://ab12cd34ef-12345.agent37.app@evil.example']:
            with self.assertRaises(AssertionError):
                bridge.receive_auth_transfer({'url': bad, 'token': 'one-use'})

    def test_invalid_auth_cannot_replace_existing_signin(self):
        destination = self.root / 'auth/auth.json'
        destination.parent.mkdir()
        destination.write_bytes(b'{"existing":true}')
        opener = mock.Mock()
        opener.open.return_value = contextlib.closing(io.BytesIO(b'not-json'))
        with mock.patch.object(bridge.urllib.request, 'build_opener', return_value=opener):
            with self.assertRaises(json.JSONDecodeError):
                bridge.receive_auth_transfer({'url':'https://ab12cd34ef-12345.agent37.app/', 'token':'one-use'})
        self.assertEqual(destination.read_bytes(), b'{"existing":true}')


if __name__ == '__main__':
    unittest.main()
