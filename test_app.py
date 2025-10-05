import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, orchestrate

class TestOfficeCube(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()

    def test_healthz(self):
        """Smoke test: Health endpoint."""
        rv = self.client.get('/healthz')
        self.assertEqual(rv.status_code, 200)
        data = rv.get_json()
        self.assertEqual(data['status'], 'healthy')

    def test_orchestrate_valid(self):
        """Unit: Valid task routes and responds."""
        rv = self.client.post('/orchestrate', json={'task': 'Design layout'})
        self.assertEqual(rv.status_code, 200)
        data = rv.get_json()
        self.assertIn('response', data)
        self.assertGreater(len(data['response']), 0)

    def test_orchestrate_invalid_length(self):
        """Unit: Invalid input rejected."""
        long_task = 'a' * 501
        rv = self.client.post('/orchestrate', json={'task': long_task})
        self.assertEqual(rv.status_code, 400)

if __name__ == '__main__':
    unittest.main()
