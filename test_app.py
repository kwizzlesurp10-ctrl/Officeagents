import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, get_agent_response, AGENTS
import unittest.mock

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
        """Integration: Valid task routes and gets LLM-flavored response."""
        with unittest.mock.patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "Mock LLM response"}}]
            }
            mock_post.return_value.raise_for_status = lambda: None
            rv = self.client.post('/orchestrate', json={'task': 'Design layout'})
            self.assertEqual(rv.status_code, 200)
            data = rv.get_json()
            self.assertIn('response', data)
            self.assertEqual(data['response'], 'Mock LLM response')
            self.assertIn('Architect', str(data['agents_involved']))

    def test_orchestrate_invalid_length(self):
        """Unit: Invalid input rejected."""
        long_task = 'a' * 501
        rv = self.client.post('/orchestrate', json={'task': long_task})
        self.assertEqual(rv.status_code, 400)

    def test_agent_prompts(self):
        """Unit: Verify all expanded prompts exist and are detailed."""
        self.assertEqual(len(AGENTS), 9)
        for agent, prompt in AGENTS.items():
            self.assertGreater(len(prompt), 100)  # Ensures expansion
            self.assertIn("You are", prompt)  # Basic structure check

    @unittest.mock.patch('requests.post')
    def test_get_agent_response_success(self, mock_post):
        """Unit: LLM call succeeds."""
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Success response"}}]
        }
        mock_post.return_value.raise_for_status = lambda: None
        response = get_agent_response("CEO", "Test task")
        self.assertEqual(response, "Success response")

    @unittest.mock.patch('requests.post')
    def test_get_agent_response_fallback(self, mock_post):
        """Unit: LLM call fails, uses fallback."""
        mock_post.side_effect = Exception("API error")
        response = get_agent_response("CEO", "Test task")
        self.assertIn("As CEO", response)
        self.assertIn("API fallback", response)

if __name__ == '__main__':
    unittest.main()
