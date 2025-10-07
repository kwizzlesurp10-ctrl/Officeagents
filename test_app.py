import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, orchestrate_with_langchain, AGENT_PROMPTS
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

    @unittest.mock.patch('langchain_core.output_parsers.JsonOutputParser')
    @unittest.mock.patch('app.llm')
    def test_orchestrate_langchain(self, mock_llm, mock_parser):
        """Integration: LangChain orchestration with mocks."""
        mock_chain = unittest.mock.MagicMock()
        mock_chain.invoke.return_value = "Mock response"
        mock_llm.return_value = mock_chain
        mock_parser.return_value = mock_chain

        # Mock router output
        mock_router = unittest.mock.MagicMock()
        mock_router.invoke.return_value = {"agent": "CEO", "subtask": "Test", "chain_next": False}
        # Patch agent_chains indirectly via global mock

        rv = self.client.post('/orchestrate', json={'task': 'Strategy plan'})
        self.assertEqual(rv.status_code, 200)
        data = rv.get_json()
        self.assertIn('response', data)
        self.assertEqual(data['response'], 'Mock response')

    def test_orchestrate_invalid_length(self):
        """Unit: Invalid input rejected."""
        long_task = 'a' * 501
        rv = self.client.post('/orchestrate', json={'task': long_task})
        self.assertEqual(rv.status_code, 400)

    def test_agent_prompts(self):
        """Unit: Verify all expanded prompts exist and are detailed."""
        self.assertEqual(len(AGENT_PROMPTS), 9)
        for agent, prompt in AGENT_PROMPTS.items():
            self.assertGreater(len(prompt), 100)  # Ensures expansion
            self.assertIn("You are", prompt)  # Basic structure check

    @unittest.mock.patch('app.llm')
    def test_orchestrate_with_langchain_success(self, mock_llm):
        """Unit: LangChain orchestration succeeds."""
        mock_chain = unittest.mock.MagicMock()
        mock_chain.invoke.side_effect = [
            {"agent": "CEO", "subtask": "Test", "chain_next": False},  # Router
            "CEO Response"  # Agent
        ]
        mock_llm.return_value = mock_chain

        result = orchestrate_with_langchain("Test task")
        self.assertIn("CEO", str(result['agents_involved']))
        self.assertEqual(result['response'], "CEO Response")

    @unittest.mock.patch('app.llm')
    def test_orchestrate_with_langchain_fallback(self, mock_llm):
        """Unit: LangChain orchestration falls back on error."""
        mock_llm.side_effect = Exception("Chain error")

        result = orchestrate_with_langchain("Test task")
        self.assertIn("Fallback", result['steps'][0])
        self.assertIn("Error", result['response'])

    @unittest.mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"})
    @unittest.mock.patch('langchain_google_genai.ChatGoogleGenerativeAI')
    def test_google_llm_selection(self, mock_google_llm):
        """Unit: Selects Google LLM when GOOGLE_API_KEY is set."""
        import importlib
        import app
        importlib.reload(app)
        mock_google_llm.assert_called_with(model="gemini-pro", google_api_key="test_key", temperature=0.7)

    @unittest.mock.patch.dict(os.environ, {"XAI_API_KEY": "test_key"})
    @unittest.mock.patch('langchain_xai.ChatXAI')
    def test_xai_llm_selection(self, mock_xai_llm):
        """Unit: Selects XAI LLM when XAI_API_KEY is set."""
        import importlib
        import app
        importlib.reload(app)
        mock_xai_llm.assert_called_with(model="grok-beta", xai_api_key="test_key", temperature=0.7)

if __name__ == '__main__':
    unittest.main()