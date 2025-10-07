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

    @unittest.mock.patch('app.db.session.commit')
    @unittest.mock.patch('app.db.session.add')
    @unittest.mock.patch('app._build_agent_chains')
    def test_orchestrate_langchain(self, mock_build_chains, mock_db_add, mock_db_commit):
        """Integration: LangChain orchestration with mocks."""
        mock_chains = {
            "Orchestration": unittest.mock.MagicMock(),
            "CEO": unittest.mock.MagicMock()
        }
        mock_chains["Orchestration"].invoke.return_value = {"agent": "CEO", "subtask": "Test", "chain_next": False}
        mock_chains["CEO"].invoke.return_value.content = "Mock response"
        mock_build_chains.return_value = mock_chains

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

    @unittest.mock.patch('app._build_agent_chains')
    def test_orchestrate_with_langchain_success(self, mock_build_chains):
        """Unit: LangChain orchestration succeeds."""
        mock_chains = {
            "Orchestration": unittest.mock.MagicMock(),
            "CEO": unittest.mock.MagicMock()
        }
        mock_chains["Orchestration"].invoke.return_value = {"agent": "CEO", "subtask": "Test", "chain_next": False}
        mock_chains["CEO"].invoke.return_value.content = "CEO Response"
        mock_build_chains.return_value = mock_chains

        result = orchestrate_with_langchain("Test task")
        self.assertIn("CEO", str(result['agents_involved']))
        self.assertEqual(result['response'], "CEO Response")

    @unittest.mock.patch('app._build_agent_chains')
    def test_orchestrate_with_chaining(self, mock_build_chains):
        """Unit: LangChain orchestration with chaining."""
        mock_chains = {
            "Orchestration": unittest.mock.MagicMock(),
            "Manager": unittest.mock.MagicMock(),
            "Architect": unittest.mock.MagicMock()
        }
        
        mock_chains["Orchestration"].invoke.return_value = {"agent": "Manager", "subtask": "Plan the project", "chain_next": True, "next_agent": "Architect"}
        mock_chains["Manager"].invoke.return_value.content = "Project plan"
        mock_chains["Architect"].invoke.return_value.content = "System design"
        mock_build_chains.return_value = mock_chains

        result = orchestrate_with_langchain("Develop a new feature")
        self.assertIn("Manager", str(result['agents_involved']))
        self.assertIn("Architect", str(result['agents_involved']))
        self.assertEqual(result['response'], "System design")

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