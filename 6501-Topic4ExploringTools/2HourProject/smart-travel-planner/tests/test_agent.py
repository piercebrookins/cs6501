"""Integration tests for the travel planning agent.

Note: These tests require valid API keys and make real API calls.
Mark them with @pytest.mark.integration to skip in CI without keys.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestTravelAgentCreation:
    """Tests for agent creation and configuration."""
    
    def test_create_agent_imports(self):
        """Test that agent module can be imported."""
        from src.agent.travel_agent import create_travel_agent, SYSTEM_PROMPT
        
        assert callable(create_travel_agent)
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100  # Should have substantial content
    
    def test_system_prompt_contains_guidelines(self):
        """Test that system prompt includes key guidelines."""
        from src.agent.travel_agent import SYSTEM_PROMPT
        
        # Check for packing guidelines
        assert "Cold" in SYSTEM_PROMPT or "cold" in SYSTEM_PROMPT
        assert "Rain" in SYSTEM_PROMPT or "rain" in SYSTEM_PROMPT
        assert "Sunny" in SYSTEM_PROMPT or "sunny" in SYSTEM_PROMPT
        
        # Check for activity recommendations
        assert "Museum" in SYSTEM_PROMPT or "museum" in SYSTEM_PROMPT
        assert "outdoor" in SYSTEM_PROMPT.lower()
    
    @patch('src.agent.travel_agent.ChatOpenAI')
    @patch('src.agent.travel_agent.create_react_agent')
    def test_create_agent_uses_correct_model(self, mock_create_agent, mock_llm):
        """Test that agent is created with specified model."""
        from src.agent.travel_agent import create_travel_agent
        
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_create_agent.return_value = MagicMock()
        
        create_travel_agent(model="gpt-4o")
        
        mock_llm.assert_called_once_with(model="gpt-4o", temperature=0.7)
        # Verify create_react_agent was called with prompt parameter
        mock_create_agent.assert_called_once()
    
    @patch('src.agent.travel_agent.ChatOpenAI')
    @patch('src.agent.travel_agent.create_react_agent')
    def test_create_agent_custom_temperature(self, mock_create_agent, mock_llm):
        """Test that custom temperature is passed to LLM."""
        from src.agent.travel_agent import create_travel_agent
        
        mock_llm.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()
        
        create_travel_agent(temperature=0.3)
        
        mock_llm.assert_called_once_with(model="gpt-4o-mini", temperature=0.3)


@pytest.mark.integration
class TestTravelAgentIntegration:
    """
    Integration tests that require real API keys.
    
    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    """
    
    @pytest.fixture
    def agent(self):
        """Create a real agent instance."""
        from src.agent.travel_agent import create_travel_agent
        return create_travel_agent()
    
    def test_agent_handles_basic_query(self, agent):
        """Test that agent can process a basic travel query."""
        from src.agent.travel_agent import invoke_agent
        
        result = invoke_agent(agent, "What's the weather like in Tokyo?")
        
        # Should return a non-empty response
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_agent_provides_packing_advice(self, agent):
        """Test that agent provides packing recommendations."""
        from src.agent.travel_agent import invoke_agent
        
        result = invoke_agent(
            agent, 
            "I'm traveling to London next week, what should I pack?"
        )
        
        result_lower = result.lower()
        
        # Should mention packing-related items
        assert any(word in result_lower for word in [
            'pack', 'bring', 'wear', 'jacket', 'umbrella', 'clothes'
        ])
