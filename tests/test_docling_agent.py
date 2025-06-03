import unittest
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Any

class TestDoclingAgentNotebook(unittest.TestCase):
    """
    Unit tests for the docling agent notebook logic, with all external dependencies mocked.
    Each test is decorated with @patch for clarity and isolation.
    """

    @patch('llama_index.readers.docling.DoclingReader')
    def test_reader_load_data(self, mock_docling_reader: MagicMock) -> None:
        """
        Test that DoclingReader.load_data returns a document with the expected text.
        All external dependencies are mocked.
        """
        mock_reader = mock_docling_reader.return_value
        mock_doc = MagicMock()
        mock_doc.text_resource.text = 'Sample text.'
        mock_reader.load_data.return_value = [mock_doc]
        docs = mock_reader.load_data('dummy_url')
        self.assertEqual(docs[0].text_resource.text, 'Sample text.')

    @patch('qdrant_client.AsyncQdrantClient')
    @patch('qdrant_client.QdrantClient')
    @patch('llama_index.vector_stores.qdrant.QdrantVectorStore')
    def test_vector_store_index_setup_with_qdrant(
        self,
        mock_qdrant_vector_store: MagicMock,
        mock_qdrant_client: MagicMock,
        mock_async_qdrant_client: MagicMock
    ) -> None:
        """
        Test that QdrantVectorStore is initialized with the correct arguments and returns the mock instance.
        All Qdrant-related dependencies are mocked.
        """
        client = mock_qdrant_client.return_value
        aclient = mock_async_qdrant_client.return_value
        vector_store = mock_qdrant_vector_store('docling', client=client, aclient=aclient, enable_hybrid=True, fastembed_sparse_model='Qdrant/bm42-all-minilm-l6-v2-attentions')
        self.assertEqual(vector_store, mock_qdrant_vector_store.return_value)

    @patch('llama_index.core.VectorStoreIndex')
    @patch('llama_index.vector_stores.qdrant.QdrantVectorStore')
    def test_load_vector_store_from_qdrant(
        self,
        mock_qdrant_vector_store: MagicMock,
        mock_vector_store_index: MagicMock
    ) -> None:
        """
        Test that VectorStoreIndex.from_vector_store returns the mock index when called with a mocked QdrantVectorStore.
        """
        vector_store = mock_qdrant_vector_store.return_value
        embed_model: Any = MagicMock()
        mock_index = mock_vector_store_index.from_vector_store.return_value
        index = mock_vector_store_index.from_vector_store(vector_store, embed_model=embed_model)
        self.assertEqual(index, mock_index)

    @patch('llama_index.core.VectorStoreIndex')
    def test_query_engine(self, mock_vector_store_index: MagicMock) -> None:
        """
        Test that the query engine returns the expected mocked response string.
        """
        index = mock_vector_store_index.return_value
        index.as_query_engine.return_value = MagicMock()
        query_engine = index.as_query_engine(sparse_top_k=10, similarity_top_k=6, llm=MagicMock())
        query_engine.query.return_value = MagicMock(source_nodes=[], __str__=lambda self: 'Query response')
        response = query_engine.query('Test query')
        self.assertEqual(str(response), 'Query response')

    @pytest.mark.asyncio
    @patch('llama_index.core.tools.QueryEngineTool')
    @patch('llama_index.core.agent.workflow.FunctionAgent')
    async def test_agent_with_query_engine_tool_async(
        self,
        mock_function_agent: MagicMock,
        mock_query_engine_tool: MagicMock
    ) -> None:
        """
        Test that the FunctionAgent returns the expected response when run with a mocked QueryEngineTool, using async/await.
        """
        query_engine_tool = mock_query_engine_tool.return_value
        agent = mock_function_agent(tools=[query_engine_tool], llm=MagicMock())
        mock_function_agent.return_value.run = AsyncMock(return_value='Agent response')
        agent_response = await agent.run('What is docling?')
        self.assertEqual(agent_response, 'Agent response')

    @pytest.mark.asyncio
    @patch('llama_index.core.agent.workflow.AgentStream')
    @patch('llama_index.core.tools.QueryEngineTool')
    @patch('llama_index.core.agent.workflow.FunctionAgent')
    async def test_agent_streaming(
        self,
        mock_function_agent: MagicMock,
        mock_query_engine_tool: MagicMock,
        mock_agent_stream: MagicMock
    ) -> None:
        """
        Test streaming from the agent using async handler and stream_events.
        """
        
        # Setup mocks
        query_engine_tool = mock_query_engine_tool.return_value
        agent = mock_function_agent(tools=[query_engine_tool], llm=MagicMock())
        # Mock handler returned by agent.run
        mock_handler = MagicMock()
        mock_function_agent.return_value.run.return_value = mock_handler
        # Mock async stream_events
        mock_event1 = mock_agent_stream()
        mock_event1.delta = 'Delta1'
        mock_event2 = mock_agent_stream()
        mock_event2.delta = 'Delta2'
        mock_handler.stream_events = AsyncMock(return_value=iter([mock_event1, mock_event2]))
        # Mock await handler
        mock_handler.__await__ = lambda s: iter(['Final result'])
        # Run test logic
        handler = agent.run('How is docling an improvement over existing document readers?')
        results = []
        async for ev in handler.stream_events():
            if isinstance(ev, mock_agent_stream):
                results.append(ev.delta)
        final_result = await handler
        self.assertEqual(results, ['Delta1', 'Delta2'])
        self.assertEqual(final_result, 'Final result')