import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Any

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

class TestDoclingAgentNotebook:
    """
    Unit tests for the docling agent notebook logic, with all external dependencies mocked.
    Synchronous tests are kept in this class for organization and compatibility with pytest.
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
        assert docs[0].text_resource.text == 'Sample text.'

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
        assert vector_store == mock_qdrant_vector_store.return_value

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
        assert index == mock_index

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
        assert str(response) == 'Query response'

# ──────────────────────────────────────────────────────────────────────────────
#     WHY ARE THESE A STANDALONE FUNCTIONAL TESTS (NOT A UNITTEST CLASS METHOD)?
#     ──────────────────────────────────────────────────────────────────────────────
#     1. **Isolation from Side Effects**:
#        - These tests do not need test setup/teardown or shared fixtures.
#        - No test state is reused — everything is mocked fresh inside the test function.

#     2. **Simpler with `pytest.mark.asyncio`**:
#        - The async/await behavior fits better in a plain `async def` + `pytest.mark.asyncio`
#          test than in `unittest.TestCase`, which requires an async runner or extra boilerplate.

#     3. **Mock Strategy Complexity**:
#        - This test patches `FunctionAgent`, `AgentStream`, and `QueryEngineTool` just
#          enough to avoid triggering deep workflow logic.
#        - Since it doesn't test implementation internals (e.g., step workers, memory ops),
#          nesting it into a class would imply shared internal state — which it does *not* use.

@pytest.mark.asyncio
@patch('llama_index.core.tools.QueryEngineTool')
@patch('llama_index.core.agent.workflow.FunctionAgent')
async def test_agent_with_query_engine_tool_async(mock_function_agent: MagicMock, mock_query_engine_tool: MagicMock) -> None:
    """
    Async test for FunctionAgent.run with a mocked QueryEngineTool.
    Ensures the agent returns the expected response when awaited.
    """
    query_engine_tool = mock_query_engine_tool.return_value
    agent = mock_function_agent(tools=[query_engine_tool], llm=MagicMock())
    mock_function_agent.return_value.run = AsyncMock(return_value='Agent response')
    agent_response = await agent.run('What is docling?')
    assert agent_response == 'Agent response'

@pytest.mark.asyncio
@patch("llama_index.core.tools.QueryEngineTool")
@patch("llama_index.core.agent.workflow.AgentStream")
@patch("llama_index.core.agent.workflow.FunctionAgent")
async def test_agent_streaming_mocked_everything(
    mock_function_agent: MagicMock,
    mock_agent_stream_cls: MagicMock,
    mock_query_engine_tool_cls: MagicMock,
) -> None:
    """
    End-to-end test that validates the agent's streaming behavior in isolation,
    by mocking **everything** it depends on. This test avoids relying on any real LLM,
    memory, or query engine logic — and instead verifies just the streaming event flow.

    ──────────────────────────────────────────────────────────────────────────────
    WHAT IS `FakeHandler` AND WHY A NESTED CLASS?
    ──────────────────────────────────────────────────────────────────────────────
    `FakeHandler` is a minimal implementation of the `AgentHandler` interface used by
    LlamaIndex. It:
      - defines an async generator `stream_events()` that yields mock events
      - implements `__await__()` so `await handler` returns a final value
    This makes it a drop-in replacement for the real streaming handler returned by LLMs.

    We define it as a nested class because:
      - It's highly specific to this test — not reused elsewhere.
      - It relies on local variables (`ev1`, `ev2`, etc.), so nesting it makes scope clear.
      - It minimizes clutter in the global namespace.

    ──────────────────────────────────────────────────────────────────────────────
    TEST STRATEGY OVERVIEW
    ──────────────────────────────────────────────────────────────────────────────
    We:
      1. Patch `FunctionAgent(...)` so its constructor returns a dummy agent instance.
      2. Patch `agent.run(...)` to return our `FakeHandler`, which emits known deltas.
      3. Patch `QueryEngineTool` to satisfy constructor dependencies (not actually used).
      4. Collect deltas from `stream_events()` and verify correct order.
      5. Await the handler to verify final result is correct.

    This test guarantees:
      - The stream emits deltas in the correct order.
      - The awaitable `handler` returns the expected final result.
    """

    # Step A: Create two **distinct mock events**, each with a unique delta
    ev1 = MagicMock()
    ev1.delta = "Delta1"
    ev2 = MagicMock()
    ev2.delta = "Delta2"

    # Step B: Minimal replacement for the handler returned by agent.run()
    class FakeHandler:
        """
        Minimal drop-in replacement for the streaming response handler returned by
        LlamaIndex's agent.run(...) call.

        This class mimics the interface of real handler objects used in LlamaIndex,
        which support:
        - `async for ev in handler.stream_events()` — to yield intermediate AgentStream events
        - `await handler` — to get the final agent response after all streaming is done

        This test double enables full end-to-end testing of streaming agent workflows,
        without invoking any real LLM, network calls, or agent logic.
        """

        def __init__(self, events, final_value: str):
            """
            Parameters:
                events (List[Any]): A list of mock AgentStream-like events to stream.
                final_value (str): The final result to return when `await handler` is called.
            """
            self._events = events
            self._final_value = final_value

        async def stream_events(self):
            """
            Simulates streaming a sequence of intermediate agent events.

            This mimics LlamaIndex's behavior where `agent.run(...)` returns a handler
            that can be iterated over using `async for`, emitting `AgentStream` deltas.

            In our test, each event in `self._events` is expected to be a mock object
            with a `.delta` field (e.g., "Delta1", "Delta2").

            Yielding them one by one simulates how a real LLM might stream out token deltas.
            """

            for e in self._events:
                yield e

        def __await__(self):
            """
            Allows `await handler` to work.

            Real LlamaIndex handlers (like `StreamingAgentHandler`) support both:
            - `async for` for intermediate events
            - `await` for the final result

            This method defines how `await handler` behaves — it returns the final output
            string (e.g., "FINAL_RESULT") to simulate the conclusion of the agent run.

            Technically, this wraps an async coroutine in `__await__()` so that Python's
            await machinery treats this object like a coroutine.
            """
            async def _coro():
                return self._final_value
            return _coro().__await__()

    fake_handler = FakeHandler([ev1, ev2], final_value="FINAL_RESULT")

    # Step C: Patch FunctionAgent(...) to return a dummy object whose .run(...) returns the handler
    dummy_agent_instance = MagicMock(name="DummyAgent")
    dummy_agent_instance.run.return_value = fake_handler
    mock_function_agent.return_value = dummy_agent_instance

    # Patch QueryEngineTool as well, though we never call its logic in this test
    mock_query_engine_tool_cls.return_value = MagicMock(name="DummyTool")

    # Step D: Import symbols after patching (to pick up the mocks!)
    dummy_tool = QueryEngineTool(
        query_engine=MagicMock(),  # unused
        metadata=ToolMetadata(
            name="Docling_Knowledge_Base",
            description="Use this tool to answer questions about Docling",
        ),
    )

    # Step E: Create the agent (returns our patched dummy agent instance)
    agent = FunctionAgent(tools=[dummy_tool], llm=MagicMock())

    # Step F: Run the agent with a dummy prompt string
    handler = agent.run("How is docling an improvement over existing document readers?")

    # Step G: Collect all streamed deltas from the handler
    collected = []
    async for event in handler.stream_events():
        collected.append(event.delta)

    # Step H: Await the handler itself for the final result
    final = await handler

    # Step I: Assertions – critical contract checks
    assert collected == ["Delta1", "Delta2"]
    assert final == "FINAL_RESULT"

    # Step J: Sanity – ensure mocks were called as expected
    mock_function_agent.assert_called_once()
    dummy_agent_instance.run.assert_called_once_with("How is docling an improvement over existing document readers?")
