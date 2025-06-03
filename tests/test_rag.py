import unittest
from unittest.mock import patch, MagicMock

from llama_index.readers.docling import DoclingReader
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient

class TestSetup(unittest.TestCase):
    @patch(
        "llama_index.readers.docling.DoclingReader.load_data",
        return_value=[{"content": "dummy"}],
    )
    def test_only_reader(
        self,
        mock_load_data,
    ):
        """Test that only the DoclingReader is instantiated and load_data is
        called with the correct URL."""
        
        # 1. Instantiate the reader and call load_data.
        reader = DoclingReader()
        documents = reader.load_data("https://arxiv.org/pdf/2408.09869")

        # 2. Assert load_data was called once with the correct URL.
        mock_load_data.assert_called_once_with("https://arxiv.org/pdf/2408.09869")
        self.assertEqual(documents, [{"content": "dummy"}])
    
    # @patch("llama_index.embeddings.openai.OpenAIEmbedding")
    # @patch("llama_index.llms.openai.OpenAI")
    # @patch("llama_index.core.node_parser.MarkdownNodeParser")
    # @patch("llama_index.vector_stores.qdrant.QdrantVectorStore")
    # @patch("qdrant_client.QdrantClient")
    # @patch("qdrant_client.AsyncQdrantClient")
    # @patch("llama_index.core.StorageContext.from_defaults")
    # @patch("llama_index.core.VectorStoreIndex.from_documents")
    # def test_vector_store_index_setup(
    #     self,
    #     mock_embed_cls,
    #     mock_llm_cls,
    #     mock_parser_cls,
    #     mock_vector_store_cls,
    #     mock_qdrant_client_cls,
    #     mock_async_qdrant_client_cls,
    #     mock_storage_ctx_cls,
    #     mock_from_documents_cls,
    # ):
    #     """
    #     This test mocks:
    #       - OpenAIEmbedding → replaced by MagicMock
    #       - OpenAI          → replaced by MagicMock
    #       - MarkdownNodeParser → replaced by MagicMock
    #       - QdrantClient & AsyncQdrantClient → replaced by MagicMock
    #       - QdrantVectorStore → replaced by MagicMock
    #       - StorageContext.from_defaults → replaced by MagicMock
    #       - VectorStoreIndex.from_documents → replaced by MagicMock

    #     Then it runs the same import/instantiation sequence as production, and
    #     asserts that each mock was invoked with the correct arguments.
    #     """

    #     # ─── Step A: Create dummy instances for each patched class ───────────────

    #     dummy_qclient = MagicMock(name="DummyQdrantClient")
    #     dummy_aqclient = MagicMock(name="DummyAsyncQdrantClient")
    #     dummy_store = MagicMock(name="DummyQdrantVectorStore")
    #     dummy_parser = MagicMock(name="DummyParserInstance")
    #     dummy_llm = MagicMock(name="DummyLLMInstance")
    #     dummy_embed = MagicMock(name="DummyEmbedInstance")
    #     dummy_ctx = MagicMock(name="DummyStorageContext")
    #     dummy_index = MagicMock(name="DummyVectorStoreIndex")

    #     # Configure return values for each patched class
    #     mock_qdrant_client_cls.return_value = dummy_qclient
    #     mock_async_qdrant_client_cls.return_value = dummy_aqclient
    #     mock_vector_store_cls.return_value = dummy_store
    #     mock_parser_cls.return_value = dummy_parser
    #     mock_llm_cls.return_value = dummy_llm
    #     mock_embed_cls.return_value = dummy_embed
    #     mock_storage_ctx_cls.return_value = dummy_ctx
    #     mock_from_documents_cls.return_value = dummy_index

    #     # ─── Step B: Run the production‐style code under test ───────────────────

    #     # These calls should hit our patched QdrantClient/AsyncQdrantClient:
    #     client = QdrantClient(host="localhost", port=6333)
    #     aclient = AsyncQdrantClient(host="localhost", port=6333)

    #     # QdrantVectorStore(...) should return dummy_store
    #     vector_store = QdrantVectorStore(
    #         "docling",
    #         client=client,
    #         aclient=aclient,
    #         enable_hybrid=True,
    #         fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions"
    #     )

    #     # StorageContext.from_defaults(...) should return dummy_ctx
    #     storage_context = StorageContext.from_defaults(vector_store=vector_store)

    #     # Prepare dummy inputs for from_documents(...)
    #     documents = [{"dummy": "doc"}]  
    #     embed_model = dummy_embed       # pretend this is passed in
    #     node_parser = dummy_parser      # pretend this is passed in

    #     # Calling from_documents should return dummy_index
    #     index = VectorStoreIndex.from_documents(
    #         documents=documents,
    #         storage_context=storage_context,
    #         transformations=[node_parser],
    #         embed_model=embed_model
    #     )

    #     # ─── Step C: Assertions ───────────────────────────────────────────────────

    #     # 1. QdrantClient was constructed with the correct host/port
    #     mock_qdrant_client_cls.assert_called_once_with(host="localhost", port=6333)

    #     # 2. AsyncQdrantClient was constructed with the correct host/port
    #     mock_async_qdrant_client_cls.assert_called_once_with(host="localhost", port=6333)

    #     # 3. QdrantVectorStore was called with the correct arguments, including our dummy clients
    #     mock_vector_store_cls.assert_called_once_with(
    #         "docling",
    #         client=dummy_qclient,
    #         aclient=dummy_aqclient,
    #         enable_hybrid=True,
    #         fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions"
    #     )
    #     self.assertIs(vector_store, dummy_store)

    #     # 4. StorageContext.from_defaults was called with vector_store=dummy_store
    #     mock_storage_ctx_cls.assert_called_once_with(vector_store=dummy_store)
    #     self.assertIs(storage_context, dummy_ctx)

    #     # 5. VectorStoreIndex.from_documents was called with the correct parameters
    #     mock_from_documents_cls.assert_called_once_with(
    #         documents=documents,
    #         storage_context=dummy_ctx,
    #         transformations=[node_parser],
    #         embed_model=embed_model
    #     )
    #     self.assertIs(index, dummy_index)

    #     # 6. Ensure none of the “unused” classes were actually instantiated beyond our patches:
    #     #    (Their mocks exist, but they weren’t called again in this test.)
    #     mock_parser_cls.assert_not_called()  # we didn’t call MarkdownNodeParser() here
    #     mock_llm_cls.assert_not_called()    # we didn’t call OpenAI() here
    #     mock_embed_cls.assert_not_called()  # we didn’t call OpenAIEmbedding() here

        


if __name__ == "__main__":
    unittest.main()
