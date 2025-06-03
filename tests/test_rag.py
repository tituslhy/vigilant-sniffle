import unittest
from unittest.mock import patch

from llama_index.readers.docling import DoclingReader


class TestSetup(unittest.TestCase):
    """
    A TestCase that patches out:
      - OpenAIEmbedding   (so no real embeddings)
      - OpenAI            (so no real LLM)
      - MarkdownNodeParser (so no parser logic)
      - DoclingReader.load_data (so no HTTP I/O)

    In this specific test, we only instantiate the reader and call load_data.
    """

    @patch(
        "llama_index.readers.docling.DoclingReader.load_data",
        return_value=[{"content": "dummy"}],
    )
    @patch("llama_index.core.node_parser.MarkdownNodeParser")
    @patch("llama_index.llms.openai.OpenAI")
    @patch("llama_index.embeddings.openai.OpenAIEmbedding")
    def test_only_reader(
        self,
        mock_embed_cls,
        mock_llm_cls,
        mock_parser_cls,
        mock_load_data,
    ):
        # We do NOT call OpenAIEmbedding(), OpenAI(), or MarkdownNodeParser() here.

        # 1. Instantiate the reader and call load_data.
        reader = DoclingReader()
        documents = reader.load_data("https://arxiv.org/pdf/2408.09869")

        # 2. Assert load_data was called once with the correct URL.
        mock_load_data.assert_called_once_with("https://arxiv.org/pdf/2408.09869")
        self.assertEqual(documents, [{"content": "dummy"}])

        # 3. Ensure embedding and LLM classes were never instantiated.
        mock_embed_cls.assert_not_called()
        mock_llm_cls.assert_not_called()

        # 4. Ensure the parser class was never instantiated.
        mock_parser_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
