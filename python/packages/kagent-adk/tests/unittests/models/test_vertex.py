import ssl
from unittest import mock

from kagent.adk.models._vertex import KAgentClaudeVertexLlm, KAgentGeminiVertexLlm
from kagent.adk.types import GeminiAnthropic, GeminiVertexAI, _create_llm_from_model_config


def test_gemini_vertex_http_options_apply_tls_to_sync_and_async_clients():
    ssl_context = mock.MagicMock(spec=ssl.SSLContext)
    with mock.patch("kagent.adk.models._transport.create_ssl_context", return_value=ssl_context) as create_ssl:
        llm = KAgentGeminiVertexLlm(
            model="gemini-2.5-flash",
            tls_ca_cert_path="/etc/kagent/tls/ca.crt",
            tls_disable_system_cas=True,
        )

        http_options = llm._http_options()

    create_ssl.assert_called_once_with(
        disable_verify=False,
        ca_cert_path="/etc/kagent/tls/ca.crt",
        disable_system_cas=True,
    )
    assert http_options.client_args == {"verify": ssl_context}
    assert http_options.async_client_args == {"verify": ssl_context}


def test_claude_vertex_client_applies_tls_without_replacing_vertex_auth():
    ssl_context = mock.MagicMock(spec=ssl.SSLContext)
    http_client = mock.MagicMock()
    with mock.patch.dict(
        "os.environ",
        {
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "GOOGLE_CLOUD_REGION": "us-central1",
        },
        clear=False,
    ):
        with mock.patch("kagent.adk.models._transport.create_ssl_context", return_value=ssl_context):
            with mock.patch("kagent.adk.models._vertex.httpx.AsyncClient", return_value=http_client) as async_client:
                with mock.patch("kagent.adk.models._vertex.AsyncAnthropicVertex") as anthropic_vertex:
                    llm = KAgentClaudeVertexLlm(
                        model="claude-3-5-sonnet-v2@20241022",
                        extra_headers={"X-Test": "value"},
                        tls_ca_cert_path="/etc/kagent/tls/ca.crt",
                    )

                    _ = llm._anthropic_client

    async_client.assert_called_once_with(verify=ssl_context)
    anthropic_vertex.assert_called_once()
    kwargs = anthropic_vertex.call_args.kwargs
    assert kwargs["project_id"] == "test-project"
    assert kwargs["region"] == "us-central1"
    assert kwargs["http_client"] is http_client
    assert kwargs["default_headers"]["X-Test"] == "value"


def test_model_config_factory_uses_kagent_vertex_wrappers():
    gemini = _create_llm_from_model_config(
        GeminiVertexAI(
            type="gemini_vertex_ai",
            model="gemini-2.5-flash",
            tls_disable_verify=True,
        )
    )
    claude = _create_llm_from_model_config(
        GeminiAnthropic(
            type="gemini_anthropic",
            model="claude-3-5-sonnet-v2@20241022",
            tls_disable_verify=True,
        )
    )

    assert isinstance(gemini, KAgentGeminiVertexLlm)
    assert isinstance(claude, KAgentClaudeVertexLlm)
    assert gemini.tls_disable_verify is True
    assert claude.tls_disable_verify is True
