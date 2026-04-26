from prompt_siren.agents.mini_swe import MiniSweAgent, MiniSweAgentConfig
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage, RunUsage


def _agent() -> MiniSweAgent:
    return MiniSweAgent(
        _config=MiniSweAgentConfig(
            mini_swe_agent_path=None,
            config_specs=[],
        )
    )


def test_mini_response_usage_is_recorded() -> None:
    response = _agent()._mini_response_to_model_response(
        {
            "content": "I will inspect the repo.",
            "extra": {
                "actions": [{"command": "pwd"}],
                "response": {
                    "model": "openai/gpt-4o",
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                        "prompt_tokens_details": {
                            "cached_tokens": 3,
                            "audio_tokens": 0,
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": 2,
                            "audio_tokens": 0,
                        },
                    },
                },
            },
        }
    )

    assert response.model_name == "openai/gpt-4o"
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.usage.cache_read_tokens == 3
    assert response.usage.details["total_tokens"] == 18
    assert response.usage.details["completion_tokens_details.reasoning_tokens"] == 2
    assert any(isinstance(part, TextPart) for part in response.parts)
    assert any(isinstance(part, ToolCallPart) for part in response.parts)


def test_append_message_accumulates_model_usage() -> None:
    run_ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(input_tokens=5))
    model_request = ModelRequest(parts=[UserPromptPart("hello")])
    model_response = ModelResponse(
        parts=[TextPart("world")],
        usage=RequestUsage(input_tokens=13, output_tokens=8),
    )

    updated_ctx = MiniSweAgent._append_message(run_ctx, model_request, model_response)

    assert run_ctx.usage.input_tokens == 5
    assert updated_ctx.usage.input_tokens == 18
    assert updated_ctx.usage.output_tokens == 8
    assert updated_ctx.usage.requests == 1
    assert updated_ctx.messages == [model_request, model_response]
