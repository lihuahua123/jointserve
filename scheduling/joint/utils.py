import json
from dataclasses import field, dataclass
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any, Dict, List, Literal, Optional, Union
from typing_extensions import Required, TypedDict
from openai.types.chat import ChatCompletionContentPartImageParam
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam)
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam)
import codecs
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Awaitable, Iterable, List, Optional, Union, cast, final

# yapf conflicts with isort for this block
# yapf: disable
from openai.types.chat import ChatCompletionContentPartImageParam
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam)
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam)
# yapf: enable
# pydantic needs the TypedDict from typing_extensions
from pydantic import ConfigDict
from transformers import PreTrainedTokenizer
from typing_extensions import Required, TypedDict
from typing_extensions import Annotated
from typing_extensions import Iterable, Iterator, List, Optional, Tuple, TypedDict, Union
from transformers import AutoTokenizer,PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import GenerationConfig, PretrainedConfig
from transformers import AutoConfig
@dataclass
class EvictionData():
    input_ids: list
    evicted_ids: list

@dataclass
class RequestFuncOutput:
    rid: str = ""
    num_gpus: int = 0
    prompt_text: str = ""
    generated_text: str = ""
    success: bool = False
    request_latency: float = 0
    normalized_latency: float = 0
    ttft: float = 0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    global_time: float = 0
    output_len: float = None
    tpot: float = None
    prefill_decode_ratio: float = None
    send_out_time: float = 0.0
    arrival_time: float = 0.0
    append_to_queue_time: float = 0.0
    route_dest: int = None
    scheduling_overhead: float = 0.0
    runtime_selected :int = 0
    max_new_tokens: int = 0
    total_latency_in_engine: float = 0.0
    waiting_latency: float = 0.0

    def update_metrics(
        self,
        tokenizer,
    ):
        # In simulation this will be set
        if self.output_len is None:
            self.output_len = len(tokenizer(self.generated_text).input_ids)
        # print(self.output_len, self.generated_text, self.success, self.error)
        if self.output_len > 1:
            self.tpot = (self.request_latency - self.ttft) / (self.output_len - 1)
        if self.request_latency:
            self.prefill_decode_ratio = self.ttft / self.request_latency
            if self.output_len:
                self.normalized_latency = self.request_latency / self.output_len

    @property
    def total_tokens(self):
        return self.prompt_len + self.output_len

    @property
    def overall_throughput(self):
        return self.total_tokens / self.request_latency

    def to_json(self):
        return json.dumps(self.__dict__)
    

class OpenAIBaseModel(BaseModel):
    # OpenAI API does not allow extra fields
    model_config = ConfigDict(extra="forbid")

class CustomChatCompletionContentPartParam(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    type: Required[str]
    """The type of the content part."""


ChatCompletionContentPartParam = Union[OpenAIChatCompletionContentPartParam,
                                       CustomChatCompletionContentPartParam]


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""
    role: Required[str]
    """The role of the message's author."""

    content: Union[str, List[ChatCompletionContentPartParam]]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """


ChatCompletionMessageParam = Union[OpenAIChatCompletionMessageParam,
                                   CustomChatCompletionMessageParam]
class ChatCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    #response_format: Optional[ResponseFormat] = None
    #seed: Optional[int] = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    #stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    #tools: Optional[List[ChatCompletionToolsParam]] = None
    # tool_choice: Optional[Union[Literal["none"], ChatCompletionNamedToolChoiceParam]] = "none"
    user: Optional[str] = None

    # doc: begin-chat-completion-sampling-params
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    prompt_token_ids: Optional[List[int]] = Field(default_factory=list)
    prompt: Optional[str] = None
    need_cache: Optional[bool] = False
    # truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    # doc: end-chat-completion-sampling-params

    # doc: begin-chat-completion-extra-params
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    documents: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=
        ("A list of dicts representing documents that will be accessible to "
         "the model if it is performing RAG (retrieval-augmented generation)."
         " If the template does not support RAG, this argument will have no "
         "effect. We recommend that each document should be a dict containing "
         "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "If this is not passed, the model's default chat template will be "
            "used instead."),
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))

    # doc: end-chat-completion-extra-params

    # def to_sampling_params(
    #         self, tokenizer: PreTrainedTokenizer,
    #         guided_decode_logits_processor: Optional[LogitsProcessor],
    #         default_max_tokens: int) -> SamplingParams:
    #     max_tokens = self.max_tokens
    #     if max_tokens is None:
    #         max_tokens = default_max_tokens

    #     # We now allow logprobs being true without top_logrobs.
    #     logits_processors = get_logits_processors(
    #         logit_bias=self.logit_bias,
    #         allowed_token_ids=None,
    #         tokenizer=tokenizer,
    #     )
    #     if guided_decode_logits_processor:
    #         logits_processors.append(guided_decode_logits_processor)

    #     return SamplingParams(
    #         n=self.n,
    #         best_of=self.best_of,
    #         presence_penalty=self.presence_penalty,
    #         frequency_penalty=self.frequency_penalty,
    #         repetition_penalty=self.repetition_penalty,
    #         temperature=self.temperature,
    #         top_p=self.top_p,
    #         top_k=self.top_k,
    #         min_p=self.min_p,
    #         seed=self.seed,
    #         stop=self.stop,
    #         stop_token_ids=self.stop_token_ids,
    #         logprobs=self.top_logprobs if self.logprobs else None,
    #         prompt_logprobs=self.top_logprobs if self.echo else None,
    #         ignore_eos=self.ignore_eos,
    #         max_tokens=max_tokens,
    #         min_tokens=self.min_tokens,
    #         use_beam_search=self.use_beam_search,
    #         early_stopping=self.early_stopping,
    #         skip_special_tokens=self.skip_special_tokens,
    #         spaces_between_special_tokens=self.spaces_between_special_tokens,
    #         include_stop_str_in_output=self.include_stop_str_in_output,
    #         length_penalty=self.length_penalty,
    #         logits_processors=logits_processors,
    #         truncate_prompt_tokens=self.truncate_prompt_tokens,
    #     )

    @model_validator(mode='before')
    @classmethod
    def validate_stream_options(cls, values):
        if (values.get('stream_options') is not None
                and not values.get('stream')):
            raise ValueError(
                "stream_options can only be set if stream is true")
        return values

    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        # you can only use one kind of guided decoding
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        # you can only either use guided decoding or tools, not both
        if guide_count > 1 and "tool_choice" in data and data[
                "tool_choice"] != "none":
            raise ValueError(
                "You can only either use guided decoding or tools, not both.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_tool_choice(cls, data):
        if "tool_choice" in data and data["tool_choice"] != "none":
            if not isinstance(data["tool_choice"], dict):
                raise ValueError("Currently only named tools are supported.")
            if "tools" not in data or data["tools"] is None:
                raise ValueError(
                    "When using `tool_choice`, `tools` must be set.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if "top_logprobs" in data and data["top_logprobs"] is not None:
            if "logprobs" not in data or data["logprobs"] is False:
                raise ValueError(
                    "when using `top_logprobs`, `logprobs` must be set to true."
                )
            elif data["top_logprobs"] < 0:
                raise ValueError(
                    "`top_logprobs` must be a value a positive value.")
        return data


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

@final  # So that it should be compatible with Dict[str, str]
class ConversationMessage(TypedDict):
    role: str
    content: str
@dataclass(frozen=True)
class ChatMessageParseResult:
    messages: List[ConversationMessage]

def _parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
) -> ChatMessageParseResult:
    texts: List[str] = []
   
    for part in parts:
        part_type = part["type"]
        if part_type == "text":
            text = cast(ChatCompletionContentPartTextParam, part)["text"]
            texts.append(text)
        else:
            raise NotImplementedError(f"Unknown part type: {part_type}")

    text_prompt = "\n".join(texts)

    

    messages = [ConversationMessage(role=role, content=text_prompt)]

    return ChatMessageParseResult(messages=messages)


def parse_chat_message_content(
    message: ChatCompletionMessageParam
) -> ChatMessageParseResult:
    role = message["role"]
    content = message.get("content")
    #print("I am here", message)

    if content is None:
        return ChatMessageParseResult(messages=[])
    if isinstance(content, str):
        messages = [ConversationMessage(role=role, content=content)]
        return ChatMessageParseResult(messages=messages)

    return _parse_chat_message_content_parts(role, content)


AnyRequest = Union[ChatCompletionRequest]

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
class TextTokensPrompt(TypedDict):
    prompt: str
    prompt_token_ids: List[int]
    


def get_config(model: str,
               trust_remote_code: bool,
               revision: Optional[str] = None,
               code_revision: Optional[str] = None,
               rope_scaling: Optional[dict] = None,
               rope_theta: Optional[float] = None) -> PretrainedConfig:
    try:
        config = AutoConfig.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision)
    except ValueError as e:
        if (not trust_remote_code and
                "requires you to execute the configuration file" in str(e)):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    # if config.model_type in _CONFIG_REGISTRY:
    #     config_class = _CONFIG_REGISTRY[config.model_type]
    #     config = config_class.from_pretrained(model,
    #                                           revision=revision,
    #                                           code_revision=code_revision)
    for key, value in [("rope_scaling", rope_scaling),
                       ("rope_theta", rope_theta)]:
        if value is not None:
            # logger.info("Updating %s from %r to %r", key,
            #             getattr(config, key, None), value)
            config.update({key: value})
    return config


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
        No op for pure text models.
    """
    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        return config.text_config
    else:
        return config
def _get_and_verify_max_len(
        hf_config: PretrainedConfig,
        max_model_len: Optional[int],
        disable_sliding_window: bool,
        sliding_window_len: Optional[int],
    ) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys.
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(hf_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len \
                else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)

    # If sliding window is manually disabled, max_length should be less
    # than the sliding window length in the model config.
    if disable_sliding_window and sliding_window_len is not None:
        max_len_key = "sliding_window" \
            if sliding_window_len < derived_max_model_len else max_len_key
        derived_max_model_len = min(derived_max_model_len, sliding_window_len)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        default_max_len = 2048
        # logger.warning(
        #     "The model's config.json does not contain any of the following "
        #     "keys to determine the original maximum length of the model: "
        #     "%s. Assuming the model's maximum length is %d.", possible_keys,
        #     default_max_len)
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        if "type" in rope_scaling:
            rope_type = rope_scaling["type"]
        elif "rope_type" in rope_scaling:
            rope_type = rope_scaling["rope_type"]
        else:
            raise ValueError(
                "rope_scaling must have a 'type' or 'rope_type' key.")

        # The correct one should be "longrope", kept "su" here
        # to be backward compatible
        if rope_type not in ("su", "longrope", "llama3"):
            if disable_sliding_window:
                # TODO(robertgshaw): Find a model that supports rope_scaling
                # with sliding window to see if this case should be allowed.
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "with rope_scaling. Please raise an issue so we can "
                    "investigate.")

            assert "factor" in rope_scaling
            scaling_factor = rope_scaling["factor"]
            if rope_type == "yarn":
                derived_max_model_len = rope_scaling[
                    "original_max_position_embeddings"]
            derived_max_model_len *= scaling_factor

    # If the user specified a max length, make sure it is smaller than the
    # derived length from the HF model config.
    if max_model_len is None:
        max_model_len = int(derived_max_model_len)
    elif max_model_len > derived_max_model_len:
        # Some models might have a separate key for specifying model_max_length
        # that will be bigger than derived_max_model_len. We compare user input
        # with model_max_length and allow this override when it's smaller.
        model_max_length = getattr(hf_config, "model_max_length", None)
        if model_max_length is not None and max_model_len <= model_max_length:
            if disable_sliding_window:
                # TODO(robertgshaw): Find a model that has model_max_length
                # with sliding window to see if this case should be allowed.
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "model_max_length in the config. Please raise an issue "
                    "so we can investigate.")
            pass
        else:
            raise ValueError(
                f"User-specified max_model_len ({max_model_len}) is greater "
                "than the derived max_model_len "
                f"({max_len_key}={derived_max_model_len} or model_max_length="
                f"{model_max_length} in model's config.json). This may lead "
                "to incorrect model outputs or CUDA errors. Make sure the "
                "value is correct and within the model context size.")
    return int(max_model_len)    
class ModelConfig:
    def __init__(self,model,disable_sliding_window: bool = False):
        self.model = model
        self.disable_sliding_window = disable_sliding_window
        self.hf_config = get_config(self.model,True)
        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            max_model_len = None,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window_len=self.get_hf_config_sliding_window())
    def get_hf_config_sliding_window(self) -> Optional[int]:
        """Get the sliding window size, or None if disabled."""

        # Some models, like Qwen2 and Qwen1.5, use `use_sliding_window` in
        # addition to sliding window size. We check if that field is present
        # and if it's False, return None.
        if (hasattr(self.hf_text_config, "use_sliding_window")
                and not self.hf_text_config.use_sliding_window):
            return None
        return getattr(self.hf_text_config, "sliding_window", None)
    def tokenize_prompt_input(
        self,
        request: AnyRequest,
        tokenizer: AnyTokenizer,
        prompt_input: Union[str, List[int]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> TextTokensPrompt:
        """
        A simpler implementation of :meth:`_tokenize_prompt_input_or_inputs`
        that assumes single input.
        """
        return next(
            self._tokenize_prompt_inputs(
                request,
                tokenizer,
                [prompt_input],
                truncate_prompt_tokens=truncate_prompt_tokens,
                add_special_tokens=add_special_tokens,
            ))
    
    def _validate_input(
        self,
        request: AnyRequest,
        input_ids: List[int],
        input_text: str,
    ) -> TextTokensPrompt:
        token_num = len(input_ids)

        if request.max_tokens is None:
            if token_num >= self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the messages, "
                    f"Please reduce the length of the messages.")
            request.max_tokens = self.max_model_len - token_num

        if token_num + request.max_tokens > self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, you requested "
                f"{request.max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{request.max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion.")

        return TextTokensPrompt(prompt=input_text, prompt_token_ids=input_ids)
    def _normalize_prompt_text_to_input(
        self,
        request: AnyRequest,
        tokenizer: AnyTokenizer,
        prompt: str,
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
        add_special_tokens: bool,
    ) -> TextTokensPrompt:
        if truncate_prompt_tokens is None:
            encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
        else:
            encoded = tokenizer(prompt,
                                add_special_tokens=add_special_tokens,
                                truncation=True,
                                max_length=truncate_prompt_tokens)

        input_ids = encoded.input_ids

        input_text = prompt

        return self._validate_input(request, input_ids, input_text)

    def _normalize_prompt_tokens_to_input(
        self,
        request: AnyRequest,
        tokenizer: AnyTokenizer,
        prompt_ids: List[int],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
    ) -> TextTokensPrompt:
        if truncate_prompt_tokens is None:
            input_ids = prompt_ids
        else:
            input_ids = prompt_ids[-truncate_prompt_tokens:]

        input_text = tokenizer.decode(input_ids)

        return self._validate_input(request, input_ids, input_text)
    def _tokenize_prompt_inputs(
        self,
        request: AnyRequest,
        tokenizer: AnyTokenizer,
        prompt_inputs: Iterable[Union[str, List[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> Iterator[TextTokensPrompt]:
        """
        A simpler implementation of :meth:`_tokenize_prompt_input_or_inputs`
        that assumes multiple inputs.
        """
        for text in prompt_inputs:
            if isinstance(text, str):
                yield self._normalize_prompt_text_to_input(
                    request=request,
                    tokenizer=tokenizer,
                    prompt=text,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=add_special_tokens,
                )
            else:
                yield self._normalize_prompt_tokens_to_input(
                    request,
                    tokenizer,
                    prompt_ids=text,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )