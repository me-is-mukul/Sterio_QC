# Inference

Default: **Ollama** serving **qwen3-vl:4b**, locally or through an ngrok tunnel.
Optional alternative: an **OpenAI-compatible API** (your own vLLM server or a hosted
provider), e.g. for **Qwen3.6-35B-A3B**, which is too large for a 6 GB laptop GPU.

## Choosing a provider (`.env`)

```ini
INFERENCE_PROVIDER=ollama               # default; or: openai_compatible

# openai_compatible  (vLLM or hosted)
VLLM_BASE_URL=http://<gpu-host>:8000/v1  # API root INCLUDING /v1 (hosted: e.g. https://api.deepinfra.com/v1/openai)
VLLM_MODEL=Qwen/Qwen3.6-35B-A3B          # exact id from GET <base>/models
VLLM_API_KEY=<token>                     # vLLM --api-key value, or the host's key

# ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-vl:4b
OLLAMA_API_KEY=
```
Only the selected provider's variables are applied (`core/config.py::ENV_OVERRIDES_BY_PROVIDER`).
API keys are read only by the provider factory and never appear in configs,
manifests, logs or outputs. A test run confirmed the key is absent from every
output file.

### Why not on the laptop
`qwen3.6:35b` is 23 GB even at 4-bit (BF16 is about 72 GB). The RTX 4050 laptop has
6 GB VRAM + 15 GB RAM, so it can't hold the model. It needs a large GPU server or a
hosted API.

## Provider interface (`inference/base.py`)

```python
class InferenceProvider(ABC):
    name: str
    def model_info(self) -> ModelInfo            # checks endpoint + model; identity, runtime version
    def generate(self, ProviderRequest) -> ProviderResponse
    def describe(self) -> dict                   # non-secret description (host, model, think, authenticated)
```
Implementations: `OpenAICompatibleProvider` (`openai_compatible.py`) and
`OllamaProvider` (`ollama_client.py`). Both use `http_json.JsonHttpClient`, which
retries connection errors, timeouts and HTTP 408/425/429/5xx with backoff, and
raises clear errors for non-JSON bodies such as tunnel error pages.
`orchestration/stages/pairs.py::create_provider` is the only place that names them.

## OpenAI-compatible (`inference/openai_compatible.py`)

* `GET {base}/models`: the model must be listed. OpenAI APIs expose no weight digest,
  so the model identity is `openai-compat:` + a hash of the model `id` and `root`
  (the checkpoint path that vLLM reports). This identity is part of the cache key.
* `GET {server}/version` (vLLM) is recorded as the runtime version (optional).
* `POST {base}/chat/completions`:
  * system prompt + user message with text and **two `image_url` parts** (base64 JPEG
    data URLs: image 1 = left, image 2 = right);
  * `response_format: {type: json_schema, json_schema: {name, schema}}` for structured output;
  * `chat_template_kwargs: {enable_thinking: false}` when `inference.think: false`
    (Qwen3 template; set `think: null` if a hosted API rejects this field);
  * `max_tokens = inference.max_output_tokens` (1024); a `finish_reason: length` is logged.
* If `content` is empty but `reasoning_content` holds the answer (vLLM reasoning
  parser), that is used. It is still validated.

### Serving Qwen3.6-35B-A3B with vLLM
```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 2}' \
  --api-key "$VLLM_API_KEY"
```
* `--limit-mm-per-prompt` must allow **2 images**: each request carries a stereo pair.
* Memory: the BF16 weights are about 72 GB, so plan for one 80 GB GPU (or several
  smaller ones with `--tensor-parallel-size`), or use a quantized checkpoint.
* Docker: run the `vllm/vllm-openai` image with the same arguments (not part of the default compose file).
* Expose it to the pipeline machine with a private network/VPN or a tunnel. Always set
  `--api-key` if it is reachable from the internet.

## Ollama (`inference/ollama_client.py`)
`GET /api/version`, `GET /api/tags` (presence + **digest**), and `POST /api/chat` with
both images, a JSON-schema `format`, `think`, `keep_alive` and
`options.num_predict = max_output_tokens`. If `content` is empty, the answer is taken
from `message.thinking` (seen with Qwen3-VL on Ollama 0.32).

## Prompt (`inference/prompts.py`)

Version `hoi-stereo-v1` (selected by `inference.prompt_version`). It states:
* image 1 = LEFT, image 2 = RIGHT, same instant, one scene;
* never double-count an interaction seen in both views;
* the RELEVANT / WASTE / UNCERTAIN definitions;
* a stationary hand holding an object is RELEVANT;
* prefer UNCERTAIN when unsure.

The same prompt is used for every provider and model. To change the text, add a new
version.

## Validation (`inference/qwen_classifier.py`)

Model output is untrusted:
* JSON is extracted and validated field by field.
* A missing or invalid label → UNCERTAIN (`invalid_response`), not cached.
* A request failure → UNCERTAIN or UNAVAILABLE (`failed`), **never WASTE**.
* Model-stated confidence is not used.

## Cache

`inference/cache/<sha256>.json`. The key covers: left/right image hashes, provider,
model, model identity/digest, prompt version, temperature, think, max_output_tokens,
and cache format. Switching model or provider is therefore a clean cache miss. Only
valid results are cached.

## Verification so far

| Path | Verified with |
|---|---|
| OpenAI-compatible | Full pipeline run against a stand-in server enforcing the request format (2 images, JSON schema, thinking flag, Bearer auth); wrong-key failure; `reasoning_content` fallback; no key leakage |
| Ollama | Full runs against a stand-in server; the real `qwen3-vl:4b` connected (answers parsed after the `thinking` fallback), but no complete real run |

Qwen3.6-35B-A3B itself has not been called yet. Its classification quality is
unmeasured until a real endpoint is configured.
