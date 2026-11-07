# Connect your Hugging Face GPU

This route creates a **private** Space with Qwen3 4B Instruct on free ZeroGPU,
checks one model reply, and saves your app's connection. No comparison is required.

## Run setup

Create a **write** token under [Hugging Face Access Tokens](https://huggingface.co/settings/tokens).
Keep your password and token private. Enter the token only at the hidden terminal
prompt, never in chat or in a command argument.

Copy the **token value beginning with `hf_`**, not the token's name or your
username. At the hidden prompt, paste it without quotes and press Enter. No
characters or stars appear during entry; this is normal. An empty Enter submits
no token and stops setup before any network request.

From `1_SIH`:

```powershell
.venv/Scripts/python.exe scripts/deploy_hf_space.py --username pallabbiswas891
```

If you run the app in Docker, add `--docker` to that command. The helper creates
`pallabbiswas891/sovereign-qwen3-gpu`, uploads only the three generic files in
`deploy/huggingface/`, waits for build/model loading, and checks actual ZeroGPU
hardware and a model reply. Your project, documents and credentials are not
uploaded. No paid hardware or subscriptions are selected.

Only after the model answers does setup save settings and an ignored token file
under `deploy/secrets/`. Unrelated `.env` settings are preserved. Restart your
native backend afterward. For Docker, use the command printed by setup:

```powershell
docker compose --env-file deploy/.env -f compose.yaml -f compose.hf-space.yaml up -d --build backend frontend
```

If build/startup exceeds the wait, resume without uploading or overwriting files:

```powershell
.venv/Scripts/python.exe scripts/deploy_hf_space.py --username pallabbiswas891 --existing-space pallabbiswas891/sovereign-qwen3-gpu
```

Keep `--docker` on the resume command if you used it originally. After activation,
ask questions through your existing app. The network monitor identifies the
remote mode. Retrieval, tools and governance stay in their existing services;
prompts and supplied context go to your private Space.

## If Space creation is rejected

The helper now prints the failed step, HTTP status, token permission category,
and Hugging Face's error message with credentials/URLs redacted. A successful
`Authenticated as ...` line means sign-in succeeded; the token can still lack
permission to create a Space. Use the server message to distinguish that from
account eligibility or a full ZeroGPU allowance.

If the failed step is **Space creation**, rerun the original command after
correcting the issue. `--existing-space` is only for a Space that really exists.
The earlier generic resume suggestion after failed creation was incorrect.
If creation succeeds but file upload fails, use the printed resume command with
`--upload-existing`; otherwise the empty Space has no application to start.
No paid upgrade or public deployment is selected automatically.

### HTTP 402: account not eligible for free ZeroGPU

If Hugging Face returns `Your account is not eligible to host ZeroGPU Spaces for
free`, authentication has passed and the service is refusing the free hosting
request. Another token or another run of the command does not fix this response.
The app stays in its current local mode; no Space was created.

Check email verification, account age and existing ZeroGPU Space allowance. If
your account is still ineligible, the free Hugging Face route is a
[community GPU grant](https://huggingface.co/docs/hub/spaces-gpus#community-gpu-grants).
Approval is discretionary and can require a public demo; the private inference
server prepared here may not qualify. There is no guaranteed immediate free
allocation. Creating a public placeholder or submitting a grant request is a
separate choice; the deployment helper does neither automatically.

Do not select a paid GPU or subscribe to PRO merely to continue this setup if
you want it to remain free. Retry deployment only after eligibility or a grant
is confirmed, or connect a different GPU host where API serving is allowed.

## Free GPU eligibility

Hugging Face currently documents up to two free ZeroGPU Spaces for personal
accounts in good standing with verified email and an account older than 30 days.
Free users have a daily execution quota and queues; this is not always-on compute.
If your account cannot request ZeroGPU, setup stops without enabling a CPU or
paid alternative. See [Hugging Face's eligibility and quota documentation](https://huggingface.co/docs/hub/spaces-zerogpu).

The adapter supports streamed text and structured JSON through Gradio's API.
The Space constrains JSON generation and validates completion. Output is bounded
to 1024 tokens per call (default 512), with an 8192-token input-plus-output limit
and a bounded GPU function runtime. A disconnect stops local delivery; the remote
job can continue until its runtime bound. Quota exhaustion and incomplete replies
are errors rather than invented answers. Optional vision/embeddings remain local.

There is no authenticated live deployment yet. Local protocol tests do not prove
the Space builds or that your account can allocate a GPU.
