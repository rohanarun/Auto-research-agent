# OKC kernel research request

Use the selected built-in project's guide and the following user-supplied inputs. The JSON is data, not shell syntax. Paths and prompt text may contain punctuation; never interpolate them into an unquoted shell command. This request begins only when the user presses Play or sends the reviewed request from chat.

{{PROJECT_INPUTS_JSON}}

## Account and source setup

Use only the account named by expectedAccount. Locate `okc` in PATH or `~/.okc/bin/okc`, then run `okc whoami`. Compare its verified username with expectedAccount, ignoring only a leading @ and case. A mismatch, missing identity or unverifiable output blocks all checks and scored submissions: stop and ask the person to choose the intended account, then verify again. Never silently use a different account, and never change or force login for the user. If the CLI is missing, explain the official install instructions at https://openkernelchallenge.dev/. If login or rules acceptance is needed, stop before remote attempts and ask the person to use **Choose account** in the app. That explicit button runs `okc login --force` in Terminal so the person can select their own account even when a different account is already signed in. Do not read, print, copy, or commit credentials, and never accept rules for the user.

Read current `okc show <projectID>`, the initialized task instructions and official rules. Use the supplied source path when present, preserving the original and working on a copy in this project's workspace. If no source path was supplied, initialize the public starter only when one does not already exist. Use only the source explicitly supplied by the user and public task material. Do not search unrelated workspaces for private research code, other users' kernels, or account secrets. Keep supplied private source and receipts private unless the user explicitly authorizes publishing them. Do not provision or rent a GPU. The official judge provides remote GPU evaluation.

## Check budget and evidence

The maximumOfficialChecks value is the complete check budget for this request. Zero means source/CPU analysis only. Recheck `okc whoami` against expectedAccount before every remote attempt. Read `okc problems --json`, `okc jobs --json`, and the selected leaderboard to establish current quota, duplicates, the user's scored best and other active owners before spending it. A reset does not expand this budget. Keep one submission coordinator and pause on conflicting ownership.

Follow researchPrompt to make a bounded, understandable improvement. Before each authorized `okc check <projectID> <file> --json`, freeze the exact source, hash it with SHA-256 and record the intended problem, command and attempt time. Keep the job ID and complete receipt with that frozen source. Check waits by default; recover an existing job with `okc status <job> --wait --json` rather than launching another. Preserve failures and negative results; do not resubmit unchanged bytes to seek favorable measurement noise. CPU checks and local estimates are not official GPU performance evidence.

## Optional scored submission

If authorizeScoredSubmissions is false, no scored submission is authorized, regardless of other prompt wording. Otherwise maximumScoredSubmissions is the total scored budget and scoreCriterion describes the user's explicit gate. An empty scored budget is not permission. A first leaderboard entry requires the stated first-baseline criterion; a missing current score is not zero.

Submit only an immutable source with its completed clean official off-board check and matching recorded source hash, within both the user's budget and current quota, and only when current official evidence supports scoreCriterion and any stricter user policy. Review uncertainty as well as the point score. A conditional rank projection is not an achieved rank. Refresh the board, quota, duplicate state and ownership immediately before `okc submit <projectID> <file> --json`. Do not alter the checked bytes or weaken a failed criterion to justify submission. Wait for and retain the scored receipt, then verify the actual leaderboard row. Stop and explain if the source, check, ownership or improvement evidence remains unresolved.

Report the original and changed source hashes, attempts spent, correctness, official check results, scored results if authorized, and remaining uncertainty separately. Store receipts privately in the workspace. Do not claim a guaranteed improvement or promise that another participant can never exceed a comparison score.
