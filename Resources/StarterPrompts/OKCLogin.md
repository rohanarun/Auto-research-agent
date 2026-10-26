# Sign in to Open Kernel Challenge

1. Install the official OKC CLI using the instructions at https://openkernelchallenge.dev/. The installer is published at https://openkernelchallenge.dev/install.sh. Review and run the installation yourself; this app does not execute it automatically.
2. Click **Sign in to OKC**. A Terminal window opens in this project's external-SSD workspace and runs `okc login`. If `okc` is not on PATH, it uses `~/.okc/bin/okc`. Complete the login and personally accept any challenge rules shown by the official CLI.
3. The same Terminal window runs `okc whoami` after successful login so you can verify the active account. The app does not capture this output or your token. It does not force a new login or sign you out of an existing account.
4. Return to this project, choose a kernel source or leave it blank for the public starter, enter your check budget, and describe the research you want. Scored submissions are off by default. To allow them, enable the option and supply both a scored budget and a first-baseline or improvement criterion.
5. Choose **Review request**, review or edit it, then choose **Use in chat**. Nothing is submitted until you send the prompt and the agent verifies the required evidence. Official checks and public scores are separate actions; both use your current account's quota.

Manual Terminal equivalents:

    okc login
    okc whoami

If the CLI is not on PATH:

    ~/.okc/bin/okc login
    ~/.okc/bin/okc whoami

Never paste an API token into the research prompt. The signed-in CLI manages authentication. To deliberately change an existing account, use the official CLI's account controls yourself; this login button does not force account replacement.
