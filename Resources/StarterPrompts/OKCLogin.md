# Choose your Open Kernel Challenge account

1. Install the official OKC CLI using the instructions at https://openkernelchallenge.dev/. The installer is published at https://openkernelchallenge.dev/install.sh. Review and run installation yourself; this app does not execute it automatically.
2. Click **Choose account**. This explicit action opens Terminal in the project's external-SSD workspace and runs `okc login --force`. This starts a fresh OKC login even when the CLI has a cached account, so no manual OKC logout is needed. The browser may still be signed into a different GitHub account: verify the identity shown there, and switch or sign into the intended browser account before authorizing. Complete login and personally review and accept any challenge rules.
3. The Terminal window runs `okc whoami` after successful login. Return to the app and click **Refresh account** to see the current verified username. The app retains only the parsed public username, never a password or API token.
4. Enter **Your expected GitHub/OKC username**. This confirms which account should spend quota; it is not a password field and does not authenticate or switch accounts. Play verifies that the current account matches. A mismatch or unverifiable identity stops the run before the coding agent starts, and the research instructions require another identity check before each remote attempt.
5. Choose a source file or leave it blank for the public starter, enter your check budget, and describe the research you want. Scored submissions are off by default. To allow scores, enable that option and enter both a scored budget and a first-baseline or improvement criterion.
6. Press **Play** to start the visible agent chat using the saved inputs. **Review request** is optional and lets you inspect or edit the generated prompt first. Editing inputs, refreshing your account and logging in do not launch research or spend judge quota.

Manual Terminal equivalents:

    okc login --force
    okc whoami

If the CLI is not on PATH:

    ~/.okc/bin/okc login --force
    ~/.okc/bin/okc whoami

Ordinary `okc login` can reuse a previously signed-in account. Choose account deliberately requests a fresh login; it runs only when you press that button. Agents must never force account changes themselves. Never paste an API token into a prompt. The official CLI manages authentication, and the app never accepts challenge rules on your behalf.
