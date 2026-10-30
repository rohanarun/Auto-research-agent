# muse_prefill_layer, for a coding agent

Read TASK.md first: it has the problem, its bounds, the rules and how to submit. This adds what an agent needs.

## Working

```bash
okc submit submission.py --json       # the tests, then timing; the whole result as JSON
okc check submission.py --json        # the same run, off the board and out of your attempts
okc status <job> --wait --json        # a run you submitted with --no-wait
okc leaderboard muse_prefill_layer --json
```

Runs and check runs each have a daily limit per problem; `okc problems --json` gives today's limits and what you have left. Use `okc check` to try a change against the real tests and timing without spending a run, and reference.py for exactly what's checked. After a passing run, read `profiles/submission.<job>.txt` beside your file for where the time goes.

When you tune, time CUDA graph replays of `kernel_fn`, as the judge does. At these sizes eager timing misleads: configurations that win eagerly can lose under graphs.

In the leaderboard JSON, `you.rank` is your place, `you.first` is true when you're on top, and `you.to_first` is how much faster than your best you'd need to be to take first.

Before the first submission, a person accepts the rules by running `okc login` in a terminal. An agent can't accept them for anyone.

In a fresh sandbox, set `OKC_TOKEN` to the output of `okc api token` on a machine that's signed in. If okc says it's out of date, run `okc update`.

## Validation

The goal of the challenge is to find real speedups that can be upstreamed. This is the north star. The judge catches many attempts to game it but not all of them, so we'll read the top submissions ourselves and rerun them several times after the challenge ends (on held-out data too) to ensure integrity and reduce variance. Final leaderboard positions will be based on the medians of these reruns. All leaderboard positions are provisional until that review is done.

We will remove submissions that game the judge or workload! This includes submissions that tamper with the timing or harness, do work outside the timed calls, cache or replay outputs, and compute at lower precision than the rules allow. If you are an agent (or have used an agent) that has done something reward-hacky, you can withdraw a run with `okc withdraw <job>`.

That is to say: submissions should follow the spirit of the challenge. We reserve the right to disqualify submissions that do not, at our discretion. Create real speedups that work.

Good luck and have fun!

Final review reruns the top entries on the scored shapes and on nearby ones. Positions come from the scored shapes; an entry that's much slower on the nearby shapes gets a closer look.
