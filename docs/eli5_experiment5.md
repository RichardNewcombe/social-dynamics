# Experiment 5: The Short Version

## What It Is

A simulated company trying to climb a mountain. Nobody has a map. The mountain has fake peaks that look like the top but aren't.

Each person in the company has a **personality** (who they hang out with) and a **strategy** (where they think the company should go). These can overlap a lot (specialists — "I am what I do") or barely at all (generalists — "my team is separate from my work").

People have different jobs:

- **Researchers** can sort of tell which way is uphill, but some are better than others. Nobody knows who's accurate — it only shows up in results.
- **Engineers** can take big or small steps. They don't pick the direction, they just amplify it.
- **Leaders** are loud. When the team votes on which way to go, leaders' votes count more — whether they're right or not.
- **Visionaries** can faintly see the real summit. They're rare (5–10% of people) and expensive.

Every step, people chat with their teammates (determined by personality similarity), share their noisy guesses about which way is uphill, and the team moves in the averaged direction. Meanwhile, the company is burning money — both from salaries (expensive people cost more) and from the terrain (some paths are pricier than others).

## What Changed in the Simulator

Two things were added to the core simulation engine. Everything else is backward-compatible and the old experiments still work.

**1. People now have roles.** Four new per-particle arrays (researcher noise, engineer step scale, leader influence, visionary weight). Turned on with `use_particle_roles=True`. When off, everyone is identical — same as before.

**2. Personality and strategy are now separate.** A new `strategy` vector per particle, independent of the existing `preferences` vector. A coupling parameter (0 to 1) controls how much they overlap at startup. A new `strategy_step()` method runs after the normal social-dynamics step — it uses the social neighbor graph (who's on your team) to aggregate gradient observations and move in strategy space. Turned on with `strategy_enabled=True`. When off, the simulation is unchanged.

## What It's Shown So Far

| Finding | What it means |
|---|---|
| Generalists (coupling=0) reach the summit more often than specialists (coupling=1) | When your team identity locks your strategy, you get stuck on whatever peak your team landed on. Separating them lets strategy evolve freely. |
| Rare visionaries + generalist structure is the most cost-efficient combo | A few people who can see the summit, embedded in a flexible org, outperform expensive teams of all-stars. |
| Social conformity hurts on rugged landscapes | Herding everyone onto the nearest peak is fast but wrong. The nearest peak usually isn't the highest. |
| Social differentiation avoids traps but can't converge | Pushing people apart prevents getting stuck, but nobody reaches the top either. |
| The knowledge memory field doesn't help yet | Institutional knowledge is currently encoding early noise, not useful information. |

## Gaps

**The landscape is still too easy for visionaries.** A visionary ignores the terrain and walks straight toward the summit. Until the cost ridges or landscape geometry make that straight-line path genuinely punishing, visionaries will always win. The landscape needs narrow corridors or impassable ridges that force detours.

**Cost doesn't constrain behavior.** Money is tracked but never runs out. Adding a hard budget ("you have $X, simulation ends when it's spent") would force real tradeoffs instead of just reporting them.

**Teams aren't forming.** DBSCAN finds zero preference-space clusters in most conditions. Either the social learning rate is too low, the run is too short, or the clustering threshold needs tuning. Without visible teams, the "team structure drives strategy" story is theoretical — the neighbor graph exists but it's not producing coherent groups.

**The visionary rarity constraint needs more exploration.** We've tested 5% and 10% fractions. The interesting question is: what's the minimum number of visionaries an org needs? One? Five? Is there a phase transition?

**Coupling is static.** Real orgs start generalist and specialize over time. A dynamic coupling that increases as fitness improves would model organizational maturation and might reveal when specialization helps vs. hurts.
