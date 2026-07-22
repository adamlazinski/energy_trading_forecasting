# What this project is, in plain English

*A layman's tour of the whole thing — no finance or machine-learning background
needed. For the technical version see [`README.md`](README.md) and the full log
in [`RESEARCH_LOG.md`](RESEARCH_LOG.md).*

---

## The one-paragraph version

Electricity has to be produced at the exact instant it's used — you can't easily
store it. So every power grid runs a last-minute "balancing" market to fix the
tiny mismatches between how much electricity was planned and how much actually
flowed. In 2024 Poland rewrote the rules of its balancing market. I spent this
project figuring out three things about the new market: **(1)** can you predict
its prices, **(2)** can you make money trading around it, and **(3)** what is a
good prediction actually worth to someone who owns a big battery? The short
answers: **yes you can predict it, no you can't really trade it, and a prediction
is worth a modest but real amount to a battery.** The interesting part is *how*
carefully I had to work to trust those answers — because it's very easy to fool
yourself in markets like this, and a lot of my work was catching myself doing it.

---

## Background: what is a "balancing market"?

Think of the national grid as a giant see-saw that must stay perfectly level at
all times. On one side is electricity being generated; on the other, electricity
being consumed. If the two don't match — a power plant trips, the wind drops,
everyone switches the kettle on at half-time — the grid operator has to top up or
soak up the difference within seconds to minutes.

To do that, the operator keeps a **merit-order list** of power plants and
batteries that have offered to ramp up or down on command, cheapest first. When
it needs energy, it activates the cheapest offers first and works up the list.
The price of the *last* (most expensive) offer it has to use sets the
**imbalance price** — in Poland this is called **CEN**. If you were the party who
caused the mismatch, that's the price you pay (or get paid). It's essentially
"the price of being wrong about your own electricity plan."

That price is wild. Most of the time it's a few hundred złoty. Occasionally — when
the grid is stressed — it spikes to thousands, or even goes *negative* (meaning
you get *paid* to consume electricity, because there's a glut). Predicting those
swings is the whole game.

## What changed in 2024, and why it made this interesting

Before June 2024, Poland priced imbalances one way; after **14 June 2024** it
switched to a **single price**, settled in **15-minute blocks**, with the grid
operator centrally deciding which offers to activate. This is the direction the
entire European Union is heading. Poland got there early — which means there was
essentially **no published research** on how this new-style market behaves. That
gap is the opportunity: I could be one of the first to study it seriously.

There was one catch that shaped everything I did. **The official imbalance price
for today isn't published until around 2 p.m. tomorrow.** So if you're trying to
make a decision *today*, the most recent *confirmed* price you're allowed to know
is from three days ago. If you forget that — if you let your model peek at
yesterday's price, which in real life wouldn't exist yet — your results look
amazing and are completely fake. This "no peeking at information you wouldn't
actually have" rule is the backbone of the whole project.

## What I actually built

Everything runs on **free, public data** — no expensive market-data
subscriptions. Roughly **150 million rows** of it: prices, grid conditions,
weather, and the full list of every balancing offer (about 67 million of those
alone). I pulled it from the Polish grid operator's public feeds, the Polish power
exchange, the pan-European transparency platform, and free weather services.

On top of that data I built four things:

1. **A price predictor.** Not a single-number guess but a *probabilistic* one —
   it outputs a range ("most likely around 400, but here's the 10%-to-90%
   spread"), which is far more useful than a point estimate for something as jumpy
   as this price.

2. **A "market efficiency" tester.** For every pair of related markets (day-ahead,
   three intraday auctions, and the balancing price), it checks: could you have
   reliably bought in one and sold in the other for a profit? With realistic
   trading costs on both ends, and — crucially — only using information you'd
   genuinely have had at the moment you'd need to act.

3. **A battery model.** A detailed simulation of how a grid-scale battery earns
   money in this market, so I could ask: does my price prediction actually *help*
   a battery make more money, and if so, how much?

4. **Live infrastructure.** Two programs that run automatically every day on a
   Mac: one snapshots the live grid data every 15 minutes (because the official
   record gets quietly *revised* after the fact, and I want proof of what was
   really visible at each moment), and one publishes a fresh forecast every
   morning *before* the real price is known — building an honest, time-stamped
   track record I can't fudge later.

## What I found

### Finding 1 — Yes, the price is predictable

My forecaster reliably beats the obvious benchmarks (like "assume today looks like
yesterday"). More interesting was *why* it works: the fancy choice of algorithm
barely mattered. What mattered was **blending several different mediocre models
together** — their mistakes cancel out. The predictable part of the price mostly
comes from things everyone already knows (the weather forecast, the demand plan),
so there's a ceiling on how good any predictor can get. I measured that ceiling
and got most of the way to it.

I also built a specialist detector just for the rare, dramatic events — the price
**spikes** and the **negative** prices — because those are where a normal
predictor is weakest and where the money and the grid stress live. That detector
finds spikes about **18 times better** than a naive baseline: 7 out of 10 real
spikes land in the top 10% of its warnings.

### Finding 2 — No, you can't really trade it

This is the finding I'm most proud of, because it's a *negative* result that took
real discipline to reach honestly.

Every profitable-looking trade I found **fell apart** the moment I insisted on
realism. A classic example: a strategy looked like it earned steady profits — until
I noticed it was "buying" in a market that had actually closed a full day before
the moment my signal appeared. You can't buy in a market that's already shut. When
I re-ran it using only markets I could *actually* trade at that moment, the profit
evaporated. The deep reason is that these markets are **efficient**: by the time
you're allowed to act, everyone else has already acted on the same public
information, so there's nothing left to capture.

My single best trading signal — reconstructed from how weather forecasts get
*revised* between model runs — genuinely worked and looked great on paper. **I
killed it myself**, because when I broke the profits down quarter by quarter, they
all came from just two quarters out of seven. That's not a strategy, that's luck
wearing a costume. I'd rather show that judgment than a fragile backtest.

### Finding 3 — A forecast is worth a modest, real amount to a battery

Since the value clearly wasn't in trading, I asked where it *is*. Answer: physical
assets, specifically big batteries.

A grid-scale battery earns around **3 million złoty per megawatt per year** in this
market — but about **90% of that is just a retainer**: the grid pays the battery to
*stand by* ready to help, whether or not it's ever called. Only a sliver comes from
cleverly timing energy trades, which is the only part a forecast can improve.

So I tested, exhaustively, whether a smarter forecast lets the battery earn more:
- Timing *when* to top the battery back up: **yes, +38,000/year, and reliably
  positive every single quarter** — small, but real and dependable.
- Everything more ambitious — pre-positioning the battery for predicted spikes,
  choosing between different grid services, mixing services to smooth income —
  **kept collapsing back to a boringly simple rule**: just always sell the one
  best service and hold steady.

The reason is always the same: the standby retainer is *so* dominant (about 15×
bigger than the trading opportunity) that being clever about the small part isn't
worth disrupting the big part. **And I quantified exactly how much that retainer
would have to shrink before cleverness starts to pay** — so the day the market
changes, the tools are ready.

## The thread running through all of it

If there's one theme, it's **intellectual honesty as a method**, not a virtue.
Markets and backtests are full of traps that make bad ideas look brilliant. So I
built the whole project around rules designed to *catch myself*: never use
information I wouldn't really have had; always include trading costs; never trust a
result that only works in some quarters; and keep the record of every idea I had to
retract. Several promising results got retired by their own audits — and I kept the
retractions in the log on purpose. In a young, barely-studied market, knowing which
exciting results are *fake* is just as valuable as finding the real ones.

## Where it stands

The forecasters, the efficiency tests, and the battery model are all built,
documented, and reproducible. The live forecaster and data-collector have been
running since July 2026, quietly building a real, time-stamped track record. The
next step is writing it up as what would be one of the first academic studies of
this new style of European electricity market — the raw material is all here.

---

*Findings are logged in detail as F1–F35 in [`RESEARCH_LOG.md`](RESEARCH_LOG.md);
the survey of how this sits against existing research is in
[`LITERATURE.md`](LITERATURE.md).*
