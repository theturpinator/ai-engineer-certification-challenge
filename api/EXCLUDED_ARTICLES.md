# Excluded articles

21 expired sweepstakes/giveaway promo articles are excluded from the index so
readers are never told about contests that ended years ago (spec issue #1,
user story 19). They were removed from `data/articles-clean.csv` during data
prep, and ingestion additionally excludes them by slug (`EXCLUDED_SLUGS` in
`ingest.py`) as a guard in case a promo re-enters a future CSV export.

**Rule:** any article whose `slug` appears in the list below — each is a
time-boxed sweepstakes, raffle, or giveaway promotion, identified by manual
review of the archive.

| Slug | Title |
|---|---|
| `1970-boss-mustang` | 1970 Boss Mustang |
| `5-ways-you-can-win-a-2020-ford-mustang-gt500` | [5] Ways YOU Can Win A 2020 Ford Mustang GT500 |
| `cruise-for-a-cause-sweepstakes-shelby-gt500se` | HOPE EDITION Cruise For A Cause Sweepstakes Shelby GT500SE |
| `customized-convenience` | CUSTOMIZED CONVENIENCE |
| `drive-home-hope` | Drive Home Hope |
| `drive-the-dream` | Drive the Dream |
| `fever-dream` | FEVER DREAM |
| `get-ready` | GET READY |
| `jdrf-2024-mustang-dark-horse` | JDRF 2024 MUSTANG DARK HORSE |
| `prize-pony` | Prize Pony |
| `roush-raffle-support-the-preservation-of-henry-fords-home-and-you-might-win-a-special-roush-mustang` | ROUSH RAFFLE Support the preservation of Henry Ford's home and you can win! |
| `shelby-gt500-sweepstakes-helps-veterans-in-need` | GOLDEN TICKETS Shelby GT500 Sweepstakes Helps Veterans In Need |
| `shelby-sweepstakes-win-an-800hp-2021-shelby-gt500se` | SHELBY SWEEPSTAKES Win an 800HP 2021 Shelby GT500SE |
| `ten-for-the-win-more-chances-to-win-great-prizes` | TEN FOR THE WIN More chances to win GREAT prizes. |
| `win-a-1968-shelby-gt500` | FIT FOR A KING Win A 1968 Shelby GT500 KR For As Little As $25 |
| `win-a-2021-mustang-gt-and-ford-performance-parts` | FREE RIDE Win a 2021 Mustang GT and $5,000 worth of Ford Performance Parts |
| `win-a-mustang-mach-e-gt-charging-station` | FREE RIDE You Could Win A Mustang Mach-E GT & Charging Station |
| `win-a-mustang-mach-e-gt-help-joey-loganos-charity` | CHARGED CHARITY Win A Mustang Mach-E GT & Help Joey Logano's Charity |
| `win-a-one-of-a-kind-shelby-snakecharmer-for-just-25` | LUCKY STRIKE You Could Win A One-Of-A-Kind Shelby SnakeCharmer For Just $25 |
| `win-an-800hp-saleen-302-for-as-little-as-25` | BLAZING FURY Win An 800HP Saleen 302 For As Little As $25 |
| `win-this-1969-mach-1-mustang` | WIN This 1969 Mach 1 Mustang! |
